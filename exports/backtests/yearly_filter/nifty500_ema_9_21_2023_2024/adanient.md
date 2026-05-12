# Adani Enterprises Ltd. (ADANIENT)

## Backtest Summary

- **Window:** 2023-03-14 10:15:00 → 2026-05-08 15:15:00 (5435 bars)
- **Last close:** 2502.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 216 |
| ALERT1 | 147 |
| ALERT2 | 144 |
| ALERT2_SKIP | 66 |
| ALERT3 | 390 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 198 |
| PARTIAL | 20 |
| TARGET_HIT | 17 |
| STOP_HIT | 185 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 221 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 74 / 147
- **Target hits / Stop hits / Partials:** 17 / 184 / 20
- **Avg / median % per leg:** 0.71% / -0.56%
- **Sum % (uncompounded):** 157.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 88 | 23 | 26.1% | 10 | 78 | 0 | 0.49% | 43.1% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.12% | 1.1% |
| BUY @ 3rd Alert (retest2) | 87 | 22 | 25.3% | 10 | 77 | 0 | 0.48% | 42.0% |
| SELL (all) | 133 | 51 | 38.3% | 7 | 106 | 20 | 0.86% | 114.1% |
| SELL @ 2nd Alert (retest1) | 4 | 4 | 100.0% | 0 | 2 | 2 | 2.99% | 12.0% |
| SELL @ 3rd Alert (retest2) | 129 | 47 | 36.4% | 7 | 104 | 18 | 0.79% | 102.2% |
| retest1 (combined) | 5 | 5 | 100.0% | 0 | 3 | 2 | 2.62% | 13.1% |
| retest2 (combined) | 216 | 69 | 31.9% | 17 | 181 | 18 | 0.67% | 144.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-15 10:15:00 | 1839.11 | 1882.57 | 1884.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-16 14:15:00 | 1823.89 | 1849.30 | 1861.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-17 14:15:00 | 1842.02 | 1835.76 | 1846.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-17 15:00:00 | 1842.02 | 1835.76 | 1846.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-17 15:15:00 | 1839.16 | 1836.44 | 1846.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-18 09:15:00 | 1845.03 | 1836.44 | 1846.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-18 09:15:00 | 1844.11 | 1837.97 | 1845.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-18 14:30:00 | 1837.17 | 1840.41 | 1844.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-18 15:00:00 | 1835.57 | 1840.41 | 1844.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-19 09:30:00 | 1831.36 | 1835.83 | 1841.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-19 12:00:00 | 1835.24 | 1835.92 | 1840.72 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-19 12:15:00 | 1840.81 | 1836.90 | 1840.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-19 13:00:00 | 1840.81 | 1836.90 | 1840.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2023-05-19 13:15:00 | 1874.40 | 1844.40 | 1843.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2023-05-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-19 13:15:00 | 1874.40 | 1844.40 | 1843.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 09:15:00 | 2040.09 | 1898.26 | 1869.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-24 11:15:00 | 2384.93 | 2466.97 | 2329.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-05-24 12:00:00 | 2384.93 | 2466.97 | 2329.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 09:15:00 | 2342.28 | 2415.72 | 2354.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-25 10:00:00 | 2342.28 | 2415.72 | 2354.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-25 10:15:00 | 2346.88 | 2401.95 | 2353.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-25 12:15:00 | 2368.45 | 2391.06 | 2353.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-30 15:15:00 | 2419.83 | 2439.80 | 2440.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2023-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-30 15:15:00 | 2419.83 | 2439.80 | 2440.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 09:15:00 | 2384.01 | 2428.64 | 2435.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-31 10:15:00 | 2454.69 | 2433.85 | 2436.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 10:15:00 | 2454.69 | 2433.85 | 2436.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 10:15:00 | 2454.69 | 2433.85 | 2436.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-31 11:00:00 | 2454.69 | 2433.85 | 2436.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2023-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 11:15:00 | 2463.17 | 2439.71 | 2439.33 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2023-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 14:15:00 | 2412.81 | 2434.80 | 2437.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-02 12:15:00 | 2366.95 | 2414.92 | 2424.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-05 09:15:00 | 2425.65 | 2400.88 | 2413.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-05 09:15:00 | 2425.65 | 2400.88 | 2413.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 09:15:00 | 2425.65 | 2400.88 | 2413.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-05 10:00:00 | 2425.65 | 2400.88 | 2413.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 10:15:00 | 2416.64 | 2404.03 | 2413.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-05 12:00:00 | 2398.12 | 2402.85 | 2412.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-09 15:15:00 | 2379.12 | 2371.56 | 2370.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2023-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-09 15:15:00 | 2379.12 | 2371.56 | 2370.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-12 11:15:00 | 2382.70 | 2375.47 | 2372.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-13 12:15:00 | 2383.24 | 2400.53 | 2390.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-13 12:15:00 | 2383.24 | 2400.53 | 2390.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 12:15:00 | 2383.24 | 2400.53 | 2390.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-13 13:00:00 | 2383.24 | 2400.53 | 2390.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-13 13:15:00 | 2389.00 | 2398.22 | 2390.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-14 09:45:00 | 2400.44 | 2396.37 | 2391.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-14 15:15:00 | 2382.02 | 2388.98 | 2389.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2023-06-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-14 15:15:00 | 2382.02 | 2388.98 | 2389.61 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2023-06-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-15 09:15:00 | 2432.34 | 2397.66 | 2393.50 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2023-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-19 12:15:00 | 2368.45 | 2404.55 | 2407.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-19 13:15:00 | 2323.66 | 2388.37 | 2399.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-21 15:15:00 | 2350.95 | 2336.84 | 2349.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-21 15:15:00 | 2350.95 | 2336.84 | 2349.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 15:15:00 | 2350.95 | 2336.84 | 2349.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-22 09:15:00 | 2340.34 | 2336.84 | 2349.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 09:15:00 | 2367.48 | 2342.97 | 2351.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-22 10:00:00 | 2367.48 | 2342.97 | 2351.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 10:15:00 | 2328.70 | 2340.11 | 2349.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-22 11:15:00 | 2318.18 | 2340.11 | 2349.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-06-23 09:15:00 | 2202.27 | 2304.33 | 2327.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-06-26 09:15:00 | 2218.67 | 2204.75 | 2252.88 | SL hit (close>ema200) qty=0.50 sl=2204.75 alert=retest2 |

### Cycle 10 — BUY (started 2023-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 10:15:00 | 2329.72 | 2258.47 | 2250.61 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 12:15:00 | 2304.03 | 2317.95 | 2319.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-10 09:15:00 | 2295.45 | 2309.78 | 2314.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 10:15:00 | 2328.70 | 2313.56 | 2315.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-10 10:15:00 | 2328.70 | 2313.56 | 2315.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 10:15:00 | 2328.70 | 2313.56 | 2315.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-10 11:00:00 | 2328.70 | 2313.56 | 2315.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 11:15:00 | 2316.10 | 2314.07 | 2315.94 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2023-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-10 13:15:00 | 2342.91 | 2321.67 | 2319.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 10:15:00 | 2357.74 | 2337.03 | 2327.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-11 15:15:00 | 2343.34 | 2343.87 | 2335.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-12 09:15:00 | 2333.74 | 2343.87 | 2335.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 09:15:00 | 2325.84 | 2340.26 | 2334.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 09:45:00 | 2332.92 | 2340.26 | 2334.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 10:15:00 | 2328.22 | 2337.85 | 2333.99 | EMA400 retest candle locked (from upside) |

### Cycle 13 — SELL (started 2023-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 13:15:00 | 2319.98 | 2331.51 | 2331.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-12 14:15:00 | 2317.26 | 2328.66 | 2330.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-13 09:15:00 | 2327.88 | 2326.65 | 2329.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-13 09:15:00 | 2327.88 | 2326.65 | 2329.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 09:15:00 | 2327.88 | 2326.65 | 2329.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-13 09:30:00 | 2327.01 | 2326.65 | 2329.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-13 10:15:00 | 2319.01 | 2325.12 | 2328.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-13 12:00:00 | 2311.79 | 2322.46 | 2326.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-13 13:15:00 | 2309.85 | 2320.72 | 2325.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-17 09:15:00 | 2352.31 | 2310.19 | 2310.94 | SL hit (close>static) qty=1.00 sl=2328.75 alert=retest2 |

### Cycle 14 — BUY (started 2023-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 10:15:00 | 2336.90 | 2315.53 | 2313.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-18 09:15:00 | 2360.65 | 2336.88 | 2326.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 14:15:00 | 2341.50 | 2345.54 | 2335.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-18 14:15:00 | 2341.50 | 2345.54 | 2335.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 14:15:00 | 2341.50 | 2345.54 | 2335.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-19 09:15:00 | 2354.20 | 2344.75 | 2336.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-19 09:45:00 | 2352.46 | 2346.71 | 2337.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-19 11:30:00 | 2356.14 | 2348.56 | 2340.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-19 13:30:00 | 2354.39 | 2349.94 | 2342.37 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 09:15:00 | 2335.68 | 2346.88 | 2342.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 09:30:00 | 2332.68 | 2346.88 | 2342.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 10:15:00 | 2340.77 | 2345.66 | 2342.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-20 11:15:00 | 2348.09 | 2345.66 | 2342.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-20 14:00:00 | 2352.41 | 2349.72 | 2345.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-21 09:30:00 | 2354.20 | 2349.33 | 2346.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-24 09:45:00 | 2350.03 | 2352.80 | 2350.70 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 10:15:00 | 2354.10 | 2353.06 | 2351.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 10:30:00 | 2348.82 | 2353.06 | 2351.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 11:15:00 | 2352.36 | 2352.92 | 2351.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-24 11:45:00 | 2345.33 | 2352.92 | 2351.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-24 12:15:00 | 2351.34 | 2352.60 | 2351.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-24 13:45:00 | 2360.50 | 2352.68 | 2351.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-24 14:15:00 | 2342.62 | 2350.67 | 2350.53 | SL hit (close<static) qty=1.00 sl=2348.63 alert=retest2 |

### Cycle 15 — SELL (started 2023-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 15:15:00 | 2342.28 | 2348.99 | 2349.78 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2023-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 09:15:00 | 2357.69 | 2350.73 | 2350.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-25 14:15:00 | 2396.57 | 2360.04 | 2354.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-26 13:15:00 | 2386.58 | 2392.14 | 2376.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-26 14:00:00 | 2386.58 | 2392.14 | 2376.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-26 14:15:00 | 2398.55 | 2393.42 | 2378.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-26 14:45:00 | 2384.30 | 2393.42 | 2378.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 2375.24 | 2389.52 | 2379.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-27 10:00:00 | 2375.24 | 2389.52 | 2379.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 10:15:00 | 2388.81 | 2389.38 | 2380.19 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2023-07-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-27 14:15:00 | 2357.11 | 2373.63 | 2374.96 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-07-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-28 10:15:00 | 2390.94 | 2378.24 | 2376.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-31 09:15:00 | 2412.08 | 2388.70 | 2382.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-01 09:15:00 | 2407.23 | 2408.69 | 2398.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-01 10:00:00 | 2407.23 | 2408.69 | 2398.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 10:15:00 | 2389.54 | 2404.86 | 2397.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 10:45:00 | 2392.54 | 2404.86 | 2397.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 11:15:00 | 2395.02 | 2402.89 | 2397.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 11:30:00 | 2389.93 | 2402.89 | 2397.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 12:15:00 | 2394.09 | 2401.13 | 2396.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-01 12:30:00 | 2387.79 | 2401.13 | 2396.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 2411.01 | 2404.61 | 2400.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-02 10:45:00 | 2434.43 | 2407.19 | 2401.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-02 13:15:00 | 2383.62 | 2402.20 | 2400.70 | SL hit (close<static) qty=1.00 sl=2385.47 alert=retest2 |

### Cycle 19 — SELL (started 2023-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-08 12:15:00 | 2401.51 | 2438.68 | 2443.05 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-08-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-10 09:15:00 | 2463.95 | 2434.28 | 2433.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-10 12:15:00 | 2490.32 | 2454.52 | 2443.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-11 14:15:00 | 2461.52 | 2470.22 | 2460.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 14:15:00 | 2461.52 | 2470.22 | 2460.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 14:15:00 | 2461.52 | 2470.22 | 2460.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-11 15:00:00 | 2461.52 | 2470.22 | 2460.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 15:15:00 | 2460.55 | 2468.29 | 2460.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-14 09:15:00 | 2372.18 | 2468.29 | 2460.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — SELL (started 2023-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 09:15:00 | 2359.58 | 2446.55 | 2451.47 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2023-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 10:15:00 | 2417.94 | 2398.55 | 2397.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-18 11:15:00 | 2529.63 | 2424.77 | 2409.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 10:15:00 | 2585.62 | 2593.18 | 2556.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-23 10:30:00 | 2587.94 | 2593.18 | 2556.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 12:15:00 | 2553.04 | 2584.26 | 2559.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 13:00:00 | 2553.04 | 2584.26 | 2559.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 13:15:00 | 2536.17 | 2574.64 | 2556.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-23 14:15:00 | 2515.72 | 2574.64 | 2556.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2023-08-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-23 15:15:00 | 2462.49 | 2533.10 | 2539.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 09:15:00 | 2440.82 | 2483.08 | 2507.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-29 09:15:00 | 2426.62 | 2416.52 | 2439.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-29 09:30:00 | 2431.47 | 2416.52 | 2439.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 10:15:00 | 2427.15 | 2418.64 | 2438.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 11:15:00 | 2447.95 | 2418.64 | 2438.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 11:15:00 | 2434.72 | 2421.86 | 2438.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 11:30:00 | 2438.16 | 2421.86 | 2438.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 12:15:00 | 2438.30 | 2425.15 | 2438.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-29 12:30:00 | 2441.94 | 2425.15 | 2438.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-29 13:15:00 | 2427.11 | 2425.54 | 2437.21 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-30 12:15:00 | 2460.26 | 2442.28 | 2440.90 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-08-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-30 15:15:00 | 2433.41 | 2439.89 | 2440.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-31 09:15:00 | 2326.28 | 2417.17 | 2429.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 11:15:00 | 2371.55 | 2366.30 | 2389.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-01 12:00:00 | 2371.55 | 2366.30 | 2389.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 13:15:00 | 2384.16 | 2372.31 | 2388.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 13:30:00 | 2386.92 | 2372.31 | 2388.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-01 14:15:00 | 2373.44 | 2372.53 | 2386.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-01 14:45:00 | 2379.12 | 2372.53 | 2386.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 2393.17 | 2377.09 | 2386.36 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2023-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-04 15:15:00 | 2395.11 | 2390.75 | 2390.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 10:15:00 | 2407.47 | 2395.49 | 2392.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-06 15:15:00 | 2413.48 | 2415.18 | 2409.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-07 09:15:00 | 2424.54 | 2417.05 | 2410.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 09:15:00 | 2424.54 | 2417.05 | 2410.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-07 09:30:00 | 2417.85 | 2417.05 | 2410.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 2468.94 | 2493.32 | 2469.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 2486.68 | 2493.32 | 2469.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 2477.13 | 2490.08 | 2470.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-13 09:15:00 | 2482.32 | 2471.85 | 2468.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-13 09:15:00 | 2431.42 | 2463.76 | 2464.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2023-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-13 09:15:00 | 2431.42 | 2463.76 | 2464.90 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2023-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-15 12:15:00 | 2468.70 | 2456.11 | 2454.70 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2023-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-18 11:15:00 | 2440.29 | 2454.10 | 2454.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-18 13:15:00 | 2434.96 | 2448.65 | 2452.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 09:15:00 | 2413.00 | 2410.52 | 2425.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-21 09:45:00 | 2416.10 | 2410.52 | 2425.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 10:15:00 | 2410.19 | 2410.45 | 2423.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-21 11:15:00 | 2402.48 | 2410.45 | 2423.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-21 12:30:00 | 2402.63 | 2408.95 | 2420.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 09:15:00 | 2404.71 | 2405.92 | 2416.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-22 11:15:00 | 2400.88 | 2404.20 | 2413.66 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 12:15:00 | 2403.50 | 2403.72 | 2411.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 12:30:00 | 2409.02 | 2403.72 | 2411.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-25 09:15:00 | 2371.55 | 2391.81 | 2403.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-25 11:15:00 | 2363.41 | 2388.12 | 2400.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-25 14:15:00 | 2410.24 | 2388.53 | 2396.09 | SL hit (close>static) qty=1.00 sl=2404.27 alert=retest2 |

### Cycle 30 — BUY (started 2023-09-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-27 12:15:00 | 2415.33 | 2398.62 | 2397.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-28 09:15:00 | 2428.66 | 2407.93 | 2402.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-28 11:15:00 | 2407.76 | 2408.81 | 2403.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-28 11:15:00 | 2407.76 | 2408.81 | 2403.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 11:15:00 | 2407.76 | 2408.81 | 2403.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 11:45:00 | 2407.86 | 2408.81 | 2403.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-28 13:15:00 | 2379.26 | 2402.43 | 2401.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-28 14:00:00 | 2379.26 | 2402.43 | 2401.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2023-09-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 14:15:00 | 2392.79 | 2400.51 | 2401.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-29 09:15:00 | 2367.00 | 2393.97 | 2397.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-04 10:15:00 | 2398.36 | 2340.24 | 2350.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-04 10:15:00 | 2398.36 | 2340.24 | 2350.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 10:15:00 | 2398.36 | 2340.24 | 2350.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-04 11:00:00 | 2398.36 | 2340.24 | 2350.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 11:15:00 | 2383.96 | 2348.98 | 2353.20 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-04 13:15:00 | 2395.40 | 2363.52 | 2359.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-06 09:15:00 | 2411.59 | 2391.21 | 2379.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-09 09:15:00 | 2361.08 | 2393.57 | 2388.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-09 09:15:00 | 2361.08 | 2393.57 | 2388.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 09:15:00 | 2361.08 | 2393.57 | 2388.40 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2023-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-09 11:15:00 | 2369.95 | 2383.28 | 2384.27 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2023-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 10:15:00 | 2417.75 | 2388.92 | 2385.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-10 13:15:00 | 2423.08 | 2402.30 | 2392.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 13:15:00 | 2418.82 | 2426.69 | 2412.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 13:15:00 | 2418.82 | 2426.69 | 2412.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 13:15:00 | 2418.82 | 2426.69 | 2412.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 14:00:00 | 2418.82 | 2426.69 | 2412.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 14:15:00 | 2409.41 | 2423.24 | 2412.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-11 15:00:00 | 2409.41 | 2423.24 | 2412.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 15:15:00 | 2420.76 | 2422.74 | 2413.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-12 09:15:00 | 2438.69 | 2422.74 | 2413.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-12 10:00:00 | 2423.42 | 2422.88 | 2414.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-12 10:30:00 | 2426.48 | 2424.73 | 2415.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-13 09:15:00 | 2360.60 | 2414.75 | 2415.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2023-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-13 09:15:00 | 2360.60 | 2414.75 | 2415.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 10:15:00 | 2341.31 | 2358.02 | 2369.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-20 09:15:00 | 2343.05 | 2333.70 | 2342.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 09:15:00 | 2343.05 | 2333.70 | 2342.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 2343.05 | 2333.70 | 2342.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 09:45:00 | 2305.48 | 2315.01 | 2328.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-25 12:15:00 | 2190.21 | 2235.05 | 2269.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-27 09:15:00 | 2171.31 | 2159.41 | 2195.06 | SL hit (close>ema200) qty=0.50 sl=2159.41 alert=retest2 |

### Cycle 36 — BUY (started 2023-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 11:15:00 | 2214.84 | 2197.92 | 2197.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-30 13:15:00 | 2226.37 | 2206.46 | 2201.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 09:15:00 | 2219.49 | 2221.57 | 2215.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-01 10:00:00 | 2219.49 | 2221.57 | 2215.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 10:15:00 | 2213.33 | 2219.92 | 2215.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 11:00:00 | 2213.33 | 2219.92 | 2215.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 11:15:00 | 2190.94 | 2214.12 | 2212.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 12:00:00 | 2190.94 | 2214.12 | 2212.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2023-11-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 12:15:00 | 2171.21 | 2205.54 | 2209.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-01 13:15:00 | 2162.34 | 2196.90 | 2204.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-02 10:15:00 | 2196.47 | 2182.23 | 2193.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-02 10:15:00 | 2196.47 | 2182.23 | 2193.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 10:15:00 | 2196.47 | 2182.23 | 2193.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 10:45:00 | 2192.88 | 2182.23 | 2193.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 11:15:00 | 2202.72 | 2186.32 | 2194.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-02 12:00:00 | 2202.72 | 2186.32 | 2194.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-02 14:15:00 | 2140.38 | 2182.23 | 2190.95 | EMA400 retest candle locked (from downside) |

### Cycle 38 — BUY (started 2023-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 11:15:00 | 2176.25 | 2174.30 | 2174.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-08 12:15:00 | 2207.18 | 2180.88 | 2177.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-09 10:15:00 | 2176.98 | 2184.91 | 2181.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-09 10:15:00 | 2176.98 | 2184.91 | 2181.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 10:15:00 | 2176.98 | 2184.91 | 2181.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 11:00:00 | 2176.98 | 2184.91 | 2181.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 11:15:00 | 2174.12 | 2182.75 | 2180.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 11:45:00 | 2174.75 | 2182.75 | 2180.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2023-11-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-09 12:15:00 | 2160.98 | 2178.40 | 2178.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-09 13:15:00 | 2152.16 | 2173.15 | 2176.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-13 12:15:00 | 2150.80 | 2142.36 | 2150.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-13 12:15:00 | 2150.80 | 2142.36 | 2150.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 12:15:00 | 2150.80 | 2142.36 | 2150.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-13 12:45:00 | 2153.61 | 2142.36 | 2150.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 13:15:00 | 2146.97 | 2143.28 | 2150.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-13 13:30:00 | 2144.69 | 2143.28 | 2150.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 15:15:00 | 2150.32 | 2145.01 | 2150.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-15 09:15:00 | 2172.23 | 2145.01 | 2150.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-15 09:15:00 | 2161.66 | 2148.34 | 2151.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-15 10:45:00 | 2156.72 | 2149.72 | 2151.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-15 13:30:00 | 2156.13 | 2152.28 | 2152.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-15 14:00:00 | 2155.26 | 2152.28 | 2152.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-15 14:15:00 | 2157.98 | 2153.42 | 2152.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2023-11-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-15 14:15:00 | 2157.98 | 2153.42 | 2152.85 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2023-11-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-16 09:15:00 | 2143.29 | 2151.82 | 2152.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-20 10:15:00 | 2123.32 | 2138.81 | 2143.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-21 09:15:00 | 2142.56 | 2115.59 | 2127.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-21 09:15:00 | 2142.56 | 2115.59 | 2127.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 09:15:00 | 2142.56 | 2115.59 | 2127.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 10:00:00 | 2142.56 | 2115.59 | 2127.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 10:15:00 | 2149.35 | 2122.34 | 2129.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-21 10:45:00 | 2148.43 | 2122.34 | 2129.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 2113.48 | 2127.23 | 2130.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-22 10:15:00 | 2106.79 | 2127.23 | 2130.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 10:30:00 | 2102.47 | 2108.83 | 2116.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-23 15:00:00 | 2108.05 | 2108.34 | 2113.51 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-24 09:45:00 | 2100.20 | 2108.10 | 2112.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 10:15:00 | 2106.64 | 2107.81 | 2112.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 10:45:00 | 2103.44 | 2107.81 | 2112.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 11:15:00 | 2119.10 | 2110.06 | 2112.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 12:00:00 | 2119.10 | 2110.06 | 2112.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 12:15:00 | 2125.11 | 2113.07 | 2113.79 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-11-24 13:15:00 | 2147.65 | 2119.99 | 2116.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2023-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-24 13:15:00 | 2147.65 | 2119.99 | 2116.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-24 15:15:00 | 2181.34 | 2134.66 | 2124.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-29 13:15:00 | 2333.07 | 2335.15 | 2280.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-29 14:00:00 | 2333.07 | 2335.15 | 2280.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-30 09:15:00 | 2286.05 | 2322.02 | 2288.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-04 09:15:00 | 2444.12 | 2292.01 | 2289.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2023-12-05 10:15:00 | 2688.53 | 2514.10 | 2422.50 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2023-12-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-13 11:15:00 | 2725.08 | 2758.79 | 2762.38 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-12-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 14:15:00 | 2780.43 | 2766.77 | 2765.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 15:15:00 | 2798.90 | 2773.19 | 2768.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-14 14:15:00 | 2804.72 | 2807.81 | 2791.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-14 15:00:00 | 2804.72 | 2807.81 | 2791.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 09:15:00 | 2839.67 | 2879.73 | 2864.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-19 10:00:00 | 2839.67 | 2879.73 | 2864.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-19 10:15:00 | 2853.24 | 2874.43 | 2863.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-19 11:30:00 | 2863.76 | 2871.46 | 2862.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-20 09:15:00 | 2859.98 | 2863.21 | 2861.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-20 12:15:00 | 2798.90 | 2848.62 | 2855.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2023-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 12:15:00 | 2798.90 | 2848.62 | 2855.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 13:15:00 | 2740.78 | 2827.06 | 2844.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-22 09:15:00 | 2737.97 | 2729.49 | 2764.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-22 09:30:00 | 2742.67 | 2729.49 | 2764.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 09:15:00 | 2785.48 | 2739.43 | 2752.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-26 10:00:00 | 2785.48 | 2739.43 | 2752.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 10:15:00 | 2784.55 | 2748.46 | 2755.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-26 13:00:00 | 2766.72 | 2757.70 | 2758.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-26 14:00:00 | 2761.67 | 2758.49 | 2758.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-26 14:15:00 | 2793.38 | 2765.47 | 2761.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2023-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-26 14:15:00 | 2793.38 | 2765.47 | 2761.95 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2023-12-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-27 14:15:00 | 2755.37 | 2763.20 | 2763.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-28 09:15:00 | 2741.27 | 2758.16 | 2760.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-28 12:15:00 | 2759.15 | 2754.59 | 2758.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-28 12:15:00 | 2759.15 | 2754.59 | 2758.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 12:15:00 | 2759.15 | 2754.59 | 2758.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-28 13:00:00 | 2759.15 | 2754.59 | 2758.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 13:15:00 | 2751.40 | 2753.95 | 2757.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-28 14:15:00 | 2744.22 | 2753.95 | 2757.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-29 10:00:00 | 2737.29 | 2742.86 | 2750.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-29 12:45:00 | 2746.70 | 2751.09 | 2753.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-29 13:15:00 | 2749.51 | 2751.09 | 2753.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 13:15:00 | 2747.33 | 2750.34 | 2752.64 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-12-29 14:15:00 | 2764.05 | 2753.08 | 2753.68 | SL hit (close>static) qty=1.00 sl=2763.03 alert=retest2 |

### Cycle 48 — BUY (started 2023-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-29 15:15:00 | 2762.11 | 2754.89 | 2754.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-01 10:15:00 | 2808.50 | 2766.03 | 2759.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-02 09:15:00 | 2800.99 | 2808.04 | 2788.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-02 10:00:00 | 2800.99 | 2808.04 | 2788.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 10:15:00 | 2777.57 | 2801.95 | 2787.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-02 10:30:00 | 2770.40 | 2801.95 | 2787.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-02 11:15:00 | 2825.61 | 2806.68 | 2791.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-02 14:00:00 | 2835.60 | 2812.71 | 2796.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-08 14:15:00 | 2864.78 | 2902.08 | 2903.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2024-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 14:15:00 | 2864.78 | 2902.08 | 2903.89 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2024-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 09:15:00 | 2966.62 | 2910.58 | 2907.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-10 11:15:00 | 2990.86 | 2949.70 | 2931.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-11 13:15:00 | 2976.32 | 2985.73 | 2966.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-11 14:00:00 | 2976.32 | 2985.73 | 2966.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 09:15:00 | 2986.84 | 2999.92 | 2988.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 09:45:00 | 2983.93 | 2999.92 | 2988.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 10:15:00 | 2998.62 | 2999.66 | 2989.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 10:45:00 | 2989.50 | 2999.66 | 2989.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 13:15:00 | 2997.89 | 2999.93 | 2991.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 13:45:00 | 3003.66 | 2999.93 | 2991.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 14:15:00 | 2994.88 | 2998.92 | 2992.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 15:15:00 | 2996.97 | 2998.92 | 2992.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 15:15:00 | 2996.97 | 2998.53 | 2992.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 09:45:00 | 2985.04 | 2995.06 | 2991.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 10:15:00 | 2976.80 | 2991.40 | 2990.28 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2024-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-16 11:15:00 | 2976.27 | 2988.38 | 2989.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-16 12:15:00 | 2959.35 | 2982.57 | 2986.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-19 09:15:00 | 2854.07 | 2847.58 | 2882.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-19 09:45:00 | 2856.88 | 2847.58 | 2882.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 14:15:00 | 2895.85 | 2834.78 | 2846.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-20 14:45:00 | 2899.68 | 2834.78 | 2846.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 15:15:00 | 2896.68 | 2847.16 | 2851.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-23 09:15:00 | 2912.33 | 2847.16 | 2851.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2024-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-23 09:15:00 | 2902.64 | 2858.26 | 2856.10 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 12:15:00 | 2837.00 | 2855.23 | 2855.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 14:15:00 | 2806.51 | 2844.63 | 2850.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 14:15:00 | 2814.75 | 2814.62 | 2829.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-24 15:00:00 | 2814.75 | 2814.62 | 2829.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 2787.27 | 2809.89 | 2824.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 10:15:00 | 2779.51 | 2809.89 | 2824.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-29 09:15:00 | 2953.83 | 2819.90 | 2815.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2024-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 09:15:00 | 2953.83 | 2819.90 | 2815.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 12:15:00 | 2960.66 | 2882.94 | 2848.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 11:15:00 | 3020.38 | 3024.72 | 2993.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 11:15:00 | 3020.38 | 3024.72 | 2993.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 11:15:00 | 3020.38 | 3024.72 | 2993.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 12:00:00 | 3020.38 | 3024.72 | 2993.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 09:15:00 | 3079.57 | 3084.46 | 3068.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-06 10:30:00 | 3123.15 | 3089.76 | 3072.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-06 13:00:00 | 3095.57 | 3091.24 | 3076.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-08 13:15:00 | 3069.49 | 3097.97 | 3101.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2024-02-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-08 13:15:00 | 3069.49 | 3097.97 | 3101.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-09 10:15:00 | 3039.00 | 3079.60 | 3091.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-09 12:15:00 | 3094.60 | 3082.03 | 3090.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-09 12:15:00 | 3094.60 | 3082.03 | 3090.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 12:15:00 | 3094.60 | 3082.03 | 3090.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-09 12:30:00 | 3104.19 | 3082.03 | 3090.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 13:15:00 | 3102.30 | 3086.08 | 3091.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-09 14:00:00 | 3102.30 | 3086.08 | 3091.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 14:15:00 | 3117.23 | 3092.31 | 3093.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-09 14:45:00 | 3112.58 | 3092.31 | 3093.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2024-02-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 15:15:00 | 3116.89 | 3097.23 | 3095.78 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 09:15:00 | 3071.23 | 3092.03 | 3093.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-13 09:15:00 | 3043.26 | 3076.13 | 3084.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 14:15:00 | 3077.15 | 3058.65 | 3071.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-13 14:15:00 | 3077.15 | 3058.65 | 3071.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 14:15:00 | 3077.15 | 3058.65 | 3071.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 15:00:00 | 3077.15 | 3058.65 | 3071.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 15:15:00 | 3082.96 | 3063.51 | 3072.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-14 09:15:00 | 3120.58 | 3063.51 | 3072.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2024-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-14 10:15:00 | 3112.05 | 3081.08 | 3079.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-15 11:15:00 | 3137.30 | 3110.52 | 3097.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 13:15:00 | 3106.42 | 3110.42 | 3099.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-15 14:00:00 | 3106.42 | 3110.42 | 3099.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 14:15:00 | 3100.95 | 3108.52 | 3099.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 14:30:00 | 3107.54 | 3108.52 | 3099.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 15:15:00 | 3093.04 | 3105.43 | 3099.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-16 09:15:00 | 3123.78 | 3105.43 | 3099.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-22 10:15:00 | 3121.74 | 3139.76 | 3141.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2024-02-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-22 10:15:00 | 3121.74 | 3139.76 | 3141.46 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-02-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 14:15:00 | 3160.23 | 3141.21 | 3141.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-22 15:15:00 | 3161.78 | 3145.32 | 3143.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-23 11:15:00 | 3141.42 | 3146.74 | 3144.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-23 11:15:00 | 3141.42 | 3146.74 | 3144.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 11:15:00 | 3141.42 | 3146.74 | 3144.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-23 12:00:00 | 3141.42 | 3146.74 | 3144.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 12:15:00 | 3177.63 | 3152.92 | 3147.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-23 14:15:00 | 3208.03 | 3160.26 | 3151.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-26 09:15:00 | 3191.01 | 3164.71 | 3155.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-27 14:30:00 | 3196.20 | 3202.86 | 3193.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-28 11:15:00 | 3154.56 | 3185.23 | 3188.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2024-02-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-28 11:15:00 | 3154.56 | 3185.23 | 3188.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 3115.49 | 3171.28 | 3181.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-29 10:15:00 | 3154.70 | 3147.76 | 3163.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-29 11:00:00 | 3154.70 | 3147.76 | 3163.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 11:15:00 | 3170.80 | 3152.37 | 3164.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 12:00:00 | 3170.80 | 3152.37 | 3164.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 12:15:00 | 3160.91 | 3154.08 | 3164.25 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-02-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 15:15:00 | 3199.20 | 3168.85 | 3168.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 11:15:00 | 3207.06 | 3183.63 | 3176.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 13:15:00 | 3223.05 | 3223.07 | 3210.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-04 13:45:00 | 3218.69 | 3223.07 | 3210.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 14:15:00 | 3211.42 | 3220.74 | 3210.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 15:00:00 | 3211.42 | 3220.74 | 3210.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 15:15:00 | 3220.63 | 3220.72 | 3211.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-05 09:15:00 | 3225.43 | 3220.72 | 3211.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-05 09:15:00 | 3222.76 | 3221.13 | 3212.38 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2024-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-06 09:15:00 | 3140.60 | 3199.89 | 3205.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-06 11:15:00 | 3116.75 | 3173.88 | 3191.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-07 09:15:00 | 3150.83 | 3145.49 | 3167.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-07 09:30:00 | 3153.73 | 3145.49 | 3167.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 09:15:00 | 3145.59 | 3139.28 | 3153.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-11 09:30:00 | 3153.25 | 3139.28 | 3153.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-11 10:15:00 | 3144.96 | 3140.41 | 3152.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-11 11:45:00 | 3125.91 | 3135.71 | 3149.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-03-13 09:15:00 | 2969.61 | 3039.05 | 3082.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-03-13 13:15:00 | 2813.32 | 2919.73 | 3005.63 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 64 — BUY (started 2024-03-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-15 12:15:00 | 3006.95 | 2986.23 | 2984.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-15 13:15:00 | 3013.30 | 2991.65 | 2987.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-18 09:15:00 | 2960.81 | 2997.64 | 2992.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-18 09:15:00 | 2960.81 | 2997.64 | 2992.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 09:15:00 | 2960.81 | 2997.64 | 2992.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-18 12:15:00 | 2995.56 | 2990.06 | 2989.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-19 11:15:00 | 2971.47 | 2992.20 | 2992.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 11:15:00 | 2971.47 | 2992.20 | 2992.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 09:15:00 | 2936.57 | 2967.23 | 2979.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 12:15:00 | 2961.78 | 2960.47 | 2972.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-03-20 13:00:00 | 2961.78 | 2960.47 | 2972.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-21 09:15:00 | 2996.00 | 2966.88 | 2971.65 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 3008.31 | 2975.17 | 2974.99 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2024-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-21 11:15:00 | 2969.53 | 2974.04 | 2974.49 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-03-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 15:15:00 | 2977.92 | 2974.88 | 2974.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 09:15:00 | 2984.07 | 2976.72 | 2975.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-26 14:15:00 | 3014.61 | 3020.58 | 3006.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-26 15:00:00 | 3014.61 | 3020.58 | 3006.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 15:15:00 | 3017.04 | 3019.87 | 3007.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 09:15:00 | 3029.64 | 3019.87 | 3007.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 10:15:00 | 3021.84 | 3019.30 | 3008.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 13:00:00 | 3024.26 | 3022.02 | 3012.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-27 13:30:00 | 3028.43 | 3023.43 | 3014.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 3039.34 | 3027.81 | 3018.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-28 09:30:00 | 3015.83 | 3027.81 | 3018.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 10:15:00 | 3132.41 | 3148.11 | 3129.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 10:45:00 | 3132.41 | 3148.11 | 3129.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 11:15:00 | 3148.89 | 3148.26 | 3131.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 11:45:00 | 3132.41 | 3148.26 | 3131.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 13:15:00 | 3138.71 | 3145.37 | 3132.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 14:00:00 | 3138.71 | 3145.37 | 3132.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 14:15:00 | 3136.28 | 3143.55 | 3133.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 15:00:00 | 3136.28 | 3143.55 | 3133.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 15:15:00 | 3139.19 | 3142.68 | 3133.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-04 09:15:00 | 3150.78 | 3142.68 | 3133.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-04 11:15:00 | 3117.77 | 3135.40 | 3132.41 | SL hit (close<static) qty=1.00 sl=3130.52 alert=retest2 |

### Cycle 69 — SELL (started 2024-04-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 14:15:00 | 3108.90 | 3128.75 | 3130.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-09 13:15:00 | 3090.14 | 3111.18 | 3117.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-10 11:15:00 | 3119.03 | 3108.73 | 3113.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-10 11:15:00 | 3119.03 | 3108.73 | 3113.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 11:15:00 | 3119.03 | 3108.73 | 3113.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 11:30:00 | 3115.68 | 3108.73 | 3113.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 12:15:00 | 3141.28 | 3115.24 | 3115.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-10 13:00:00 | 3141.28 | 3115.24 | 3115.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2024-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-10 13:15:00 | 3134.44 | 3119.08 | 3117.38 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2024-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-15 09:15:00 | 3055.82 | 3109.14 | 3116.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-18 14:15:00 | 2917.23 | 3007.64 | 3034.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-22 12:15:00 | 2953.05 | 2949.02 | 2971.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-22 14:15:00 | 2970.50 | 2954.44 | 2970.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 14:15:00 | 2970.50 | 2954.44 | 2970.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-22 15:00:00 | 2970.50 | 2954.44 | 2970.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-22 15:15:00 | 2978.50 | 2959.25 | 2971.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-23 09:15:00 | 2964.69 | 2959.25 | 2971.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-23 11:15:00 | 2990.76 | 2969.20 | 2972.90 | SL hit (close>static) qty=1.00 sl=2981.31 alert=retest2 |

### Cycle 72 — BUY (started 2024-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 10:15:00 | 2984.07 | 2973.53 | 2973.34 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-24 13:15:00 | 2966.62 | 2972.87 | 2973.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-24 14:15:00 | 2951.06 | 2968.50 | 2971.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-25 13:15:00 | 2983.11 | 2961.87 | 2965.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-25 13:15:00 | 2983.11 | 2961.87 | 2965.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 13:15:00 | 2983.11 | 2961.87 | 2965.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-25 14:00:00 | 2983.11 | 2961.87 | 2965.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2024-04-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-25 14:15:00 | 3021.84 | 2973.86 | 2970.39 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2024-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 14:15:00 | 2959.26 | 2979.13 | 2981.29 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-05-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-02 11:15:00 | 3004.43 | 2985.36 | 2983.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-02 13:15:00 | 3015.10 | 2995.09 | 2988.29 | Break + close above crossover candle high |

### Cycle 77 — SELL (started 2024-05-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-02 14:15:00 | 2935.16 | 2983.10 | 2983.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-03 10:15:00 | 2921.83 | 2956.81 | 2970.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 2800.50 | 2778.17 | 2816.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-08 10:00:00 | 2800.50 | 2778.17 | 2816.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-09 09:15:00 | 2768.41 | 2772.97 | 2795.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 10:30:00 | 2757.07 | 2769.04 | 2791.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 11:00:00 | 2753.34 | 2769.04 | 2791.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 13:15:00 | 2755.86 | 2729.75 | 2736.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 14:15:00 | 2792.07 | 2749.40 | 2744.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2024-05-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 14:15:00 | 2792.07 | 2749.40 | 2744.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 15:15:00 | 2797.40 | 2759.00 | 2749.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 09:15:00 | 2922.08 | 2943.97 | 2899.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 10:00:00 | 2922.08 | 2943.97 | 2899.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 13:15:00 | 2890.03 | 2927.76 | 2905.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-16 13:45:00 | 2899.39 | 2927.76 | 2905.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 2949.71 | 2932.15 | 2909.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-17 10:00:00 | 2965.56 | 2940.32 | 2917.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 09:45:00 | 2958.29 | 2964.77 | 2947.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-05-23 12:15:00 | 3262.12 | 3104.97 | 3050.56 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 79 — SELL (started 2024-05-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 13:15:00 | 3166.29 | 3191.94 | 3194.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 14:15:00 | 3152.28 | 3184.01 | 3190.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 11:15:00 | 3180.88 | 3172.45 | 3181.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 11:15:00 | 3180.88 | 3172.45 | 3181.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 11:15:00 | 3180.88 | 3172.45 | 3181.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 11:45:00 | 3186.50 | 3172.45 | 3181.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 12:15:00 | 3197.41 | 3177.44 | 3183.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 13:00:00 | 3197.41 | 3177.44 | 3183.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 3190.58 | 3180.07 | 3183.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 15:00:00 | 3155.63 | 3175.18 | 3181.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-31 11:15:00 | 3210.93 | 3165.55 | 3163.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2024-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 11:15:00 | 3210.93 | 3165.55 | 3163.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-31 12:15:00 | 3255.09 | 3183.46 | 3172.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 3325.24 | 3447.86 | 3364.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 3325.24 | 3447.86 | 3364.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 3325.24 | 3447.86 | 3364.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 3180.64 | 3447.86 | 3364.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 3019.99 | 3362.29 | 3333.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 3019.99 | 3362.29 | 3333.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 2827.21 | 3255.27 | 3287.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 2769.09 | 2988.84 | 3128.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 14:15:00 | 3008.17 | 2962.95 | 3058.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 15:00:00 | 3008.17 | 2962.95 | 3058.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 3110.11 | 2998.09 | 3058.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 3110.11 | 2998.09 | 3058.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 3109.62 | 3020.40 | 3062.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 11:30:00 | 3086.55 | 3033.66 | 3065.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 14:30:00 | 3085.48 | 3059.77 | 3070.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-07 11:15:00 | 3089.94 | 3074.37 | 3075.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-07 11:15:00 | 3099.10 | 3079.31 | 3077.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2024-06-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 11:15:00 | 3099.10 | 3079.31 | 3077.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 12:15:00 | 3120.77 | 3087.60 | 3081.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 3122.76 | 3130.60 | 3113.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 15:00:00 | 3122.76 | 3130.60 | 3113.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 15:15:00 | 3114.28 | 3127.34 | 3113.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:15:00 | 3136.72 | 3127.34 | 3113.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 3125.28 | 3126.93 | 3114.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 12:15:00 | 3148.16 | 3126.78 | 3116.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 13:00:00 | 3147.19 | 3130.86 | 3119.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 14:15:00 | 3145.98 | 3133.62 | 3121.43 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:15:00 | 3152.28 | 3123.26 | 3122.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 3146.95 | 3128.00 | 3124.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 11:00:00 | 3165.37 | 3135.48 | 3128.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 15:00:00 | 3169.68 | 3148.44 | 3137.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 11:45:00 | 3162.46 | 3177.67 | 3168.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-19 14:30:00 | 3160.96 | 3168.81 | 3166.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 3164.64 | 3167.97 | 3166.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 09:15:00 | 3164.69 | 3167.97 | 3166.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 3203.18 | 3175.01 | 3169.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 10:15:00 | 3204.29 | 3175.01 | 3169.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 09:15:00 | 3121.50 | 3163.74 | 3168.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 09:15:00 | 3121.50 | 3163.74 | 3168.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 12:15:00 | 3104.53 | 3140.75 | 3155.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 10:15:00 | 3084.61 | 3081.88 | 3096.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-26 10:30:00 | 3079.28 | 3081.88 | 3096.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 3088.15 | 3079.22 | 3088.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:45:00 | 3091.40 | 3079.22 | 3088.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 3072.44 | 3077.86 | 3086.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:30:00 | 3090.91 | 3077.86 | 3086.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 3080.88 | 3074.44 | 3081.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 3080.88 | 3074.44 | 3081.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 3078.11 | 3075.18 | 3081.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:15:00 | 3100.66 | 3075.18 | 3081.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 3096.54 | 3079.45 | 3082.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:45:00 | 3100.32 | 3079.45 | 3082.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 3075.64 | 3080.16 | 3082.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 12:15:00 | 3067.50 | 3080.16 | 3082.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 14:30:00 | 3069.53 | 3079.00 | 3081.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 11:00:00 | 3072.39 | 3077.78 | 3080.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 11:30:00 | 3071.81 | 3075.98 | 3079.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 13:15:00 | 3083.93 | 3075.77 | 3078.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 14:00:00 | 3083.93 | 3075.77 | 3078.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 14:15:00 | 3087.81 | 3078.18 | 3079.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 15:00:00 | 3087.81 | 3078.18 | 3079.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-02 10:15:00 | 3093.38 | 3081.04 | 3080.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2024-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 10:15:00 | 3093.38 | 3081.04 | 3080.36 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2024-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 11:15:00 | 3056.83 | 3076.20 | 3078.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 13:15:00 | 3054.36 | 3068.83 | 3074.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 12:15:00 | 3062.26 | 3061.15 | 3067.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-03 13:00:00 | 3062.26 | 3061.15 | 3067.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 13:15:00 | 3064.54 | 3061.82 | 3067.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 13:45:00 | 3063.23 | 3061.82 | 3067.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 3094.60 | 3068.38 | 3069.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 14:30:00 | 3089.75 | 3068.38 | 3069.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — BUY (started 2024-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 15:15:00 | 3087.81 | 3072.27 | 3071.26 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-07-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 10:15:00 | 3052.62 | 3068.03 | 3069.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 12:15:00 | 3044.18 | 3053.12 | 3059.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-05 14:15:00 | 3052.91 | 3050.65 | 3057.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-05 15:00:00 | 3052.91 | 3050.65 | 3057.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 15:15:00 | 3050.15 | 3050.55 | 3056.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 09:15:00 | 3040.21 | 3050.55 | 3056.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 3028.77 | 3046.20 | 3053.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 10:30:00 | 3019.41 | 3036.31 | 3048.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-08 15:15:00 | 3017.04 | 3016.37 | 3033.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-09 11:45:00 | 3013.16 | 3020.73 | 3030.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 09:30:00 | 3017.04 | 3015.61 | 3024.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 3015.63 | 3006.27 | 3013.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:45:00 | 3024.41 | 3006.27 | 3013.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 10:15:00 | 3020.67 | 3009.15 | 3014.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 11:00:00 | 3020.67 | 3009.15 | 3014.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 3007.73 | 3008.87 | 3013.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 14:45:00 | 2985.34 | 3002.47 | 3009.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 09:30:00 | 2990.76 | 2997.82 | 3006.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-16 09:15:00 | 3025.33 | 2996.49 | 2994.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2024-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 09:15:00 | 3025.33 | 2996.49 | 2994.78 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 2968.56 | 2998.24 | 2999.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 2944.67 | 2981.51 | 2990.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 10:15:00 | 2943.16 | 2920.84 | 2937.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 10:15:00 | 2943.16 | 2920.84 | 2937.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 2943.16 | 2920.84 | 2937.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:00:00 | 2943.16 | 2920.84 | 2937.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 2917.96 | 2920.26 | 2935.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 2836.86 | 2920.26 | 2935.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 14:15:00 | 2907.58 | 2912.82 | 2929.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 15:00:00 | 2899.68 | 2910.20 | 2926.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-25 10:45:00 | 2906.66 | 2895.98 | 2906.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 2889.40 | 2894.67 | 2904.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 11:45:00 | 2898.08 | 2894.67 | 2904.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 2903.61 | 2892.41 | 2900.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:15:00 | 2908.41 | 2892.41 | 2900.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 2922.08 | 2898.34 | 2902.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:00:00 | 2922.08 | 2898.34 | 2902.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-07-26 10:15:00 | 2950.68 | 2908.81 | 2906.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 10:15:00 | 2950.68 | 2908.81 | 2906.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 11:15:00 | 2952.08 | 2917.47 | 2910.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 09:15:00 | 2984.17 | 2987.84 | 2967.51 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-30 11:45:00 | 3013.79 | 2995.73 | 2974.68 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 09:15:00 | 3047.62 | 3089.85 | 3067.50 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-02 09:15:00 | 3047.62 | 3089.85 | 3067.50 | SL hit (close<ema400) qty=1.00 sl=3067.50 alert=retest1 |

### Cycle 91 — SELL (started 2024-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 09:15:00 | 3014.03 | 3068.79 | 3068.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 10:15:00 | 2954.51 | 3045.93 | 3058.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 3032.16 | 2988.43 | 3016.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 3032.16 | 2988.43 | 3016.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 3032.16 | 2988.43 | 3016.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 10:00:00 | 3032.16 | 2988.43 | 3016.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 3028.96 | 2996.54 | 3018.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:00:00 | 2995.71 | 3007.42 | 3018.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 09:15:00 | 3040.74 | 3005.67 | 3014.22 | SL hit (close>static) qty=1.00 sl=3039.09 alert=retest2 |

### Cycle 92 — BUY (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 11:15:00 | 3066.48 | 3024.92 | 3021.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 14:15:00 | 3085.68 | 3050.07 | 3035.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 3066.87 | 3080.74 | 3062.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 15:00:00 | 3066.87 | 3080.74 | 3062.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 3075.74 | 3079.74 | 3063.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-09 09:15:00 | 3107.15 | 3079.74 | 3063.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 09:15:00 | 2977.43 | 3066.69 | 3067.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 09:15:00 | 2977.43 | 3066.69 | 3067.85 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2024-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 09:15:00 | 3026.93 | 2997.41 | 2996.31 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-08-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 12:15:00 | 2979.66 | 2999.20 | 3001.09 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 10:15:00 | 3027.56 | 3002.93 | 3001.00 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 11:15:00 | 2995.61 | 3010.57 | 3012.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 12:15:00 | 2980.49 | 3004.55 | 3009.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 09:15:00 | 2991.15 | 2980.61 | 2988.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 09:15:00 | 2991.15 | 2980.61 | 2988.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 2991.15 | 2980.61 | 2988.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:00:00 | 2991.15 | 2980.61 | 2988.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 2987.95 | 2982.08 | 2988.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 12:45:00 | 2981.85 | 2983.65 | 2988.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 13:15:00 | 2980.39 | 2983.65 | 2988.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 10:15:00 | 2937.88 | 2933.53 | 2933.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 10:15:00 | 2937.88 | 2933.53 | 2933.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 12:15:00 | 2947.91 | 2937.29 | 2934.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 09:15:00 | 2924.45 | 2937.53 | 2936.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 09:15:00 | 2924.45 | 2937.53 | 2936.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 2924.45 | 2937.53 | 2936.20 | EMA400 retest candle locked (from upside) |

### Cycle 99 — SELL (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 10:15:00 | 2922.61 | 2934.55 | 2934.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 09:15:00 | 2892.75 | 2918.65 | 2924.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-06 11:15:00 | 2917.18 | 2913.30 | 2920.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-06 12:00:00 | 2917.18 | 2913.30 | 2920.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 12:15:00 | 2907.10 | 2912.06 | 2919.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-06 14:00:00 | 2898.76 | 2909.40 | 2917.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 12:45:00 | 2899.34 | 2880.20 | 2886.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 14:30:00 | 2900.21 | 2887.14 | 2888.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 15:15:00 | 2889.94 | 2868.59 | 2872.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 09:15:00 | 2887.27 | 2875.74 | 2875.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 2887.27 | 2875.74 | 2875.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 09:15:00 | 2921.06 | 2886.79 | 2881.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 14:15:00 | 2895.27 | 2896.07 | 2888.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 15:00:00 | 2895.27 | 2896.07 | 2888.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 15:15:00 | 2891.00 | 2895.06 | 2888.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:15:00 | 2893.91 | 2895.06 | 2888.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 2877.48 | 2891.54 | 2887.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 10:00:00 | 2877.48 | 2891.54 | 2887.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 2869.87 | 2887.21 | 2886.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 11:00:00 | 2869.87 | 2887.21 | 2886.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2024-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 11:15:00 | 2867.54 | 2883.27 | 2884.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 13:15:00 | 2854.16 | 2875.57 | 2880.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 2853.53 | 2844.56 | 2859.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 2853.53 | 2844.56 | 2859.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 2884.99 | 2853.72 | 2860.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:00:00 | 2884.99 | 2853.72 | 2860.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 2890.37 | 2861.05 | 2863.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:45:00 | 2892.41 | 2861.05 | 2863.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2024-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 11:15:00 | 2908.36 | 2870.51 | 2867.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 12:15:00 | 2926.05 | 2881.62 | 2872.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 12:15:00 | 3007.34 | 3010.05 | 2988.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-26 13:00:00 | 3007.34 | 3010.05 | 2988.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 3033.76 | 3048.20 | 3032.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 13:30:00 | 3040.26 | 3048.20 | 3032.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 3042.20 | 3047.00 | 3033.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 14:30:00 | 3026.78 | 3047.00 | 3033.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 3035.46 | 3044.69 | 3033.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:15:00 | 3023.87 | 3044.69 | 3033.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 3036.14 | 3042.98 | 3033.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:00:00 | 3036.14 | 3042.98 | 3033.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 3021.93 | 3038.77 | 3032.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 11:00:00 | 3021.93 | 3038.77 | 3032.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 11:15:00 | 3036.43 | 3038.30 | 3033.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 12:15:00 | 3048.64 | 3038.30 | 3033.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 13:15:00 | 3008.31 | 3039.98 | 3042.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2024-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 13:15:00 | 3008.31 | 3039.98 | 3042.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 2969.24 | 3010.47 | 3023.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 2955.96 | 2946.79 | 2977.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 10:00:00 | 2955.96 | 2946.79 | 2977.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 2968.51 | 2951.14 | 2976.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:30:00 | 2972.93 | 2951.14 | 2976.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 3011.03 | 2963.11 | 2979.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 12:00:00 | 3011.03 | 2963.11 | 2979.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 3048.06 | 2980.10 | 2986.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:00:00 | 3048.06 | 2980.10 | 2986.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2024-10-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 13:15:00 | 3049.47 | 2993.98 | 2991.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 15:15:00 | 3073.27 | 3021.07 | 3005.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 09:15:00 | 2989.89 | 3039.05 | 3028.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 09:15:00 | 2989.89 | 3039.05 | 3028.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 2989.89 | 3039.05 | 3028.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 09:30:00 | 2976.90 | 3039.05 | 3028.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 3003.17 | 3031.87 | 3025.80 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2024-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 12:15:00 | 2995.71 | 3017.63 | 3019.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 13:15:00 | 2982.43 | 3010.59 | 3016.53 | Break + close below crossover candle low |

### Cycle 106 — BUY (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 14:15:00 | 3076.85 | 3023.84 | 3022.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 15:15:00 | 3097.31 | 3038.54 | 3028.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 10:15:00 | 3039.34 | 3045.35 | 3034.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 10:15:00 | 3039.34 | 3045.35 | 3034.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 3039.34 | 3045.35 | 3034.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:00:00 | 3039.34 | 3045.35 | 3034.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 3039.34 | 3044.15 | 3034.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 12:15:00 | 3043.31 | 3044.15 | 3034.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:30:00 | 3047.96 | 3040.53 | 3036.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 12:15:00 | 3014.81 | 3032.30 | 3033.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2024-10-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 12:15:00 | 3014.81 | 3032.30 | 3033.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-14 14:15:00 | 3007.34 | 3024.65 | 3029.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 11:15:00 | 3019.90 | 3016.70 | 3023.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 11:15:00 | 3019.90 | 3016.70 | 3023.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 11:15:00 | 3019.90 | 3016.70 | 3023.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 11:45:00 | 3024.11 | 3016.70 | 3023.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 12:15:00 | 3009.43 | 3015.25 | 3022.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 12:45:00 | 3021.45 | 3015.25 | 3022.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 13:15:00 | 3016.36 | 3015.47 | 3021.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 13:45:00 | 3017.67 | 3015.47 | 3021.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 3004.87 | 3013.35 | 3020.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:30:00 | 3010.54 | 3013.35 | 3020.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 15:15:00 | 3008.55 | 3012.39 | 3018.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:15:00 | 3002.20 | 3012.39 | 3018.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 3002.06 | 3010.32 | 3017.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 11:30:00 | 2988.10 | 3002.80 | 3012.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 15:00:00 | 2992.61 | 2996.32 | 3006.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 11:15:00 | 2842.98 | 2891.58 | 2921.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-21 12:15:00 | 2838.69 | 2880.38 | 2913.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 10:15:00 | 2785.86 | 2783.00 | 2825.29 | SL hit (close>ema200) qty=0.50 sl=2783.00 alert=retest2 |

### Cycle 108 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 2774.47 | 2710.82 | 2706.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 2844.47 | 2744.66 | 2723.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 12:15:00 | 2851.89 | 2855.12 | 2814.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-31 13:00:00 | 2851.89 | 2855.12 | 2814.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 2830.90 | 2853.41 | 2831.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 2830.90 | 2853.41 | 2831.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 2784.26 | 2839.58 | 2827.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 2784.26 | 2839.58 | 2827.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 2794.98 | 2830.66 | 2824.55 | EMA400 retest candle locked (from upside) |

### Cycle 109 — SELL (started 2024-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 13:15:00 | 2796.53 | 2817.10 | 2819.03 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-11-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 15:15:00 | 2828.52 | 2817.02 | 2816.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 2885.19 | 2830.65 | 2822.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 2874.81 | 2905.85 | 2873.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 2874.81 | 2905.85 | 2873.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 2874.81 | 2905.85 | 2873.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 2874.81 | 2905.85 | 2873.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 2874.28 | 2899.54 | 2873.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 2868.71 | 2899.54 | 2873.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 2876.37 | 2894.91 | 2873.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:45:00 | 2865.75 | 2894.91 | 2873.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 12:15:00 | 2891.00 | 2894.12 | 2875.48 | EMA400 retest candle locked (from upside) |

### Cycle 111 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 2853.97 | 2869.68 | 2871.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 14:15:00 | 2842.58 | 2864.26 | 2868.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 11:15:00 | 2753.00 | 2742.37 | 2757.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-18 12:00:00 | 2753.00 | 2742.37 | 2757.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 2761.14 | 2746.12 | 2758.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:30:00 | 2762.06 | 2746.12 | 2758.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 13:15:00 | 2749.80 | 2746.86 | 2757.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 14:00:00 | 2749.80 | 2746.86 | 2757.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 2790.27 | 2751.86 | 2756.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 2790.27 | 2751.86 | 2756.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — BUY (started 2024-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 10:15:00 | 2794.35 | 2760.36 | 2760.05 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 2728.13 | 2760.21 | 2761.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 2188.27 | 2641.40 | 2706.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 2213.67 | 2186.13 | 2303.01 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-25 15:00:00 | 2186.14 | 2214.13 | 2275.59 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-26 09:30:00 | 2180.95 | 2194.47 | 2255.81 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 13:15:00 | 2076.83 | 2143.44 | 2209.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-26 14:15:00 | 2071.90 | 2132.74 | 2198.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-11-27 09:15:00 | 2161.95 | 2133.46 | 2187.37 | SL hit (close>ema200) qty=0.50 sl=2133.46 alert=retest1 |

### Cycle 114 — BUY (started 2024-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-27 13:15:00 | 2309.31 | 2207.86 | 2207.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-28 09:15:00 | 2424.29 | 2285.11 | 2246.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 10:15:00 | 2330.74 | 2353.02 | 2313.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-29 11:00:00 | 2330.74 | 2353.02 | 2313.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 11:15:00 | 2409.17 | 2418.41 | 2398.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 11:45:00 | 2401.41 | 2418.41 | 2398.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 09:15:00 | 2398.51 | 2414.11 | 2404.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 10:00:00 | 2398.51 | 2414.11 | 2404.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 10:15:00 | 2412.08 | 2413.70 | 2405.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 12:00:00 | 2437.87 | 2418.54 | 2408.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 11:15:00 | 2431.47 | 2429.12 | 2426.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 14:45:00 | 2423.71 | 2427.00 | 2426.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-09 15:15:00 | 2418.67 | 2425.33 | 2425.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2024-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 15:15:00 | 2418.67 | 2425.33 | 2425.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 09:15:00 | 2404.27 | 2421.12 | 2423.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 10:15:00 | 2435.35 | 2393.60 | 2398.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 10:15:00 | 2435.35 | 2393.60 | 2398.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 2435.35 | 2393.60 | 2398.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 11:00:00 | 2435.35 | 2393.60 | 2398.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 2416.93 | 2398.27 | 2399.83 | EMA400 retest candle locked (from downside) |

### Cycle 116 — BUY (started 2024-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 12:15:00 | 2446.50 | 2407.91 | 2404.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 13:15:00 | 2459.92 | 2437.68 | 2423.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 09:15:00 | 2441.99 | 2442.00 | 2429.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-16 10:00:00 | 2441.99 | 2442.00 | 2429.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 2438.64 | 2439.55 | 2431.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 14:45:00 | 2443.30 | 2438.40 | 2432.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 15:15:00 | 2430.50 | 2436.82 | 2432.07 | SL hit (close<static) qty=1.00 sl=2430.89 alert=retest2 |

### Cycle 117 — SELL (started 2024-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 14:15:00 | 2414.07 | 2429.29 | 2430.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 10:15:00 | 2403.79 | 2418.59 | 2424.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 11:15:00 | 2365.16 | 2356.23 | 2373.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 11:15:00 | 2365.16 | 2356.23 | 2373.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 2365.16 | 2356.23 | 2373.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 13:00:00 | 2339.03 | 2352.79 | 2370.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 14:15:00 | 2324.29 | 2314.84 | 2314.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 118 — BUY (started 2024-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 14:15:00 | 2324.29 | 2314.84 | 2314.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 15:15:00 | 2327.73 | 2317.42 | 2315.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 10:15:00 | 2442.62 | 2455.90 | 2413.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-31 11:00:00 | 2442.62 | 2455.90 | 2413.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 2484.84 | 2496.46 | 2483.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 2484.84 | 2496.46 | 2483.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 2483.82 | 2493.93 | 2483.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 2462.49 | 2493.93 | 2483.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 2453.23 | 2485.79 | 2480.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:45:00 | 2445.91 | 2485.79 | 2480.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 2414.94 | 2471.62 | 2474.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 2401.27 | 2441.79 | 2459.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 2431.95 | 2426.99 | 2446.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 09:45:00 | 2425.36 | 2426.99 | 2446.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 2435.10 | 2428.93 | 2442.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 12:45:00 | 2436.41 | 2428.93 | 2442.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 2444.70 | 2432.08 | 2443.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:30:00 | 2443.34 | 2432.08 | 2443.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 2444.80 | 2434.63 | 2443.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 15:00:00 | 2444.80 | 2434.63 | 2443.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 2447.76 | 2437.25 | 2443.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:15:00 | 2430.50 | 2437.25 | 2443.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 09:15:00 | 2443.93 | 2438.59 | 2443.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:30:00 | 2440.44 | 2438.59 | 2443.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 10:15:00 | 2431.47 | 2437.17 | 2442.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 11:45:00 | 2420.46 | 2433.76 | 2440.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-09 10:30:00 | 2419.83 | 2428.95 | 2435.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 14:15:00 | 2299.44 | 2348.83 | 2383.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-10 14:15:00 | 2298.84 | 2348.83 | 2383.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 13:15:00 | 2178.41 | 2256.51 | 2317.28 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 120 — BUY (started 2025-01-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 11:15:00 | 2320.07 | 2305.95 | 2305.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 2369.71 | 2321.18 | 2312.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 09:15:00 | 2334.33 | 2346.21 | 2333.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 2334.33 | 2346.21 | 2333.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 2334.33 | 2346.21 | 2333.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:30:00 | 2327.83 | 2346.21 | 2333.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 2322.40 | 2341.45 | 2332.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:00:00 | 2322.40 | 2341.45 | 2332.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 2318.67 | 2336.89 | 2330.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:45:00 | 2317.07 | 2336.89 | 2330.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 13:15:00 | 2323.27 | 2332.11 | 2329.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 13:30:00 | 2321.58 | 2332.11 | 2329.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 2326.28 | 2330.04 | 2329.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:15:00 | 2326.76 | 2330.04 | 2329.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 2329.96 | 2330.03 | 2329.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 10:45:00 | 2343.78 | 2332.48 | 2330.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-20 11:30:00 | 2345.38 | 2335.55 | 2331.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 11:45:00 | 2344.46 | 2339.89 | 2338.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 13:15:00 | 2328.17 | 2336.08 | 2336.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2025-01-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 13:15:00 | 2328.17 | 2336.08 | 2336.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 2317.55 | 2332.37 | 2334.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 2309.31 | 2293.88 | 2309.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 14:15:00 | 2309.31 | 2293.88 | 2309.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 2309.31 | 2293.88 | 2309.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 15:00:00 | 2309.31 | 2293.88 | 2309.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 2304.47 | 2295.99 | 2309.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:15:00 | 2286.34 | 2295.99 | 2309.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 2311.64 | 2299.12 | 2309.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 2311.64 | 2299.12 | 2309.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 2304.81 | 2300.26 | 2308.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 09:45:00 | 2290.89 | 2303.10 | 2307.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 12:30:00 | 2289.39 | 2300.56 | 2305.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 2176.35 | 2215.06 | 2246.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-28 09:15:00 | 2174.92 | 2215.06 | 2246.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-28 11:15:00 | 2219.15 | 2214.53 | 2241.11 | SL hit (close>ema200) qty=0.50 sl=2214.53 alert=retest2 |

### Cycle 122 — BUY (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 15:15:00 | 2250.95 | 2241.29 | 2241.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 2274.17 | 2247.86 | 2244.03 | Break + close above crossover candle high |

### Cycle 123 — SELL (started 2025-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 13:15:00 | 2153.76 | 2233.84 | 2239.61 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2025-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 12:15:00 | 2223.90 | 2210.44 | 2209.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 15:15:00 | 2232.97 | 2219.92 | 2214.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 2240.48 | 2245.22 | 2234.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 10:00:00 | 2240.48 | 2245.22 | 2234.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 2233.01 | 2242.10 | 2234.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:00:00 | 2233.01 | 2242.10 | 2234.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 2237.52 | 2241.19 | 2234.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:45:00 | 2232.82 | 2241.19 | 2234.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 13:15:00 | 2238.25 | 2240.60 | 2235.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 13:45:00 | 2234.86 | 2240.60 | 2235.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 15:15:00 | 2238.54 | 2240.10 | 2235.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 09:15:00 | 2251.14 | 2240.10 | 2235.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-10 09:45:00 | 2244.79 | 2256.98 | 2250.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 10:15:00 | 2227.34 | 2251.05 | 2248.18 | SL hit (close<static) qty=1.00 sl=2235.10 alert=retest2 |

### Cycle 125 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 11:15:00 | 2216.97 | 2244.24 | 2245.34 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-11 09:15:00 | 2305.53 | 2244.83 | 2243.36 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2025-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 09:15:00 | 2219.10 | 2247.75 | 2248.43 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2025-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-12 10:15:00 | 2259.38 | 2250.08 | 2249.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-12 11:15:00 | 2288.47 | 2257.75 | 2252.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-13 10:15:00 | 2257.59 | 2271.06 | 2263.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-13 10:15:00 | 2257.59 | 2271.06 | 2263.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 10:15:00 | 2257.59 | 2271.06 | 2263.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-13 10:45:00 | 2258.56 | 2271.06 | 2263.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 11:15:00 | 2238.10 | 2264.47 | 2261.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-13 12:00:00 | 2238.10 | 2264.47 | 2261.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2025-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 12:15:00 | 2231.12 | 2257.80 | 2258.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-13 13:15:00 | 2188.32 | 2243.90 | 2252.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 12:15:00 | 2116.00 | 2110.10 | 2147.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 13:00:00 | 2116.00 | 2110.10 | 2147.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 2156.81 | 2122.89 | 2147.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 2156.81 | 2122.89 | 2147.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 2169.03 | 2132.12 | 2149.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 09:15:00 | 2147.07 | 2132.12 | 2149.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 15:15:00 | 2145.52 | 2132.41 | 2140.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 09:15:00 | 2039.72 | 2079.75 | 2099.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-24 09:15:00 | 2038.24 | 2079.75 | 2099.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-25 09:15:00 | 2073.44 | 2062.66 | 2078.91 | SL hit (close>ema200) qty=0.50 sl=2062.66 alert=retest2 |

### Cycle 130 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 2065.10 | 2041.20 | 2040.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 14:15:00 | 2078.82 | 2062.54 | 2052.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 2174.75 | 2179.86 | 2154.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 2174.75 | 2179.86 | 2154.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 2161.81 | 2181.68 | 2171.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:30:00 | 2158.07 | 2181.68 | 2171.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 2156.52 | 2176.65 | 2170.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 2172.18 | 2176.65 | 2170.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 2193.94 | 2181.53 | 2173.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 10:30:00 | 2184.64 | 2181.53 | 2173.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 2181.78 | 2183.51 | 2177.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 14:45:00 | 2181.34 | 2183.51 | 2177.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 2183.91 | 2183.59 | 2177.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-12 09:30:00 | 2191.04 | 2183.33 | 2178.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-12 10:15:00 | 2170.77 | 2180.82 | 2177.67 | SL hit (close<static) qty=1.00 sl=2177.41 alert=retest2 |

### Cycle 131 — SELL (started 2025-03-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 11:15:00 | 2128.84 | 2170.42 | 2173.23 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 10:15:00 | 2183.76 | 2173.08 | 2172.07 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 11:15:00 | 2163.70 | 2171.21 | 2171.31 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 10:15:00 | 2178.97 | 2171.31 | 2170.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 2212.99 | 2186.57 | 2179.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 2231.46 | 2242.73 | 2227.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 10:00:00 | 2231.46 | 2242.73 | 2227.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 2254.25 | 2245.04 | 2229.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 09:15:00 | 2280.13 | 2256.99 | 2241.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 12:15:00 | 2263.60 | 2278.26 | 2278.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2025-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 12:15:00 | 2263.60 | 2278.26 | 2278.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 14:15:00 | 2249.98 | 2270.84 | 2275.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 09:15:00 | 2282.60 | 2269.65 | 2273.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 09:15:00 | 2282.60 | 2269.65 | 2273.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 2282.60 | 2269.65 | 2273.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:00:00 | 2282.60 | 2269.65 | 2273.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 2273.59 | 2270.44 | 2273.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:45:00 | 2284.79 | 2270.44 | 2273.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 2268.59 | 2270.07 | 2273.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 13:30:00 | 2257.25 | 2266.97 | 2271.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 14:15:00 | 2255.12 | 2266.97 | 2271.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 14:15:00 | 2291.72 | 2254.56 | 2259.08 | SL hit (close>static) qty=1.00 sl=2282.17 alert=retest2 |

### Cycle 136 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 2285.12 | 2266.48 | 2264.08 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 13:15:00 | 2242.27 | 2262.09 | 2263.17 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2025-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 15:15:00 | 2268.50 | 2262.03 | 2261.95 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 09:15:00 | 2255.12 | 2260.64 | 2261.33 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 10:15:00 | 2283.38 | 2265.19 | 2263.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 11:15:00 | 2294.77 | 2271.11 | 2266.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 2273.93 | 2316.90 | 2302.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 2273.93 | 2316.90 | 2302.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 2273.93 | 2316.90 | 2302.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 2273.93 | 2316.90 | 2302.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 2292.25 | 2311.97 | 2301.87 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2025-04-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 12:15:00 | 2255.65 | 2295.03 | 2295.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 2254.00 | 2286.83 | 2291.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 2176.49 | 2166.94 | 2207.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 09:15:00 | 2176.49 | 2166.94 | 2207.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 2176.49 | 2166.94 | 2207.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 10:30:00 | 2161.71 | 2169.25 | 2205.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-08 11:15:00 | 2170.82 | 2169.25 | 2205.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:45:00 | 2162.68 | 2190.59 | 2203.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 13:45:00 | 2173.10 | 2181.92 | 2194.47 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 2257.74 | 2194.20 | 2196.62 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 2257.74 | 2194.20 | 2196.62 | SL hit (close>static) qty=1.00 sl=2224.97 alert=retest2 |

### Cycle 142 — BUY (started 2025-04-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 10:15:00 | 2245.23 | 2204.41 | 2201.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 2313.19 | 2249.31 | 2227.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 10:15:00 | 2322.79 | 2323.54 | 2287.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 11:00:00 | 2322.79 | 2323.54 | 2287.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 2355.95 | 2345.71 | 2327.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:30:00 | 2350.03 | 2345.71 | 2327.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 2361.57 | 2372.77 | 2363.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 2361.57 | 2372.77 | 2363.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 2363.12 | 2370.84 | 2363.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 2351.87 | 2370.84 | 2363.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 11:15:00 | 2371.07 | 2370.89 | 2363.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 10:15:00 | 2383.87 | 2372.44 | 2367.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-24 12:15:00 | 2356.14 | 2368.00 | 2366.74 | SL hit (close<static) qty=1.00 sl=2361.67 alert=retest2 |

### Cycle 143 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 2305.73 | 2354.32 | 2360.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 2272.67 | 2337.99 | 2352.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 2301.56 | 2300.38 | 2323.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 10:00:00 | 2301.56 | 2300.38 | 2323.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 2315.42 | 2303.39 | 2322.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:45:00 | 2320.75 | 2303.39 | 2322.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 2319.98 | 2306.70 | 2322.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 11:30:00 | 2324.34 | 2306.70 | 2322.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 2304.95 | 2306.35 | 2321.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:30:00 | 2314.16 | 2306.35 | 2321.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 2297.58 | 2300.94 | 2313.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 11:45:00 | 2273.73 | 2295.13 | 2308.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 10:15:00 | 2354.59 | 2274.27 | 2268.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 2354.59 | 2274.27 | 2268.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 11:15:00 | 2403.06 | 2300.03 | 2280.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 2342.86 | 2346.38 | 2314.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 10:00:00 | 2342.86 | 2346.38 | 2314.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 2308.83 | 2335.57 | 2315.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 12:00:00 | 2308.83 | 2335.57 | 2315.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 2297.29 | 2327.92 | 2313.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:00:00 | 2297.29 | 2327.92 | 2313.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 2262.10 | 2297.65 | 2302.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 2225.65 | 2268.29 | 2281.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 2314.35 | 2218.95 | 2234.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 2314.35 | 2218.95 | 2234.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 2314.35 | 2218.95 | 2234.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 2314.35 | 2218.95 | 2234.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 2341.69 | 2263.49 | 2253.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 2347.90 | 2290.88 | 2268.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 13:15:00 | 2462.78 | 2470.92 | 2448.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 14:00:00 | 2462.78 | 2470.92 | 2448.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 2456.29 | 2466.88 | 2454.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 2460.17 | 2466.88 | 2454.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 2447.08 | 2462.92 | 2453.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:45:00 | 2444.85 | 2462.92 | 2453.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 2436.41 | 2457.62 | 2452.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 2436.41 | 2457.62 | 2452.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 2429.43 | 2446.99 | 2448.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 2417.22 | 2439.53 | 2444.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 2433.70 | 2430.31 | 2437.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 2433.70 | 2430.31 | 2437.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 2433.70 | 2430.31 | 2437.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:30:00 | 2442.23 | 2430.31 | 2437.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 2424.10 | 2429.07 | 2435.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 11:15:00 | 2422.36 | 2429.07 | 2435.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 10:15:00 | 2446.98 | 2423.85 | 2427.61 | SL hit (close>static) qty=1.00 sl=2437.29 alert=retest2 |

### Cycle 148 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 2461.52 | 2436.02 | 2432.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 2482.17 | 2453.79 | 2442.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 15:15:00 | 2465.98 | 2466.56 | 2455.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 09:15:00 | 2464.24 | 2466.56 | 2455.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 2456.58 | 2464.56 | 2455.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 2456.58 | 2464.56 | 2455.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 2471.22 | 2465.89 | 2457.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 2483.82 | 2465.89 | 2457.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 14:30:00 | 2473.35 | 2470.13 | 2462.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 12:15:00 | 2445.53 | 2460.02 | 2460.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2025-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 12:15:00 | 2445.53 | 2460.02 | 2460.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 13:15:00 | 2443.10 | 2456.64 | 2458.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 12:15:00 | 2472.38 | 2452.93 | 2454.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 12:15:00 | 2472.38 | 2452.93 | 2454.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 2472.38 | 2452.93 | 2454.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:00:00 | 2472.38 | 2452.93 | 2454.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 2448.43 | 2452.03 | 2454.06 | EMA400 retest candle locked (from downside) |

### Cycle 150 — BUY (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 15:15:00 | 2465.40 | 2456.94 | 2456.08 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 10:15:00 | 2444.36 | 2454.24 | 2454.99 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 11:15:00 | 2473.16 | 2456.65 | 2454.68 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 2430.50 | 2450.60 | 2453.03 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 2447.76 | 2425.88 | 2424.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 10:15:00 | 2463.95 | 2437.78 | 2431.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 09:15:00 | 2525.02 | 2528.51 | 2505.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 10:00:00 | 2525.02 | 2528.51 | 2505.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 2495.75 | 2518.25 | 2507.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 2500.59 | 2518.25 | 2507.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 2504.86 | 2515.57 | 2507.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:15:00 | 2514.46 | 2513.51 | 2507.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 2484.31 | 2505.21 | 2504.46 | SL hit (close<static) qty=1.00 sl=2494.10 alert=retest2 |

### Cycle 155 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 2481.11 | 2500.39 | 2502.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 2465.01 | 2493.75 | 2499.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 2458.81 | 2438.67 | 2456.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 10:15:00 | 2458.81 | 2438.67 | 2456.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 2458.81 | 2438.67 | 2456.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 2458.81 | 2438.67 | 2456.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 2460.75 | 2443.09 | 2456.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:00:00 | 2445.14 | 2455.84 | 2459.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:45:00 | 2442.81 | 2453.31 | 2457.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 11:45:00 | 2436.12 | 2448.94 | 2455.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 12:15:00 | 2402.29 | 2386.28 | 2384.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 2402.29 | 2386.28 | 2384.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 2446.40 | 2402.55 | 2393.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 2427.69 | 2429.06 | 2412.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 15:00:00 | 2427.69 | 2429.06 | 2412.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 2531.62 | 2539.92 | 2528.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:00:00 | 2531.62 | 2539.92 | 2528.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 13:15:00 | 2535.01 | 2538.94 | 2529.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 13:45:00 | 2529.97 | 2538.94 | 2529.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 2538.89 | 2538.93 | 2530.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 14:30:00 | 2533.46 | 2538.93 | 2530.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 2543.15 | 2547.84 | 2541.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:30:00 | 2535.30 | 2547.84 | 2541.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 2539.57 | 2546.19 | 2541.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:00:00 | 2539.57 | 2546.19 | 2541.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 11:15:00 | 2536.37 | 2544.22 | 2540.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 11:45:00 | 2535.59 | 2544.22 | 2540.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 12:15:00 | 2537.05 | 2542.79 | 2540.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 13:30:00 | 2546.35 | 2541.95 | 2540.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 15:15:00 | 2528.42 | 2537.84 | 2538.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — SELL (started 2025-07-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 15:15:00 | 2528.42 | 2537.84 | 2538.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 2512.91 | 2530.79 | 2534.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 15:15:00 | 2508.64 | 2506.31 | 2515.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 09:15:00 | 2514.17 | 2506.31 | 2515.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 2505.05 | 2506.06 | 2514.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 11:00:00 | 2500.30 | 2504.91 | 2513.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 11:30:00 | 2495.55 | 2503.91 | 2512.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 10:15:00 | 2521.82 | 2511.00 | 2512.27 | SL hit (close>static) qty=1.00 sl=2520.66 alert=retest2 |

### Cycle 158 — BUY (started 2025-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 09:15:00 | 2511.84 | 2500.70 | 2500.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 10:15:00 | 2522.99 | 2505.16 | 2502.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 15:15:00 | 2535.11 | 2537.37 | 2529.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 09:15:00 | 2537.72 | 2537.37 | 2529.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 2526.96 | 2535.28 | 2528.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 2526.96 | 2535.28 | 2528.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 2514.65 | 2531.16 | 2527.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 2510.00 | 2531.16 | 2527.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 2513.10 | 2524.35 | 2524.88 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 10:15:00 | 2537.53 | 2524.98 | 2524.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 14:15:00 | 2540.15 | 2531.50 | 2528.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 10:15:00 | 2530.74 | 2532.27 | 2529.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 10:15:00 | 2530.74 | 2532.27 | 2529.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 2530.74 | 2532.27 | 2529.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 2530.74 | 2532.27 | 2529.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 11:15:00 | 2524.83 | 2530.78 | 2528.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:30:00 | 2526.48 | 2530.78 | 2528.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 2521.53 | 2528.93 | 2528.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:00:00 | 2521.53 | 2528.93 | 2528.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 161 — SELL (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 13:15:00 | 2519.40 | 2527.02 | 2527.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 2510.38 | 2523.70 | 2525.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 2532.59 | 2524.04 | 2525.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 2532.59 | 2524.04 | 2525.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 2532.59 | 2524.04 | 2525.61 | EMA400 retest candle locked (from downside) |

### Cycle 162 — BUY (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 12:15:00 | 2531.33 | 2527.03 | 2526.71 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 12:15:00 | 2525.51 | 2527.26 | 2527.49 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 13:15:00 | 2530.36 | 2527.88 | 2527.75 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 2502.05 | 2523.29 | 2525.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 2480.82 | 2514.80 | 2521.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 2500.11 | 2490.71 | 2504.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 2500.11 | 2490.71 | 2504.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 2500.11 | 2490.71 | 2504.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:30:00 | 2496.42 | 2490.71 | 2504.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 2484.69 | 2489.50 | 2502.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:00:00 | 2476.84 | 2486.97 | 2499.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:30:00 | 2474.80 | 2482.75 | 2496.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 14:15:00 | 2353.00 | 2415.51 | 2439.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-31 14:15:00 | 2351.06 | 2415.51 | 2439.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-06 11:15:00 | 2229.16 | 2255.62 | 2284.40 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 166 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 2216.73 | 2184.54 | 2182.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 11:15:00 | 2241.26 | 2203.36 | 2192.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 10:15:00 | 2212.07 | 2216.59 | 2205.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 10:45:00 | 2214.89 | 2216.59 | 2205.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 2187.93 | 2210.86 | 2203.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:00:00 | 2187.93 | 2210.86 | 2203.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 2208.39 | 2210.36 | 2204.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 13:15:00 | 2213.14 | 2210.36 | 2204.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 14:30:00 | 2212.27 | 2210.27 | 2205.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 15:15:00 | 2217.21 | 2210.27 | 2205.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 10:00:00 | 2210.81 | 2211.49 | 2206.84 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 2218.18 | 2213.26 | 2208.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 12:45:00 | 2224.87 | 2214.85 | 2209.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 2252.79 | 2213.79 | 2210.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 11:15:00 | 2259.77 | 2291.08 | 2293.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 167 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 2259.77 | 2291.08 | 2293.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 13:15:00 | 2256.86 | 2279.18 | 2287.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 2226.13 | 2216.27 | 2233.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 09:45:00 | 2226.91 | 2216.27 | 2233.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 2254.83 | 2223.98 | 2235.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:45:00 | 2254.54 | 2223.98 | 2235.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 2233.69 | 2225.93 | 2235.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:15:00 | 2222.45 | 2225.93 | 2235.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 2224.87 | 2213.10 | 2211.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 168 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 2224.87 | 2213.10 | 2211.91 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 15:15:00 | 2208.49 | 2211.53 | 2211.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 10:15:00 | 2200.92 | 2208.95 | 2210.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 11:15:00 | 2209.36 | 2209.03 | 2210.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-03 12:00:00 | 2209.36 | 2209.03 | 2210.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 2206.35 | 2208.50 | 2210.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 12:30:00 | 2208.49 | 2208.50 | 2210.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 2209.46 | 2208.69 | 2210.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:00:00 | 2209.46 | 2208.69 | 2210.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 170 — BUY (started 2025-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 14:15:00 | 2220.70 | 2211.09 | 2211.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 10:15:00 | 2222.64 | 2216.13 | 2213.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 12:15:00 | 2215.27 | 2216.77 | 2214.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 12:15:00 | 2215.27 | 2216.77 | 2214.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 12:15:00 | 2215.27 | 2216.77 | 2214.39 | EMA400 retest candle locked (from upside) |

### Cycle 171 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 2209.26 | 2212.99 | 2213.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 2203.54 | 2211.10 | 2212.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 2215.76 | 2204.97 | 2208.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 2215.76 | 2204.97 | 2208.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 2215.76 | 2204.97 | 2208.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 2215.76 | 2204.97 | 2208.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 2209.07 | 2205.79 | 2208.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:15:00 | 2215.37 | 2205.79 | 2208.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 2215.37 | 2207.70 | 2208.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 2241.74 | 2207.70 | 2208.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 172 — BUY (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 09:15:00 | 2252.50 | 2216.66 | 2212.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 2261.81 | 2245.09 | 2237.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 2318.04 | 2323.30 | 2304.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 14:30:00 | 2320.85 | 2323.30 | 2304.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 2312.03 | 2318.03 | 2309.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 2309.02 | 2318.03 | 2309.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 2311.83 | 2316.79 | 2310.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 15:15:00 | 2309.80 | 2316.79 | 2310.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 2309.80 | 2315.39 | 2310.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:15:00 | 2322.21 | 2315.39 | 2310.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-09-22 13:15:00 | 2554.43 | 2501.78 | 2442.10 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 173 — SELL (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 15:15:00 | 2494.78 | 2529.17 | 2533.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 2484.98 | 2514.05 | 2525.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 13:15:00 | 2463.66 | 2434.41 | 2446.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 13:15:00 | 2463.66 | 2434.41 | 2446.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 2463.66 | 2434.41 | 2446.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 2463.66 | 2434.41 | 2446.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 2510.29 | 2449.59 | 2452.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 15:00:00 | 2510.29 | 2449.59 | 2452.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 174 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 2514.84 | 2462.64 | 2458.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 2523.67 | 2501.87 | 2492.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 11:15:00 | 2495.36 | 2502.21 | 2494.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 11:15:00 | 2495.36 | 2502.21 | 2494.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 2495.36 | 2502.21 | 2494.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:45:00 | 2498.36 | 2502.21 | 2494.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 2501.08 | 2501.98 | 2495.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:15:00 | 2494.19 | 2501.98 | 2495.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 2493.61 | 2500.31 | 2494.95 | EMA400 retest candle locked (from upside) |

### Cycle 175 — SELL (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 15:15:00 | 2477.81 | 2489.15 | 2490.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 2458.32 | 2482.99 | 2487.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 09:15:00 | 2467.82 | 2455.06 | 2466.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 09:15:00 | 2467.82 | 2455.06 | 2466.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 2467.82 | 2455.06 | 2466.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:30:00 | 2467.92 | 2455.06 | 2466.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 2474.13 | 2458.88 | 2467.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 2474.13 | 2458.88 | 2467.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 2469.67 | 2461.03 | 2467.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 12:15:00 | 2463.46 | 2461.03 | 2467.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 13:00:00 | 2463.75 | 2461.58 | 2467.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 14:00:00 | 2461.33 | 2461.53 | 2466.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 12:15:00 | 2473.83 | 2469.30 | 2468.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 2473.83 | 2469.30 | 2468.83 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 2447.76 | 2465.57 | 2467.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 10:15:00 | 2427.11 | 2445.98 | 2454.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 2440.58 | 2434.91 | 2445.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-14 14:30:00 | 2435.73 | 2434.91 | 2445.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 2458.90 | 2440.87 | 2446.60 | EMA400 retest candle locked (from downside) |

### Cycle 178 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 2471.12 | 2453.94 | 2451.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 2484.01 | 2463.19 | 2457.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 2473.25 | 2474.98 | 2467.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 2473.25 | 2474.98 | 2467.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 2473.25 | 2474.98 | 2467.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:45:00 | 2475.19 | 2474.98 | 2467.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 2492.74 | 2478.53 | 2469.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:30:00 | 2469.86 | 2478.53 | 2469.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 2476.74 | 2478.56 | 2472.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 2473.93 | 2478.56 | 2472.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 2471.12 | 2477.07 | 2472.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:45:00 | 2472.09 | 2477.07 | 2472.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 2467.34 | 2475.12 | 2471.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 09:15:00 | 2490.32 | 2475.12 | 2471.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-20 10:15:00 | 2480.23 | 2474.34 | 2471.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 2478.97 | 2473.29 | 2472.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 14:15:00 | 2477.71 | 2473.29 | 2472.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 2470.73 | 2472.78 | 2472.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 2470.73 | 2472.78 | 2472.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 2475.77 | 2480.11 | 2477.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:00:00 | 2475.77 | 2480.11 | 2477.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 2464.24 | 2476.93 | 2475.86 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-23 14:15:00 | 2464.24 | 2476.93 | 2475.86 | SL hit (close<static) qty=1.00 sl=2467.34 alert=retest2 |

### Cycle 179 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 2467.34 | 2475.01 | 2475.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 09:15:00 | 2456.00 | 2471.21 | 2473.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 11:15:00 | 2422.74 | 2420.61 | 2432.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 11:45:00 | 2424.00 | 2420.61 | 2432.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 2427.49 | 2421.98 | 2432.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 13:00:00 | 2427.49 | 2421.98 | 2432.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 2413.44 | 2420.27 | 2430.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:15:00 | 2406.36 | 2420.27 | 2430.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 2437.38 | 2424.10 | 2429.74 | SL hit (close>static) qty=1.00 sl=2433.41 alert=retest2 |

### Cycle 180 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 2477.91 | 2434.86 | 2434.12 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 2426.43 | 2446.94 | 2448.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 2404.81 | 2434.15 | 2441.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 12:15:00 | 2416.25 | 2413.56 | 2426.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 13:00:00 | 2416.25 | 2413.56 | 2426.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 2300.30 | 2290.19 | 2314.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 2322.60 | 2290.19 | 2314.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 2295.55 | 2293.55 | 2310.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:30:00 | 2282.17 | 2293.38 | 2303.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 2436.22 | 2320.07 | 2309.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 182 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 2436.22 | 2320.07 | 2309.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 10:15:00 | 2462.78 | 2422.61 | 2393.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 14:15:00 | 2438.25 | 2439.51 | 2412.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 15:00:00 | 2438.25 | 2439.51 | 2412.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 2443.50 | 2448.49 | 2437.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:45:00 | 2438.50 | 2448.49 | 2437.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 2435.50 | 2445.56 | 2438.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:00:00 | 2435.50 | 2445.56 | 2438.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 2439.70 | 2444.38 | 2438.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 2434.20 | 2444.38 | 2438.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 2437.60 | 2443.03 | 2438.24 | EMA400 retest candle locked (from upside) |

### Cycle 183 — SELL (started 2025-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 15:15:00 | 2429.60 | 2435.86 | 2435.88 | EMA200 below EMA400 |

### Cycle 184 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 2462.10 | 2441.11 | 2438.26 | EMA200 above EMA400 |

### Cycle 185 — SELL (started 2025-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 11:15:00 | 2431.70 | 2439.89 | 2440.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 14:15:00 | 2425.90 | 2434.40 | 2437.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 09:15:00 | 2297.90 | 2287.37 | 2316.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-28 10:00:00 | 2297.90 | 2287.37 | 2316.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 2320.80 | 2297.47 | 2316.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:30:00 | 2323.50 | 2297.47 | 2316.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 2308.70 | 2299.71 | 2315.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 13:30:00 | 2303.10 | 2298.49 | 2313.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-03 13:15:00 | 2187.94 | 2219.22 | 2244.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 2216.40 | 2211.06 | 2234.01 | SL hit (close>ema200) qty=0.50 sl=2211.06 alert=retest2 |

### Cycle 186 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 2256.60 | 2230.43 | 2229.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 2266.50 | 2237.65 | 2232.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 2242.60 | 2242.69 | 2236.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 10:00:00 | 2242.60 | 2242.69 | 2236.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 11:15:00 | 2236.10 | 2243.50 | 2237.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 12:00:00 | 2236.10 | 2243.50 | 2237.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 2227.00 | 2240.20 | 2236.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 2227.00 | 2240.20 | 2236.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 187 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 2208.20 | 2233.80 | 2234.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 2199.90 | 2221.61 | 2228.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 2244.20 | 2225.74 | 2228.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 11:15:00 | 2244.20 | 2225.74 | 2228.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 2244.20 | 2225.74 | 2228.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 2244.20 | 2225.74 | 2228.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 2248.80 | 2230.36 | 2230.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 2248.80 | 2230.36 | 2230.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — BUY (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 13:15:00 | 2249.00 | 2234.08 | 2232.30 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 14:15:00 | 2213.00 | 2233.45 | 2234.70 | EMA200 below EMA400 |

### Cycle 190 — BUY (started 2025-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 12:15:00 | 2244.10 | 2235.07 | 2234.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 13:15:00 | 2278.80 | 2243.82 | 2238.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 14:15:00 | 2279.00 | 2282.12 | 2272.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 15:00:00 | 2279.00 | 2282.12 | 2272.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 2253.40 | 2275.43 | 2271.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 09:30:00 | 2255.00 | 2275.43 | 2271.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 2249.00 | 2270.14 | 2269.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 2249.00 | 2270.14 | 2269.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 191 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 2246.20 | 2265.35 | 2266.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 2235.30 | 2259.34 | 2264.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 2241.80 | 2235.03 | 2242.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 2241.80 | 2235.03 | 2242.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 2241.80 | 2235.03 | 2242.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 11:45:00 | 2242.50 | 2235.03 | 2242.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 2242.60 | 2236.55 | 2242.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:30:00 | 2241.50 | 2236.55 | 2242.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 2229.70 | 2235.18 | 2241.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:15:00 | 2222.30 | 2235.18 | 2241.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 15:15:00 | 2222.70 | 2233.96 | 2240.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 2224.70 | 2230.75 | 2237.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 15:15:00 | 2243.90 | 2231.84 | 2234.72 | SL hit (close>static) qty=1.00 sl=2243.00 alert=retest2 |

### Cycle 192 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 2257.90 | 2237.05 | 2236.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 13:15:00 | 2268.50 | 2254.26 | 2246.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 11:15:00 | 2259.60 | 2260.48 | 2252.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 11:30:00 | 2261.10 | 2260.48 | 2252.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 2251.60 | 2258.71 | 2252.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 2251.60 | 2258.71 | 2252.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 2255.50 | 2258.06 | 2252.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:30:00 | 2252.50 | 2258.06 | 2252.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 2251.40 | 2256.73 | 2252.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 2251.90 | 2256.73 | 2252.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 2246.00 | 2254.59 | 2252.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 2257.20 | 2254.59 | 2252.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 2248.10 | 2252.25 | 2251.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:45:00 | 2247.20 | 2252.25 | 2251.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2025-12-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 11:15:00 | 2240.70 | 2249.94 | 2250.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 12:15:00 | 2238.50 | 2247.65 | 2249.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 09:15:00 | 2239.20 | 2236.55 | 2242.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 09:15:00 | 2239.20 | 2236.55 | 2242.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 2239.20 | 2236.55 | 2242.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:30:00 | 2238.10 | 2236.55 | 2242.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 10:15:00 | 2235.10 | 2236.26 | 2242.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 10:30:00 | 2246.60 | 2236.26 | 2242.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 2230.00 | 2233.17 | 2237.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:45:00 | 2233.70 | 2233.17 | 2237.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 2217.30 | 2211.47 | 2218.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 2217.30 | 2211.47 | 2218.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 2226.20 | 2215.11 | 2219.00 | EMA400 retest candle locked (from downside) |

### Cycle 194 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 2235.60 | 2222.13 | 2221.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 12:15:00 | 2238.70 | 2225.45 | 2223.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 2280.00 | 2280.67 | 2268.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 14:00:00 | 2280.00 | 2280.67 | 2268.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 2265.00 | 2277.07 | 2270.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 11:00:00 | 2265.00 | 2277.07 | 2270.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 2271.10 | 2275.87 | 2270.84 | EMA400 retest candle locked (from upside) |

### Cycle 195 — SELL (started 2026-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 15:15:00 | 2262.00 | 2267.21 | 2267.87 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 2276.10 | 2268.58 | 2268.36 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 09:15:00 | 2253.10 | 2268.31 | 2268.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 2234.90 | 2261.63 | 2265.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 2176.00 | 2162.44 | 2189.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:45:00 | 2175.00 | 2162.44 | 2189.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 2163.70 | 2166.49 | 2182.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 2148.10 | 2161.32 | 2175.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:30:00 | 2151.20 | 2158.81 | 2170.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:15:00 | 2151.70 | 2159.16 | 2167.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:15:00 | 2152.00 | 2157.20 | 2165.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 2179.30 | 2160.79 | 2165.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:00:00 | 2179.30 | 2160.79 | 2165.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 2185.80 | 2165.79 | 2167.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:30:00 | 2184.40 | 2165.79 | 2167.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-16 11:15:00 | 2189.20 | 2170.47 | 2169.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 198 — BUY (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 11:15:00 | 2189.20 | 2170.47 | 2169.36 | EMA200 above EMA400 |

### Cycle 199 — SELL (started 2026-01-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 14:15:00 | 2156.30 | 2167.79 | 2168.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 2146.60 | 2162.15 | 2165.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 2089.60 | 2055.79 | 2079.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 2089.60 | 2055.79 | 2079.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 2089.60 | 2055.79 | 2079.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 2092.10 | 2055.79 | 2079.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 2070.20 | 2058.67 | 2078.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:45:00 | 2063.50 | 2073.61 | 2078.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:15:00 | 2064.00 | 2073.61 | 2078.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 12:15:00 | 1960.32 | 2046.48 | 2065.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-23 12:15:00 | 1960.80 | 2046.48 | 2065.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-23 14:15:00 | 1857.15 | 1985.50 | 2032.76 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 200 — BUY (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 13:15:00 | 2018.50 | 1991.22 | 1990.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 09:15:00 | 2030.00 | 2006.04 | 1998.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 2010.10 | 2014.83 | 2007.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 2010.10 | 2014.83 | 2007.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 2010.10 | 2014.83 | 2007.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 11:15:00 | 2027.00 | 2015.07 | 2008.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 13:15:00 | 1965.80 | 2000.32 | 2003.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 201 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 1965.80 | 2000.32 | 2003.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 1943.60 | 1988.98 | 1997.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 1982.80 | 1971.89 | 1983.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 12:15:00 | 1982.80 | 1971.89 | 1983.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 1982.80 | 1971.89 | 1983.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 13:00:00 | 1982.80 | 1971.89 | 1983.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 1978.70 | 1973.25 | 1983.36 | EMA400 retest candle locked (from downside) |

### Cycle 202 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 2181.90 | 2021.22 | 2003.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 11:15:00 | 2239.10 | 2189.37 | 2124.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 11:15:00 | 2214.50 | 2215.40 | 2172.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 12:00:00 | 2214.50 | 2215.40 | 2172.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 13:15:00 | 2215.70 | 2219.19 | 2200.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:30:00 | 2208.90 | 2219.19 | 2200.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 2217.50 | 2240.51 | 2229.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:30:00 | 2209.00 | 2240.51 | 2229.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 2245.50 | 2241.51 | 2230.80 | EMA400 retest candle locked (from upside) |

### Cycle 203 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 2222.50 | 2227.25 | 2227.74 | EMA200 below EMA400 |

### Cycle 204 — BUY (started 2026-02-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 14:15:00 | 2235.40 | 2229.41 | 2228.68 | EMA200 above EMA400 |

### Cycle 205 — SELL (started 2026-02-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 09:15:00 | 2215.00 | 2226.62 | 2227.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 10:15:00 | 2208.30 | 2222.96 | 2225.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 11:15:00 | 2171.70 | 2154.15 | 2174.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 12:00:00 | 2171.70 | 2154.15 | 2174.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 12:15:00 | 2180.30 | 2159.38 | 2175.20 | EMA400 retest candle locked (from downside) |

### Cycle 206 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 2199.80 | 2183.36 | 2181.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 2234.20 | 2193.53 | 2186.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 09:15:00 | 2210.40 | 2214.87 | 2200.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 10:00:00 | 2210.40 | 2214.87 | 2200.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 2197.60 | 2211.41 | 2200.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:45:00 | 2198.60 | 2211.41 | 2200.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 2207.70 | 2210.67 | 2200.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 15:00:00 | 2211.80 | 2209.74 | 2202.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 09:15:00 | 2193.30 | 2206.98 | 2202.81 | SL hit (close<static) qty=1.00 sl=2193.80 alert=retest2 |

### Cycle 207 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 2166.90 | 2198.96 | 2199.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 14:15:00 | 2156.20 | 2178.44 | 2188.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 2165.00 | 2164.43 | 2175.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 14:00:00 | 2165.00 | 2164.43 | 2175.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 2192.20 | 2169.31 | 2174.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:45:00 | 2198.60 | 2169.31 | 2174.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 2188.90 | 2173.23 | 2176.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:45:00 | 2180.00 | 2174.76 | 2176.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 13:15:00 | 2180.60 | 2176.93 | 2177.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 14:15:00 | 2192.90 | 2178.68 | 2178.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 208 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 2192.90 | 2178.68 | 2178.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 2200.90 | 2185.55 | 2182.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 2208.90 | 2211.94 | 2201.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 11:45:00 | 2207.40 | 2211.94 | 2201.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 13:15:00 | 2203.40 | 2209.51 | 2201.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:00:00 | 2203.40 | 2209.51 | 2201.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 2216.60 | 2210.93 | 2203.15 | EMA400 retest candle locked (from upside) |

### Cycle 209 — SELL (started 2026-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 12:15:00 | 2190.90 | 2200.52 | 2200.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 14:15:00 | 2160.20 | 2191.39 | 2196.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 14:15:00 | 2127.50 | 2126.24 | 2153.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-02 15:00:00 | 2127.50 | 2126.24 | 2153.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 2080.50 | 2064.03 | 2088.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 2080.50 | 2064.03 | 2088.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 2080.00 | 2067.23 | 2087.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 2084.20 | 2069.38 | 2086.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 2062.90 | 2068.09 | 2084.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:15:00 | 2059.00 | 2068.27 | 2081.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:45:00 | 2061.50 | 2066.45 | 2079.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1956.05 | 2040.69 | 2063.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 1958.42 | 2040.69 | 2063.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 13:15:00 | 2000.10 | 1997.12 | 2016.67 | SL hit (close>ema200) qty=0.50 sl=1997.12 alert=retest2 |

### Cycle 210 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 2010.00 | 1978.91 | 1976.35 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 1948.40 | 1984.01 | 1984.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 1941.40 | 1970.47 | 1977.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 1959.20 | 1955.14 | 1966.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 1959.20 | 1955.14 | 1966.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 1959.20 | 1955.14 | 1966.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 1964.50 | 1955.14 | 1966.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 1852.20 | 1856.22 | 1892.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:15:00 | 1848.30 | 1856.22 | 1892.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 10:15:00 | 1907.00 | 1857.89 | 1870.56 | SL hit (close>static) qty=1.00 sl=1893.00 alert=retest2 |

### Cycle 212 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 1891.50 | 1877.76 | 1877.72 | EMA200 above EMA400 |

### Cycle 213 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 1849.00 | 1874.30 | 1876.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 1832.80 | 1861.87 | 1870.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 1850.40 | 1803.28 | 1822.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 1850.40 | 1803.28 | 1822.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 1850.40 | 1803.28 | 1822.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 1850.40 | 1803.28 | 1822.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 1844.60 | 1811.54 | 1824.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:45:00 | 1856.00 | 1811.54 | 1824.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 214 — BUY (started 2026-04-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 13:15:00 | 1854.90 | 1835.31 | 1833.76 | EMA200 above EMA400 |

### Cycle 215 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 1786.20 | 1827.32 | 1830.69 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 1839.20 | 1827.57 | 1826.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 1863.40 | 1834.74 | 1829.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 1868.70 | 1870.30 | 1852.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 10:45:00 | 1866.90 | 1870.30 | 1852.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2124.80 | 2082.59 | 2043.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 2144.90 | 2082.59 | 2043.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 2132.10 | 2093.87 | 2052.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 14:45:00 | 2130.00 | 2117.77 | 2078.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 15:15:00 | 2130.20 | 2117.77 | 2078.71 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 2249.80 | 2247.03 | 2235.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 10:15:00 | 2261.90 | 2247.03 | 2235.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 14:00:00 | 2261.70 | 2263.29 | 2259.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-28 09:15:00 | 2359.39 | 2328.13 | 2301.77 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-18 14:30:00 | 1837.17 | 2023-05-19 13:15:00 | 1874.40 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2023-05-18 15:00:00 | 1835.57 | 2023-05-19 13:15:00 | 1874.40 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2023-05-19 09:30:00 | 1831.36 | 2023-05-19 13:15:00 | 1874.40 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2023-05-19 12:00:00 | 1835.24 | 2023-05-19 13:15:00 | 1874.40 | STOP_HIT | 1.00 | -2.13% |
| BUY | retest2 | 2023-05-25 12:15:00 | 2368.45 | 2023-05-30 15:15:00 | 2419.83 | STOP_HIT | 1.00 | 2.17% |
| SELL | retest2 | 2023-06-05 12:00:00 | 2398.12 | 2023-06-09 15:15:00 | 2379.12 | STOP_HIT | 1.00 | 0.79% |
| BUY | retest2 | 2023-06-14 09:45:00 | 2400.44 | 2023-06-14 15:15:00 | 2382.02 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2023-06-22 11:15:00 | 2318.18 | 2023-06-23 09:15:00 | 2202.27 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-06-22 11:15:00 | 2318.18 | 2023-06-26 09:15:00 | 2218.67 | STOP_HIT | 0.50 | 4.29% |
| SELL | retest2 | 2023-07-13 12:00:00 | 2311.79 | 2023-07-17 09:15:00 | 2352.31 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2023-07-13 13:15:00 | 2309.85 | 2023-07-17 09:15:00 | 2352.31 | STOP_HIT | 1.00 | -1.84% |
| BUY | retest2 | 2023-07-19 09:15:00 | 2354.20 | 2023-07-24 14:15:00 | 2342.62 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2023-07-19 09:45:00 | 2352.46 | 2023-07-24 15:15:00 | 2342.28 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2023-07-19 11:30:00 | 2356.14 | 2023-07-24 15:15:00 | 2342.28 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2023-07-19 13:30:00 | 2354.39 | 2023-07-24 15:15:00 | 2342.28 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2023-07-20 11:15:00 | 2348.09 | 2023-07-24 15:15:00 | 2342.28 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2023-07-20 14:00:00 | 2352.41 | 2023-07-24 15:15:00 | 2342.28 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2023-07-21 09:30:00 | 2354.20 | 2023-07-24 15:15:00 | 2342.28 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2023-07-24 09:45:00 | 2350.03 | 2023-07-24 15:15:00 | 2342.28 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2023-07-24 13:45:00 | 2360.50 | 2023-07-24 15:15:00 | 2342.28 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2023-08-02 10:45:00 | 2434.43 | 2023-08-02 13:15:00 | 2383.62 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2023-08-03 10:30:00 | 2444.51 | 2023-08-08 12:15:00 | 2401.51 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2023-08-03 15:00:00 | 2462.20 | 2023-08-08 12:15:00 | 2401.51 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2023-08-04 09:45:00 | 2457.98 | 2023-08-08 12:15:00 | 2401.51 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2023-09-13 09:15:00 | 2482.32 | 2023-09-13 09:15:00 | 2431.42 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2023-09-21 11:15:00 | 2402.48 | 2023-09-25 14:15:00 | 2410.24 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2023-09-21 12:30:00 | 2402.63 | 2023-09-27 12:15:00 | 2415.33 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2023-09-22 09:15:00 | 2404.71 | 2023-09-27 12:15:00 | 2415.33 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2023-09-22 11:15:00 | 2400.88 | 2023-09-27 12:15:00 | 2415.33 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2023-09-25 11:15:00 | 2363.41 | 2023-09-27 12:15:00 | 2415.33 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2023-10-12 09:15:00 | 2438.69 | 2023-10-13 09:15:00 | 2360.60 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest2 | 2023-10-12 10:00:00 | 2423.42 | 2023-10-13 09:15:00 | 2360.60 | STOP_HIT | 1.00 | -2.59% |
| BUY | retest2 | 2023-10-12 10:30:00 | 2426.48 | 2023-10-13 09:15:00 | 2360.60 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2023-10-23 09:45:00 | 2305.48 | 2023-10-25 12:15:00 | 2190.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-23 09:45:00 | 2305.48 | 2023-10-27 09:15:00 | 2171.31 | STOP_HIT | 0.50 | 5.82% |
| SELL | retest2 | 2023-11-15 10:45:00 | 2156.72 | 2023-11-15 14:15:00 | 2157.98 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2023-11-15 13:30:00 | 2156.13 | 2023-11-15 14:15:00 | 2157.98 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2023-11-15 14:00:00 | 2155.26 | 2023-11-15 14:15:00 | 2157.98 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2023-11-22 10:15:00 | 2106.79 | 2023-11-24 13:15:00 | 2147.65 | STOP_HIT | 1.00 | -1.94% |
| SELL | retest2 | 2023-11-23 10:30:00 | 2102.47 | 2023-11-24 13:15:00 | 2147.65 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2023-11-23 15:00:00 | 2108.05 | 2023-11-24 13:15:00 | 2147.65 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2023-11-24 09:45:00 | 2100.20 | 2023-11-24 13:15:00 | 2147.65 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2023-12-04 09:15:00 | 2444.12 | 2023-12-05 10:15:00 | 2688.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-12-19 11:30:00 | 2863.76 | 2023-12-20 12:15:00 | 2798.90 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2023-12-20 09:15:00 | 2859.98 | 2023-12-20 12:15:00 | 2798.90 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2023-12-26 13:00:00 | 2766.72 | 2023-12-26 14:15:00 | 2793.38 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2023-12-26 14:00:00 | 2761.67 | 2023-12-26 14:15:00 | 2793.38 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2023-12-28 14:15:00 | 2744.22 | 2023-12-29 14:15:00 | 2764.05 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2023-12-29 10:00:00 | 2737.29 | 2023-12-29 14:15:00 | 2764.05 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2023-12-29 12:45:00 | 2746.70 | 2023-12-29 14:15:00 | 2764.05 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2023-12-29 13:15:00 | 2749.51 | 2023-12-29 14:15:00 | 2764.05 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-01-02 14:00:00 | 2835.60 | 2024-01-08 14:15:00 | 2864.78 | STOP_HIT | 1.00 | 1.03% |
| SELL | retest2 | 2024-01-25 10:15:00 | 2779.51 | 2024-01-29 09:15:00 | 2953.83 | STOP_HIT | 1.00 | -6.27% |
| BUY | retest2 | 2024-02-06 10:30:00 | 3123.15 | 2024-02-08 13:15:00 | 3069.49 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-02-06 13:00:00 | 3095.57 | 2024-02-08 13:15:00 | 3069.49 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-02-16 09:15:00 | 3123.78 | 2024-02-22 10:15:00 | 3121.74 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2024-02-23 14:15:00 | 3208.03 | 2024-02-28 11:15:00 | 3154.56 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2024-02-26 09:15:00 | 3191.01 | 2024-02-28 11:15:00 | 3154.56 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-02-27 14:30:00 | 3196.20 | 2024-02-28 11:15:00 | 3154.56 | STOP_HIT | 1.00 | -1.30% |
| SELL | retest2 | 2024-03-11 11:45:00 | 3125.91 | 2024-03-13 09:15:00 | 2969.61 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-03-11 11:45:00 | 3125.91 | 2024-03-13 13:15:00 | 2813.32 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-03-18 12:15:00 | 2995.56 | 2024-03-19 11:15:00 | 2971.47 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-03-27 09:15:00 | 3029.64 | 2024-04-04 11:15:00 | 3117.77 | STOP_HIT | 1.00 | 2.91% |
| BUY | retest2 | 2024-03-27 10:15:00 | 3021.84 | 2024-04-04 14:15:00 | 3108.90 | STOP_HIT | 1.00 | 2.88% |
| BUY | retest2 | 2024-03-27 13:00:00 | 3024.26 | 2024-04-04 14:15:00 | 3108.90 | STOP_HIT | 1.00 | 2.80% |
| BUY | retest2 | 2024-03-27 13:30:00 | 3028.43 | 2024-04-04 14:15:00 | 3108.90 | STOP_HIT | 1.00 | 2.66% |
| BUY | retest2 | 2024-04-04 09:15:00 | 3150.78 | 2024-04-04 14:15:00 | 3108.90 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2024-04-23 09:15:00 | 2964.69 | 2024-04-23 11:15:00 | 2990.76 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-04-23 13:00:00 | 2963.72 | 2024-04-24 10:15:00 | 2984.07 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2024-04-23 15:15:00 | 2964.69 | 2024-04-24 10:15:00 | 2984.07 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2024-05-09 10:30:00 | 2757.07 | 2024-05-13 14:15:00 | 2792.07 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-05-09 11:00:00 | 2753.34 | 2024-05-13 14:15:00 | 2792.07 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2024-05-13 13:15:00 | 2755.86 | 2024-05-13 14:15:00 | 2792.07 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2024-05-17 10:00:00 | 2965.56 | 2024-05-23 12:15:00 | 3262.12 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-21 09:45:00 | 2958.29 | 2024-05-23 12:15:00 | 3254.12 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-05-29 15:00:00 | 3155.63 | 2024-05-31 11:15:00 | 3210.93 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2024-06-06 11:30:00 | 3086.55 | 2024-06-07 11:15:00 | 3099.10 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2024-06-06 14:30:00 | 3085.48 | 2024-06-07 11:15:00 | 3099.10 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2024-06-07 11:15:00 | 3089.94 | 2024-06-07 11:15:00 | 3099.10 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2024-06-11 12:15:00 | 3148.16 | 2024-06-21 09:15:00 | 3121.50 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-06-11 13:00:00 | 3147.19 | 2024-06-21 09:15:00 | 3121.50 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2024-06-11 14:15:00 | 3145.98 | 2024-06-21 09:15:00 | 3121.50 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-06-14 09:15:00 | 3152.28 | 2024-06-21 09:15:00 | 3121.50 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2024-06-14 11:00:00 | 3165.37 | 2024-06-21 09:15:00 | 3121.50 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2024-06-14 15:00:00 | 3169.68 | 2024-06-21 09:15:00 | 3121.50 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-06-19 11:45:00 | 3162.46 | 2024-06-21 09:15:00 | 3121.50 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-06-19 14:30:00 | 3160.96 | 2024-06-21 09:15:00 | 3121.50 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-06-20 10:15:00 | 3204.29 | 2024-06-21 09:15:00 | 3121.50 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2024-06-28 12:15:00 | 3067.50 | 2024-07-02 10:15:00 | 3093.38 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2024-06-28 14:30:00 | 3069.53 | 2024-07-02 10:15:00 | 3093.38 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2024-07-01 11:00:00 | 3072.39 | 2024-07-02 10:15:00 | 3093.38 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2024-07-01 11:30:00 | 3071.81 | 2024-07-02 10:15:00 | 3093.38 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-07-08 10:30:00 | 3019.41 | 2024-07-16 09:15:00 | 3025.33 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2024-07-08 15:15:00 | 3017.04 | 2024-07-16 09:15:00 | 3025.33 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2024-07-09 11:45:00 | 3013.16 | 2024-07-16 09:15:00 | 3025.33 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2024-07-10 09:30:00 | 3017.04 | 2024-07-16 09:15:00 | 3025.33 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2024-07-11 14:45:00 | 2985.34 | 2024-07-16 09:15:00 | 3025.33 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-07-12 09:30:00 | 2990.76 | 2024-07-16 09:15:00 | 3025.33 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-07-23 12:15:00 | 2836.86 | 2024-07-26 10:15:00 | 2950.68 | STOP_HIT | 1.00 | -4.01% |
| SELL | retest2 | 2024-07-23 14:15:00 | 2907.58 | 2024-07-26 10:15:00 | 2950.68 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-07-23 15:00:00 | 2899.68 | 2024-07-26 10:15:00 | 2950.68 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2024-07-25 10:45:00 | 2906.66 | 2024-07-26 10:15:00 | 2950.68 | STOP_HIT | 1.00 | -1.51% |
| BUY | retest1 | 2024-07-30 11:45:00 | 3013.79 | 2024-08-02 09:15:00 | 3047.62 | STOP_HIT | 1.00 | 1.12% |
| BUY | retest2 | 2024-08-02 11:15:00 | 3112.58 | 2024-08-05 09:15:00 | 3014.03 | STOP_HIT | 1.00 | -3.17% |
| BUY | retest2 | 2024-08-02 13:15:00 | 3108.75 | 2024-08-05 09:15:00 | 3014.03 | STOP_HIT | 1.00 | -3.05% |
| SELL | retest2 | 2024-08-06 14:00:00 | 2995.71 | 2024-08-07 09:15:00 | 3040.74 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-08-09 09:15:00 | 3107.15 | 2024-08-12 09:15:00 | 2977.43 | STOP_HIT | 1.00 | -4.17% |
| SELL | retest2 | 2024-08-27 12:45:00 | 2981.85 | 2024-09-03 10:15:00 | 2937.88 | STOP_HIT | 1.00 | 1.47% |
| SELL | retest2 | 2024-08-27 13:15:00 | 2980.39 | 2024-09-03 10:15:00 | 2937.88 | STOP_HIT | 1.00 | 1.43% |
| SELL | retest2 | 2024-09-06 14:00:00 | 2898.76 | 2024-09-13 09:15:00 | 2887.27 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2024-09-10 12:45:00 | 2899.34 | 2024-09-13 09:15:00 | 2887.27 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2024-09-10 14:30:00 | 2900.21 | 2024-09-13 09:15:00 | 2887.27 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2024-09-12 15:15:00 | 2889.94 | 2024-09-13 09:15:00 | 2887.27 | STOP_HIT | 1.00 | 0.09% |
| BUY | retest2 | 2024-10-01 12:15:00 | 3048.64 | 2024-10-03 13:15:00 | 3008.31 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-10-11 12:15:00 | 3043.31 | 2024-10-14 12:15:00 | 3014.81 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2024-10-14 09:30:00 | 3047.96 | 2024-10-14 12:15:00 | 3014.81 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2024-10-16 11:30:00 | 2988.10 | 2024-10-21 11:15:00 | 2842.98 | PARTIAL | 0.50 | 4.86% |
| SELL | retest2 | 2024-10-16 15:00:00 | 2992.61 | 2024-10-21 12:15:00 | 2838.69 | PARTIAL | 0.50 | 5.14% |
| SELL | retest2 | 2024-10-16 11:30:00 | 2988.10 | 2024-10-23 10:15:00 | 2785.86 | STOP_HIT | 0.50 | 6.77% |
| SELL | retest2 | 2024-10-16 15:00:00 | 2992.61 | 2024-10-23 10:15:00 | 2785.86 | STOP_HIT | 0.50 | 6.91% |
| SELL | retest1 | 2024-11-25 15:00:00 | 2186.14 | 2024-11-26 13:15:00 | 2076.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-11-26 09:30:00 | 2180.95 | 2024-11-26 14:15:00 | 2071.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2024-11-25 15:00:00 | 2186.14 | 2024-11-27 09:15:00 | 2161.95 | STOP_HIT | 0.50 | 1.11% |
| SELL | retest1 | 2024-11-26 09:30:00 | 2180.95 | 2024-11-27 09:15:00 | 2161.95 | STOP_HIT | 0.50 | 0.87% |
| BUY | retest2 | 2024-12-05 12:00:00 | 2437.87 | 2024-12-09 15:15:00 | 2418.67 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-12-09 11:15:00 | 2431.47 | 2024-12-09 15:15:00 | 2418.67 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-12-09 14:45:00 | 2423.71 | 2024-12-09 15:15:00 | 2418.67 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2024-12-16 14:45:00 | 2443.30 | 2024-12-16 15:15:00 | 2430.50 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-12-20 13:00:00 | 2339.03 | 2024-12-26 14:15:00 | 2324.29 | STOP_HIT | 1.00 | 0.63% |
| SELL | retest2 | 2025-01-08 11:45:00 | 2420.46 | 2025-01-10 14:15:00 | 2299.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-09 10:30:00 | 2419.83 | 2025-01-10 14:15:00 | 2298.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 11:45:00 | 2420.46 | 2025-01-13 13:15:00 | 2178.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-09 10:30:00 | 2419.83 | 2025-01-13 13:15:00 | 2177.85 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-01-20 10:45:00 | 2343.78 | 2025-01-21 13:15:00 | 2328.17 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-01-20 11:30:00 | 2345.38 | 2025-01-21 13:15:00 | 2328.17 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-01-21 11:45:00 | 2344.46 | 2025-01-21 13:15:00 | 2328.17 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-01-24 09:45:00 | 2290.89 | 2025-01-28 09:15:00 | 2176.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 12:30:00 | 2289.39 | 2025-01-28 09:15:00 | 2174.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 09:45:00 | 2290.89 | 2025-01-28 11:15:00 | 2219.15 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2025-01-24 12:30:00 | 2289.39 | 2025-01-28 11:15:00 | 2219.15 | STOP_HIT | 0.50 | 3.07% |
| BUY | retest2 | 2025-02-07 09:15:00 | 2251.14 | 2025-02-10 10:15:00 | 2227.34 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-02-10 09:45:00 | 2244.79 | 2025-02-10 10:15:00 | 2227.34 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-02-18 09:15:00 | 2147.07 | 2025-02-24 09:15:00 | 2039.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-18 15:15:00 | 2145.52 | 2025-02-24 09:15:00 | 2038.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-18 09:15:00 | 2147.07 | 2025-02-25 09:15:00 | 2073.44 | STOP_HIT | 0.50 | 3.43% |
| SELL | retest2 | 2025-02-18 15:15:00 | 2145.52 | 2025-02-25 09:15:00 | 2073.44 | STOP_HIT | 0.50 | 3.36% |
| BUY | retest2 | 2025-03-12 09:30:00 | 2191.04 | 2025-03-12 10:15:00 | 2170.77 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-03-21 09:15:00 | 2280.13 | 2025-03-25 12:15:00 | 2263.60 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-03-26 13:30:00 | 2257.25 | 2025-03-27 14:15:00 | 2291.72 | STOP_HIT | 1.00 | -1.53% |
| SELL | retest2 | 2025-03-26 14:15:00 | 2255.12 | 2025-03-27 14:15:00 | 2291.72 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-04-08 10:30:00 | 2161.71 | 2025-04-11 09:15:00 | 2257.74 | STOP_HIT | 1.00 | -4.44% |
| SELL | retest2 | 2025-04-08 11:15:00 | 2170.82 | 2025-04-11 09:15:00 | 2257.74 | STOP_HIT | 1.00 | -4.00% |
| SELL | retest2 | 2025-04-09 09:45:00 | 2162.68 | 2025-04-11 09:15:00 | 2257.74 | STOP_HIT | 1.00 | -4.40% |
| SELL | retest2 | 2025-04-09 13:45:00 | 2173.10 | 2025-04-11 09:15:00 | 2257.74 | STOP_HIT | 1.00 | -3.89% |
| BUY | retest2 | 2025-04-24 10:15:00 | 2383.87 | 2025-04-24 12:15:00 | 2356.14 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-04-29 11:45:00 | 2273.73 | 2025-05-05 10:15:00 | 2354.59 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2025-05-22 11:15:00 | 2422.36 | 2025-05-23 10:15:00 | 2446.98 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-05-27 11:15:00 | 2483.82 | 2025-05-28 12:15:00 | 2445.53 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-05-27 14:30:00 | 2473.35 | 2025-05-28 12:15:00 | 2445.53 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-06-12 09:15:00 | 2514.46 | 2025-06-12 10:15:00 | 2484.31 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-06-17 10:00:00 | 2445.14 | 2025-06-23 12:15:00 | 2402.29 | STOP_HIT | 1.00 | 1.75% |
| SELL | retest2 | 2025-06-17 10:45:00 | 2442.81 | 2025-06-23 12:15:00 | 2402.29 | STOP_HIT | 1.00 | 1.66% |
| SELL | retest2 | 2025-06-17 11:45:00 | 2436.12 | 2025-06-23 12:15:00 | 2402.29 | STOP_HIT | 1.00 | 1.39% |
| BUY | retest2 | 2025-07-03 13:30:00 | 2546.35 | 2025-07-03 15:15:00 | 2528.42 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-07-08 11:00:00 | 2500.30 | 2025-07-09 10:15:00 | 2521.82 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-07-08 11:30:00 | 2495.55 | 2025-07-09 10:15:00 | 2521.82 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-07-09 13:00:00 | 2498.56 | 2025-07-15 09:15:00 | 2511.84 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-07-09 13:30:00 | 2500.30 | 2025-07-15 09:15:00 | 2511.84 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-07-10 10:45:00 | 2501.17 | 2025-07-15 09:15:00 | 2511.84 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-07-10 12:45:00 | 2504.37 | 2025-07-15 09:15:00 | 2511.84 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-07-10 14:45:00 | 2501.85 | 2025-07-15 09:15:00 | 2511.84 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-07-14 11:15:00 | 2502.63 | 2025-07-15 09:15:00 | 2511.84 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-07-28 12:00:00 | 2476.84 | 2025-07-31 14:15:00 | 2353.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-28 12:30:00 | 2474.80 | 2025-07-31 14:15:00 | 2351.06 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-28 12:00:00 | 2476.84 | 2025-08-06 11:15:00 | 2229.16 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-28 12:30:00 | 2474.80 | 2025-08-06 14:15:00 | 2227.32 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-08-13 13:15:00 | 2213.14 | 2025-08-22 11:15:00 | 2259.77 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest2 | 2025-08-13 14:30:00 | 2212.27 | 2025-08-22 11:15:00 | 2259.77 | STOP_HIT | 1.00 | 2.15% |
| BUY | retest2 | 2025-08-13 15:15:00 | 2217.21 | 2025-08-22 11:15:00 | 2259.77 | STOP_HIT | 1.00 | 1.92% |
| BUY | retest2 | 2025-08-14 10:00:00 | 2210.81 | 2025-08-22 11:15:00 | 2259.77 | STOP_HIT | 1.00 | 2.21% |
| BUY | retest2 | 2025-08-14 12:45:00 | 2224.87 | 2025-08-22 11:15:00 | 2259.77 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2025-08-18 09:15:00 | 2252.79 | 2025-08-22 11:15:00 | 2259.77 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2025-08-28 12:15:00 | 2222.45 | 2025-09-02 10:15:00 | 2224.87 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-09-16 09:15:00 | 2322.21 | 2025-09-22 13:15:00 | 2554.43 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-10-09 12:15:00 | 2463.46 | 2025-10-10 12:15:00 | 2473.83 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-10-09 13:00:00 | 2463.75 | 2025-10-10 12:15:00 | 2473.83 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-10-09 14:00:00 | 2461.33 | 2025-10-10 12:15:00 | 2473.83 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-10-20 09:15:00 | 2490.32 | 2025-10-23 14:15:00 | 2464.24 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-10-20 10:15:00 | 2480.23 | 2025-10-23 14:15:00 | 2464.24 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-10-21 13:45:00 | 2478.97 | 2025-10-23 14:15:00 | 2464.24 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-10-21 14:15:00 | 2477.71 | 2025-10-23 14:15:00 | 2464.24 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-10-28 14:15:00 | 2406.36 | 2025-10-29 09:15:00 | 2437.38 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-11-11 09:30:00 | 2282.17 | 2025-11-12 09:15:00 | 2436.22 | STOP_HIT | 1.00 | -6.75% |
| SELL | retest2 | 2025-11-28 13:30:00 | 2303.10 | 2025-12-03 13:15:00 | 2187.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-28 13:30:00 | 2303.10 | 2025-12-04 09:15:00 | 2216.40 | STOP_HIT | 0.50 | 3.76% |
| SELL | retest2 | 2025-12-18 14:15:00 | 2222.30 | 2025-12-19 15:15:00 | 2243.90 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-12-18 15:15:00 | 2222.70 | 2025-12-19 15:15:00 | 2243.90 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-12-19 11:15:00 | 2224.70 | 2025-12-19 15:15:00 | 2243.90 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-01-13 14:00:00 | 2148.10 | 2026-01-16 11:15:00 | 2189.20 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2026-01-14 09:30:00 | 2151.20 | 2026-01-16 11:15:00 | 2189.20 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2026-01-14 13:15:00 | 2151.70 | 2026-01-16 11:15:00 | 2189.20 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2026-01-14 15:15:00 | 2152.00 | 2026-01-16 11:15:00 | 2189.20 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2026-01-23 10:45:00 | 2063.50 | 2026-01-23 12:15:00 | 1960.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 11:15:00 | 2064.00 | 2026-01-23 12:15:00 | 1960.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 10:45:00 | 2063.50 | 2026-01-23 14:15:00 | 1857.15 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-23 11:15:00 | 2064.00 | 2026-01-23 14:15:00 | 1857.60 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-01 11:15:00 | 2027.00 | 2026-02-01 13:15:00 | 1965.80 | STOP_HIT | 1.00 | -3.02% |
| BUY | retest2 | 2026-02-18 15:00:00 | 2211.80 | 2026-02-19 09:15:00 | 2193.30 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2026-02-23 11:45:00 | 2180.00 | 2026-02-23 14:15:00 | 2192.90 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2026-02-23 13:15:00 | 2180.60 | 2026-02-23 14:15:00 | 2192.90 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2026-03-06 13:15:00 | 2059.00 | 2026-03-09 09:15:00 | 1956.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 13:45:00 | 2061.50 | 2026-03-09 09:15:00 | 1958.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 13:15:00 | 2059.00 | 2026-03-10 13:15:00 | 2000.10 | STOP_HIT | 0.50 | 2.86% |
| SELL | retest2 | 2026-03-06 13:45:00 | 2061.50 | 2026-03-10 13:15:00 | 2000.10 | STOP_HIT | 0.50 | 2.98% |
| SELL | retest2 | 2026-03-24 10:15:00 | 1848.30 | 2026-03-25 10:15:00 | 1907.00 | STOP_HIT | 1.00 | -3.18% |
| BUY | retest2 | 2026-04-13 10:15:00 | 2144.90 | 2026-04-28 09:15:00 | 2359.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 10:45:00 | 2132.10 | 2026-04-28 09:15:00 | 2345.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 14:45:00 | 2130.00 | 2026-04-28 09:15:00 | 2343.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 15:15:00 | 2130.20 | 2026-04-28 09:15:00 | 2343.22 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-23 10:15:00 | 2261.90 | 2026-05-04 09:15:00 | 2488.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-24 14:00:00 | 2261.70 | 2026-05-04 09:15:00 | 2487.87 | TARGET_HIT | 1.00 | 10.00% |
