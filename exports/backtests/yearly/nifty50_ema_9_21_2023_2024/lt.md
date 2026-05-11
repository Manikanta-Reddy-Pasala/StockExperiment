# LT (LT)

## Backtest Summary

- **Window:** 2023-03-13 09:15:00 → 2026-05-08 15:15:00 (5443 bars)
- **Last close:** 3978.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 214 |
| ALERT1 | 142 |
| ALERT2 | 138 |
| ALERT2_SKIP | 64 |
| ALERT3 | 375 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 140 |
| PARTIAL | 8 |
| TARGET_HIT | 7 |
| STOP_HIT | 139 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 153 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 67 / 86
- **Target hits / Stop hits / Partials:** 7 / 138 / 8
- **Avg / median % per leg:** 0.98% / -0.32%
- **Sum % (uncompounded):** 149.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 85 | 41 | 48.2% | 6 | 79 | 0 | 1.29% | 109.7% |
| BUY @ 2nd Alert (retest1) | 4 | 3 | 75.0% | 0 | 4 | 0 | 0.27% | 1.1% |
| BUY @ 3rd Alert (retest2) | 81 | 38 | 46.9% | 6 | 75 | 0 | 1.34% | 108.6% |
| SELL (all) | 68 | 26 | 38.2% | 1 | 59 | 8 | 0.58% | 39.5% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.70% | 0.7% |
| SELL @ 3rd Alert (retest2) | 67 | 25 | 37.3% | 1 | 58 | 8 | 0.58% | 38.8% |
| retest1 (combined) | 5 | 4 | 80.0% | 0 | 5 | 0 | 0.35% | 1.8% |
| retest2 (combined) | 148 | 63 | 42.6% | 7 | 133 | 8 | 1.00% | 147.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 13:15:00 | 2218.85 | 2208.49 | 2207.76 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2023-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-23 12:15:00 | 2197.75 | 2206.47 | 2207.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-23 15:15:00 | 2195.00 | 2201.90 | 2204.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-24 09:15:00 | 2203.85 | 2202.29 | 2204.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-24 09:15:00 | 2203.85 | 2202.29 | 2204.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 09:15:00 | 2203.85 | 2202.29 | 2204.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-24 09:30:00 | 2203.50 | 2202.29 | 2204.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-24 10:15:00 | 2207.70 | 2203.37 | 2205.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-05-24 11:45:00 | 2196.60 | 2201.60 | 2204.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-26 09:15:00 | 2210.70 | 2198.87 | 2198.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2023-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-26 09:15:00 | 2210.70 | 2198.87 | 2198.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-26 11:15:00 | 2218.15 | 2205.43 | 2201.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-30 10:15:00 | 2218.75 | 2224.77 | 2218.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-30 10:15:00 | 2218.75 | 2224.77 | 2218.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 10:15:00 | 2218.75 | 2224.77 | 2218.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 11:00:00 | 2218.75 | 2224.77 | 2218.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 11:15:00 | 2215.75 | 2222.97 | 2218.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 12:00:00 | 2215.75 | 2222.97 | 2218.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-30 12:15:00 | 2212.90 | 2220.95 | 2217.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-30 12:30:00 | 2210.00 | 2220.95 | 2217.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2023-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 09:15:00 | 2212.20 | 2215.25 | 2215.57 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2023-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-31 10:15:00 | 2219.55 | 2216.11 | 2215.93 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2023-05-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 11:15:00 | 2205.10 | 2213.91 | 2214.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-31 12:15:00 | 2202.45 | 2211.62 | 2213.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-31 15:15:00 | 2209.55 | 2209.39 | 2212.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-31 15:15:00 | 2209.55 | 2209.39 | 2212.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-31 15:15:00 | 2209.55 | 2209.39 | 2212.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 09:15:00 | 2203.85 | 2209.39 | 2212.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 09:15:00 | 2216.55 | 2210.82 | 2212.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-01 09:45:00 | 2216.50 | 2210.82 | 2212.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-01 10:15:00 | 2213.25 | 2211.31 | 2212.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-01 12:00:00 | 2210.00 | 2211.04 | 2212.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-01 13:45:00 | 2206.60 | 2211.04 | 2212.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-01 15:00:00 | 2209.45 | 2210.73 | 2211.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-02 10:15:00 | 2218.95 | 2213.09 | 2212.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2023-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-02 10:15:00 | 2218.95 | 2213.09 | 2212.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-02 12:15:00 | 2229.85 | 2217.45 | 2214.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-12 09:15:00 | 2335.00 | 2351.50 | 2334.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 09:15:00 | 2335.00 | 2351.50 | 2334.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 2335.00 | 2351.50 | 2334.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-12 10:00:00 | 2335.00 | 2351.50 | 2334.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 10:15:00 | 2342.40 | 2349.68 | 2335.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-13 12:00:00 | 2346.60 | 2344.05 | 2339.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-14 09:45:00 | 2348.15 | 2348.96 | 2343.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-26 10:15:00 | 2379.05 | 2392.59 | 2394.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2023-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-26 10:15:00 | 2379.05 | 2392.59 | 2394.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-26 12:15:00 | 2373.75 | 2387.45 | 2391.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-27 09:15:00 | 2383.55 | 2382.00 | 2387.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 09:15:00 | 2383.55 | 2382.00 | 2387.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 2383.55 | 2382.00 | 2387.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:30:00 | 2382.15 | 2382.00 | 2387.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 2389.45 | 2383.49 | 2387.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 11:00:00 | 2389.45 | 2383.49 | 2387.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 11:15:00 | 2394.65 | 2385.72 | 2388.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 11:30:00 | 2392.80 | 2385.72 | 2388.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 13:15:00 | 2389.60 | 2387.43 | 2388.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-27 14:15:00 | 2386.60 | 2387.43 | 2388.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-28 09:15:00 | 2405.30 | 2391.44 | 2390.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2023-06-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 09:15:00 | 2405.30 | 2391.44 | 2390.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-28 10:15:00 | 2417.10 | 2396.58 | 2392.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-03 13:15:00 | 2457.00 | 2458.04 | 2440.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-03 14:00:00 | 2457.00 | 2458.04 | 2440.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-05 09:15:00 | 2478.65 | 2466.37 | 2456.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-06 10:15:00 | 2493.30 | 2471.43 | 2463.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-07 12:15:00 | 2453.60 | 2474.22 | 2473.27 | SL hit (close<static) qty=1.00 sl=2454.35 alert=retest2 |

### Cycle 10 — SELL (started 2023-07-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-07 13:15:00 | 2454.05 | 2470.19 | 2471.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-07 14:15:00 | 2447.55 | 2465.66 | 2469.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-10 15:15:00 | 2446.10 | 2443.17 | 2453.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-11 09:15:00 | 2467.05 | 2443.17 | 2453.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 09:15:00 | 2470.70 | 2448.67 | 2454.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 10:00:00 | 2470.70 | 2448.67 | 2454.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-11 10:15:00 | 2473.05 | 2453.55 | 2456.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-11 10:30:00 | 2475.45 | 2453.55 | 2456.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2023-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 12:15:00 | 2474.05 | 2458.32 | 2458.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-11 15:15:00 | 2478.35 | 2466.75 | 2462.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-12 10:15:00 | 2467.00 | 2467.54 | 2463.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-12 11:15:00 | 2469.75 | 2467.54 | 2463.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 11:15:00 | 2459.15 | 2465.86 | 2463.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-12 12:00:00 | 2459.15 | 2465.86 | 2463.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 12:15:00 | 2462.35 | 2465.16 | 2463.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-12 14:00:00 | 2474.90 | 2467.11 | 2464.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-12 14:15:00 | 2453.75 | 2464.44 | 2463.22 | SL hit (close<static) qty=1.00 sl=2455.00 alert=retest2 |

### Cycle 12 — SELL (started 2023-07-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 14:15:00 | 2448.95 | 2462.55 | 2463.56 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2023-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-14 12:15:00 | 2469.30 | 2463.53 | 2463.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-14 14:15:00 | 2474.25 | 2466.55 | 2464.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-17 14:15:00 | 2469.95 | 2474.12 | 2470.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-17 14:15:00 | 2469.95 | 2474.12 | 2470.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 14:15:00 | 2469.95 | 2474.12 | 2470.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-17 15:00:00 | 2469.95 | 2474.12 | 2470.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 15:15:00 | 2474.95 | 2474.29 | 2471.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 09:15:00 | 2478.90 | 2474.29 | 2471.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 10:15:00 | 2477.05 | 2473.83 | 2471.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-18 15:15:00 | 2479.00 | 2474.75 | 2472.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-20 10:15:00 | 2481.00 | 2486.14 | 2481.53 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 10:15:00 | 2474.60 | 2483.83 | 2480.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-20 11:00:00 | 2474.60 | 2483.83 | 2480.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 11:15:00 | 2485.25 | 2484.11 | 2481.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-20 12:15:00 | 2488.95 | 2484.11 | 2481.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-20 14:15:00 | 2489.00 | 2484.44 | 2481.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-20 15:00:00 | 2491.05 | 2485.76 | 2482.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-02 11:15:00 | 2630.85 | 2651.06 | 2652.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2023-08-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-02 11:15:00 | 2630.85 | 2651.06 | 2652.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-02 13:15:00 | 2606.00 | 2638.36 | 2646.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-03 10:15:00 | 2633.70 | 2632.06 | 2640.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-03 11:00:00 | 2633.70 | 2632.06 | 2640.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-03 11:15:00 | 2627.10 | 2631.06 | 2639.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-03 12:45:00 | 2618.60 | 2627.86 | 2636.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 12:30:00 | 2625.75 | 2621.71 | 2627.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-04 15:00:00 | 2624.40 | 2622.39 | 2626.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-07 09:15:00 | 2651.70 | 2629.79 | 2629.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2023-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-07 09:15:00 | 2651.70 | 2629.79 | 2629.44 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2023-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 10:15:00 | 2629.50 | 2633.88 | 2634.16 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2023-08-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-09 14:15:00 | 2653.00 | 2636.92 | 2635.33 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2023-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 09:15:00 | 2625.10 | 2638.51 | 2639.04 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2023-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-14 11:15:00 | 2653.00 | 2640.87 | 2639.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-14 13:15:00 | 2656.50 | 2644.69 | 2641.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-17 09:15:00 | 2673.95 | 2677.24 | 2664.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-17 09:30:00 | 2675.90 | 2677.24 | 2664.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 11:15:00 | 2650.35 | 2669.91 | 2663.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 12:00:00 | 2650.35 | 2669.91 | 2663.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 12:15:00 | 2644.90 | 2664.90 | 2661.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-17 13:00:00 | 2644.90 | 2664.90 | 2661.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2023-08-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 14:15:00 | 2650.05 | 2659.52 | 2659.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-17 15:15:00 | 2648.00 | 2657.22 | 2658.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-18 09:15:00 | 2668.55 | 2659.49 | 2659.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 09:15:00 | 2668.55 | 2659.49 | 2659.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 09:15:00 | 2668.55 | 2659.49 | 2659.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-18 09:30:00 | 2671.00 | 2659.49 | 2659.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2023-08-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-18 10:15:00 | 2667.75 | 2661.14 | 2660.26 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2023-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-18 14:15:00 | 2639.95 | 2658.10 | 2659.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 15:15:00 | 2634.75 | 2653.43 | 2657.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-21 13:15:00 | 2645.65 | 2644.49 | 2650.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-21 13:45:00 | 2649.70 | 2644.49 | 2650.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 14:15:00 | 2664.00 | 2648.39 | 2651.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-21 15:00:00 | 2664.00 | 2648.39 | 2651.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-21 15:15:00 | 2656.15 | 2649.94 | 2651.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-22 09:15:00 | 2678.50 | 2649.94 | 2651.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2023-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-22 09:15:00 | 2666.95 | 2653.34 | 2653.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 11:15:00 | 2683.55 | 2662.67 | 2657.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-24 11:15:00 | 2706.20 | 2716.55 | 2699.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-24 12:00:00 | 2706.20 | 2716.55 | 2699.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 12:15:00 | 2695.00 | 2712.24 | 2698.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 13:00:00 | 2695.00 | 2712.24 | 2698.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 13:15:00 | 2689.50 | 2707.69 | 2697.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 13:30:00 | 2689.15 | 2707.69 | 2697.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2023-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-25 09:15:00 | 2652.75 | 2690.80 | 2692.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 10:15:00 | 2646.00 | 2681.84 | 2687.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 09:15:00 | 2672.25 | 2656.64 | 2669.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 09:15:00 | 2672.25 | 2656.64 | 2669.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 09:15:00 | 2672.25 | 2656.64 | 2669.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 10:15:00 | 2676.45 | 2656.64 | 2669.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 10:15:00 | 2678.35 | 2660.98 | 2670.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 10:45:00 | 2677.85 | 2660.98 | 2670.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 11:15:00 | 2673.30 | 2663.44 | 2670.73 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2023-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-28 15:15:00 | 2692.10 | 2677.50 | 2675.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-29 09:15:00 | 2701.45 | 2682.29 | 2677.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-30 13:15:00 | 2720.00 | 2721.04 | 2706.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-30 14:00:00 | 2720.00 | 2721.04 | 2706.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 14:15:00 | 2708.75 | 2718.58 | 2707.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-30 15:00:00 | 2708.75 | 2718.58 | 2707.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 15:15:00 | 2710.80 | 2717.02 | 2707.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 09:15:00 | 2703.00 | 2717.02 | 2707.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 2700.90 | 2713.80 | 2706.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 09:45:00 | 2705.90 | 2713.80 | 2706.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 10:15:00 | 2686.10 | 2708.26 | 2704.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-31 11:00:00 | 2686.10 | 2708.26 | 2704.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2023-08-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-31 12:15:00 | 2683.90 | 2700.05 | 2701.57 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2023-08-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-31 15:15:00 | 2716.00 | 2704.11 | 2703.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-04 09:15:00 | 2725.45 | 2708.73 | 2705.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-04 13:15:00 | 2706.10 | 2709.69 | 2707.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-04 13:15:00 | 2706.10 | 2709.69 | 2707.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 13:15:00 | 2706.10 | 2709.69 | 2707.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 14:00:00 | 2706.10 | 2709.69 | 2707.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 14:15:00 | 2710.85 | 2709.92 | 2707.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-04 14:30:00 | 2707.25 | 2709.92 | 2707.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 15:15:00 | 2713.00 | 2710.54 | 2708.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-05 09:15:00 | 2731.80 | 2710.54 | 2708.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 12:45:00 | 2717.50 | 2725.16 | 2721.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 14:30:00 | 2722.10 | 2724.43 | 2721.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-06 15:00:00 | 2730.10 | 2724.43 | 2721.36 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Target hit | 2023-09-12 09:15:00 | 3004.98 | 2915.55 | 2878.30 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2023-09-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 11:15:00 | 2909.75 | 2916.44 | 2916.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-20 12:15:00 | 2900.00 | 2913.15 | 2915.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-21 14:15:00 | 2897.50 | 2888.03 | 2896.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-21 14:15:00 | 2897.50 | 2888.03 | 2896.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 14:15:00 | 2897.50 | 2888.03 | 2896.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-21 15:00:00 | 2897.50 | 2888.03 | 2896.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-21 15:15:00 | 2901.60 | 2890.74 | 2896.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 09:15:00 | 2922.65 | 2890.74 | 2896.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-22 09:15:00 | 2913.90 | 2895.37 | 2898.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-22 09:45:00 | 2925.00 | 2895.37 | 2898.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2023-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-22 10:15:00 | 2930.95 | 2902.49 | 2901.38 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2023-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-25 10:15:00 | 2880.10 | 2902.50 | 2903.72 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2023-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-26 09:15:00 | 2930.00 | 2905.62 | 2903.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-27 12:15:00 | 2974.75 | 2931.16 | 2919.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-29 14:15:00 | 3014.00 | 3021.24 | 2998.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-29 15:00:00 | 3014.00 | 3021.24 | 2998.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 3010.05 | 3019.63 | 3001.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-03 09:45:00 | 3000.00 | 3019.63 | 3001.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-04 10:15:00 | 3035.00 | 3041.55 | 3025.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 09:15:00 | 3063.00 | 3027.31 | 3022.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 10:00:00 | 3047.55 | 3031.36 | 3025.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-05 10:45:00 | 3054.00 | 3036.31 | 3027.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-12 13:15:00 | 3078.35 | 3083.04 | 3083.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2023-10-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 13:15:00 | 3078.35 | 3083.04 | 3083.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-13 09:15:00 | 3070.25 | 3080.43 | 3081.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-13 12:15:00 | 3097.00 | 3080.21 | 3081.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-13 12:15:00 | 3097.00 | 3080.21 | 3081.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 12:15:00 | 3097.00 | 3080.21 | 3081.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-13 13:00:00 | 3097.00 | 3080.21 | 3081.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 13:15:00 | 3086.15 | 3081.39 | 3081.61 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2023-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-13 14:15:00 | 3091.05 | 3083.33 | 3082.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-16 14:15:00 | 3102.65 | 3091.00 | 3087.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-17 09:15:00 | 3086.05 | 3091.16 | 3088.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-17 09:15:00 | 3086.05 | 3091.16 | 3088.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 09:15:00 | 3086.05 | 3091.16 | 3088.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-17 10:15:00 | 3078.25 | 3091.16 | 3088.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 10:15:00 | 3087.35 | 3090.40 | 3087.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-17 10:30:00 | 3080.70 | 3090.40 | 3087.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-17 11:15:00 | 3075.00 | 3087.32 | 3086.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-17 12:00:00 | 3075.00 | 3087.32 | 3086.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2023-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-17 12:15:00 | 3069.60 | 3083.78 | 3085.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-17 13:15:00 | 3065.80 | 3080.18 | 3083.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 11:15:00 | 3054.10 | 3046.57 | 3058.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 11:15:00 | 3054.10 | 3046.57 | 3058.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 11:15:00 | 3054.10 | 3046.57 | 3058.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 11:45:00 | 3049.15 | 3046.57 | 3058.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 12:15:00 | 3055.50 | 3048.36 | 3057.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 12:45:00 | 3057.55 | 3048.36 | 3057.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 13:15:00 | 3065.05 | 3051.70 | 3058.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 13:30:00 | 3059.40 | 3051.70 | 3058.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 14:15:00 | 3055.00 | 3052.36 | 3058.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 09:15:00 | 3040.45 | 3052.86 | 3057.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-10-26 09:15:00 | 2888.43 | 2928.15 | 2961.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-10-27 09:15:00 | 2884.75 | 2881.39 | 2916.24 | SL hit (close>ema200) qty=0.50 sl=2881.39 alert=retest2 |

### Cycle 35 — BUY (started 2023-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-30 15:15:00 | 2929.00 | 2913.43 | 2912.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-31 14:15:00 | 2936.80 | 2919.79 | 2915.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-01 09:15:00 | 2887.40 | 2915.74 | 2914.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-01 09:15:00 | 2887.40 | 2915.74 | 2914.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 09:15:00 | 2887.40 | 2915.74 | 2914.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-01 10:00:00 | 2887.40 | 2915.74 | 2914.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2023-11-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-01 10:15:00 | 2893.55 | 2911.31 | 2912.87 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2023-11-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 13:15:00 | 2912.60 | 2910.39 | 2910.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 15:15:00 | 2921.00 | 2913.54 | 2911.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-03 11:15:00 | 2911.25 | 2917.98 | 2914.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-03 11:15:00 | 2911.25 | 2917.98 | 2914.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 11:15:00 | 2911.25 | 2917.98 | 2914.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-03 12:00:00 | 2911.25 | 2917.98 | 2914.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-03 12:15:00 | 2918.10 | 2918.00 | 2915.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-06 09:15:00 | 2942.95 | 2916.20 | 2914.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-23 09:15:00 | 3070.55 | 3077.66 | 3077.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2023-11-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-23 09:15:00 | 3070.55 | 3077.66 | 3077.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-23 10:15:00 | 3058.95 | 3073.92 | 3076.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-24 11:15:00 | 3066.05 | 3057.14 | 3063.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-24 11:15:00 | 3066.05 | 3057.14 | 3063.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 11:15:00 | 3066.05 | 3057.14 | 3063.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-24 12:00:00 | 3066.05 | 3057.14 | 3063.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-24 12:15:00 | 3056.75 | 3057.06 | 3062.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-28 10:00:00 | 3043.60 | 3052.92 | 3058.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-28 14:30:00 | 3049.90 | 3048.46 | 3054.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-29 12:15:00 | 3061.40 | 3057.30 | 3056.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2023-11-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-29 12:15:00 | 3061.40 | 3057.30 | 3056.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-29 13:15:00 | 3085.50 | 3062.94 | 3059.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 12:15:00 | 3291.10 | 3291.57 | 3244.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-05 12:30:00 | 3292.25 | 3291.57 | 3244.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-11 09:15:00 | 3386.70 | 3384.95 | 3366.34 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2023-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 11:15:00 | 3353.15 | 3366.39 | 3366.64 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2023-12-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-13 13:15:00 | 3392.90 | 3364.49 | 3362.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-13 14:15:00 | 3397.60 | 3371.11 | 3365.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-20 11:15:00 | 3499.15 | 3499.76 | 3483.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-20 11:45:00 | 3504.20 | 3499.76 | 3483.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 3461.70 | 3490.91 | 3482.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 14:00:00 | 3461.70 | 3490.91 | 3482.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 3416.20 | 3475.97 | 3476.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 15:15:00 | 3413.00 | 3463.37 | 3470.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 14:15:00 | 3424.85 | 3421.48 | 3440.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-21 15:15:00 | 3438.50 | 3421.48 | 3440.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 09:15:00 | 3452.30 | 3430.37 | 3441.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 10:00:00 | 3452.30 | 3430.37 | 3441.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-22 10:15:00 | 3471.80 | 3438.65 | 3444.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-22 11:00:00 | 3471.80 | 3438.65 | 3444.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2023-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 12:15:00 | 3461.00 | 3449.58 | 3448.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 14:15:00 | 3485.00 | 3459.19 | 3453.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-28 14:15:00 | 3515.05 | 3528.14 | 3513.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-28 15:00:00 | 3515.05 | 3528.14 | 3513.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-28 15:15:00 | 3518.40 | 3526.19 | 3514.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-29 09:15:00 | 3527.05 | 3526.19 | 3514.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-29 10:15:00 | 3497.95 | 3517.49 | 3512.25 | SL hit (close<static) qty=1.00 sl=3505.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-02 09:15:00 | 3483.55 | 3508.62 | 3511.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-02 10:15:00 | 3446.65 | 3496.23 | 3505.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-03 13:15:00 | 3454.35 | 3448.52 | 3466.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-01-03 13:30:00 | 3453.55 | 3448.52 | 3466.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 09:15:00 | 3448.95 | 3446.17 | 3461.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-04 09:30:00 | 3463.20 | 3446.17 | 3461.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 12:15:00 | 3465.00 | 3452.28 | 3460.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-04 13:00:00 | 3465.00 | 3452.28 | 3460.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 13:15:00 | 3467.75 | 3455.38 | 3461.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-04 13:30:00 | 3478.00 | 3455.38 | 3461.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-04 15:15:00 | 3459.95 | 3456.49 | 3460.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-05 09:15:00 | 3479.05 | 3456.49 | 3460.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 09:15:00 | 3472.85 | 3459.77 | 3461.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-05 09:30:00 | 3471.20 | 3459.77 | 3461.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2024-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-05 10:15:00 | 3476.10 | 3463.03 | 3462.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-05 11:15:00 | 3486.75 | 3467.78 | 3465.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-08 10:15:00 | 3501.45 | 3504.86 | 3487.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-08 11:00:00 | 3501.45 | 3504.86 | 3487.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 14:15:00 | 3491.30 | 3509.23 | 3495.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-08 15:00:00 | 3491.30 | 3509.23 | 3495.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-08 15:15:00 | 3510.00 | 3509.38 | 3497.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-09 09:15:00 | 3544.00 | 3509.38 | 3497.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-11 13:15:00 | 3509.50 | 3528.17 | 3529.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2024-01-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-11 13:15:00 | 3509.50 | 3528.17 | 3529.98 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-01-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-12 14:15:00 | 3566.25 | 3532.79 | 3529.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-15 09:15:00 | 3580.00 | 3547.86 | 3536.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-15 12:15:00 | 3547.70 | 3554.36 | 3543.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-15 12:45:00 | 3548.10 | 3554.36 | 3543.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 13:15:00 | 3550.45 | 3553.58 | 3543.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 13:45:00 | 3547.90 | 3553.58 | 3543.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 14:15:00 | 3542.95 | 3551.45 | 3543.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-15 15:00:00 | 3542.95 | 3551.45 | 3543.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-15 15:15:00 | 3546.90 | 3550.54 | 3544.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-16 09:15:00 | 3553.00 | 3550.54 | 3544.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-16 09:15:00 | 3556.30 | 3551.69 | 3545.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-16 10:45:00 | 3585.15 | 3557.20 | 3548.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-16 11:45:00 | 3567.20 | 3558.03 | 3549.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-16 12:15:00 | 3571.00 | 3558.03 | 3549.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-16 13:15:00 | 3569.65 | 3558.63 | 3550.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-17 09:15:00 | 3610.00 | 3575.24 | 3561.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-19 09:15:00 | 3628.50 | 3593.43 | 3581.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-23 11:15:00 | 3618.40 | 3633.27 | 3625.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-23 12:15:00 | 3589.50 | 3619.06 | 3619.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2024-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 12:15:00 | 3589.50 | 3619.06 | 3619.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 14:15:00 | 3547.35 | 3600.33 | 3610.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-24 10:15:00 | 3616.55 | 3593.02 | 3603.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-24 10:15:00 | 3616.55 | 3593.02 | 3603.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 10:15:00 | 3616.55 | 3593.02 | 3603.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-24 11:00:00 | 3616.55 | 3593.02 | 3603.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 11:15:00 | 3577.90 | 3590.00 | 3601.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-24 12:15:00 | 3572.90 | 3590.00 | 3601.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-24 13:30:00 | 3572.55 | 3588.84 | 3598.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-25 14:30:00 | 3569.30 | 3594.66 | 3597.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-29 09:15:00 | 3669.70 | 3607.96 | 3602.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2024-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-29 09:15:00 | 3669.70 | 3607.96 | 3602.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-29 10:15:00 | 3695.90 | 3625.55 | 3611.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-30 10:15:00 | 3676.45 | 3684.06 | 3654.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-30 11:00:00 | 3676.45 | 3684.06 | 3654.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 14:15:00 | 3634.00 | 3672.66 | 3658.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-30 15:00:00 | 3634.00 | 3672.66 | 3658.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-30 15:15:00 | 3639.85 | 3666.09 | 3656.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-31 09:15:00 | 3445.65 | 3666.09 | 3656.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — SELL (started 2024-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-31 09:15:00 | 3414.00 | 3615.68 | 3634.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-02 13:15:00 | 3386.30 | 3409.02 | 3457.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-06 09:15:00 | 3370.50 | 3365.83 | 3399.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-06 10:00:00 | 3370.50 | 3365.83 | 3399.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 12:15:00 | 3400.25 | 3381.15 | 3398.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 13:15:00 | 3403.25 | 3381.15 | 3398.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 13:15:00 | 3410.10 | 3386.94 | 3399.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 14:00:00 | 3410.10 | 3386.94 | 3399.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-06 14:15:00 | 3424.00 | 3394.35 | 3401.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-06 15:00:00 | 3424.00 | 3394.35 | 3401.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2024-02-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-07 10:15:00 | 3430.45 | 3410.58 | 3408.15 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2024-02-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-07 14:15:00 | 3391.20 | 3404.37 | 3405.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-08 09:15:00 | 3382.50 | 3398.66 | 3402.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-13 09:15:00 | 3320.90 | 3311.33 | 3327.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-13 10:00:00 | 3320.90 | 3311.33 | 3327.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 10:15:00 | 3329.40 | 3314.94 | 3328.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-13 10:45:00 | 3326.05 | 3314.94 | 3328.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-13 11:15:00 | 3319.10 | 3315.77 | 3327.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-13 15:15:00 | 3316.25 | 3319.32 | 3326.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-16 09:15:00 | 3353.50 | 3312.78 | 3311.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2024-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 09:15:00 | 3353.50 | 3312.78 | 3311.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-16 11:15:00 | 3366.00 | 3330.82 | 3320.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-19 09:15:00 | 3352.05 | 3359.63 | 3341.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-19 09:15:00 | 3352.05 | 3359.63 | 3341.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 09:15:00 | 3352.05 | 3359.63 | 3341.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 10:00:00 | 3352.05 | 3359.63 | 3341.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 14:15:00 | 3338.90 | 3351.20 | 3343.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-19 15:00:00 | 3338.90 | 3351.20 | 3343.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-19 15:15:00 | 3341.15 | 3349.19 | 3343.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 09:15:00 | 3350.00 | 3349.19 | 3343.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 10:15:00 | 3345.65 | 3347.38 | 3343.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-20 15:00:00 | 3358.50 | 3347.18 | 3344.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-21 09:15:00 | 3339.25 | 3348.04 | 3345.43 | SL hit (close<static) qty=1.00 sl=3341.85 alert=retest2 |

### Cycle 54 — SELL (started 2024-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 11:15:00 | 3327.00 | 3342.70 | 3343.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 13:15:00 | 3318.00 | 3334.73 | 3339.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-22 12:15:00 | 3313.90 | 3309.62 | 3322.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-22 13:00:00 | 3313.90 | 3309.62 | 3322.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 13:15:00 | 3331.70 | 3314.03 | 3322.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 14:00:00 | 3331.70 | 3314.03 | 3322.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 14:15:00 | 3366.65 | 3324.56 | 3326.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-22 15:00:00 | 3366.65 | 3324.56 | 3326.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2024-02-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-22 15:15:00 | 3363.00 | 3332.25 | 3330.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-23 10:15:00 | 3372.00 | 3343.66 | 3335.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-28 10:15:00 | 3465.75 | 3485.07 | 3457.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-28 10:15:00 | 3465.75 | 3485.07 | 3457.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 10:15:00 | 3465.75 | 3485.07 | 3457.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 11:00:00 | 3465.75 | 3485.07 | 3457.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 11:15:00 | 3458.95 | 3479.85 | 3457.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 11:45:00 | 3453.85 | 3479.85 | 3457.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 12:15:00 | 3460.00 | 3475.88 | 3457.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-28 12:30:00 | 3467.65 | 3475.88 | 3457.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 13:15:00 | 3487.45 | 3478.19 | 3460.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-01 09:15:00 | 3529.00 | 3473.58 | 3466.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-12 14:15:00 | 3619.80 | 3638.80 | 3641.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2024-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-12 14:15:00 | 3619.80 | 3638.80 | 3641.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 09:15:00 | 3606.25 | 3629.60 | 3636.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 09:15:00 | 3575.75 | 3563.32 | 3590.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 09:15:00 | 3575.75 | 3563.32 | 3590.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 3575.75 | 3563.32 | 3590.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 10:00:00 | 3575.75 | 3563.32 | 3590.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 10:15:00 | 3598.80 | 3570.41 | 3591.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-15 09:30:00 | 3565.65 | 3587.63 | 3594.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-03-21 10:15:00 | 3575.85 | 3526.13 | 3528.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-21 10:15:00 | 3587.15 | 3538.33 | 3533.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2024-03-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-21 10:15:00 | 3587.15 | 3538.33 | 3533.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-22 09:15:00 | 3603.95 | 3567.14 | 3551.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-02 10:15:00 | 3811.00 | 3821.27 | 3782.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-02 11:00:00 | 3811.00 | 3821.27 | 3782.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 11:15:00 | 3783.70 | 3813.76 | 3782.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-02 11:45:00 | 3784.05 | 3813.76 | 3782.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 12:15:00 | 3783.95 | 3807.80 | 3782.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-02 12:30:00 | 3782.60 | 3807.80 | 3782.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 13:15:00 | 3780.80 | 3802.40 | 3782.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-02 13:30:00 | 3782.20 | 3802.40 | 3782.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-02 14:15:00 | 3810.60 | 3804.04 | 3785.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-02 14:30:00 | 3794.30 | 3804.04 | 3785.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 3782.50 | 3801.32 | 3787.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 09:30:00 | 3775.00 | 3801.32 | 3787.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 10:15:00 | 3778.60 | 3796.78 | 3786.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 10:45:00 | 3777.70 | 3796.78 | 3786.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 11:15:00 | 3778.00 | 3793.02 | 3785.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 11:45:00 | 3777.00 | 3793.02 | 3785.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 13:15:00 | 3782.00 | 3790.09 | 3785.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 14:00:00 | 3782.00 | 3790.09 | 3785.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 14:15:00 | 3782.70 | 3788.61 | 3785.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-03 14:30:00 | 3777.70 | 3788.61 | 3785.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 15:15:00 | 3782.90 | 3787.47 | 3785.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-04 09:15:00 | 3790.25 | 3787.47 | 3785.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-04 09:15:00 | 3745.40 | 3779.05 | 3781.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2024-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-04 09:15:00 | 3745.40 | 3779.05 | 3781.52 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2024-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-04 13:15:00 | 3806.60 | 3781.02 | 3780.82 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2024-04-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-05 09:15:00 | 3725.60 | 3776.80 | 3779.53 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2024-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-08 12:15:00 | 3796.70 | 3774.22 | 3771.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 14:15:00 | 3808.40 | 3782.69 | 3776.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-09 13:15:00 | 3790.25 | 3795.80 | 3787.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-09 13:15:00 | 3790.25 | 3795.80 | 3787.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 13:15:00 | 3790.25 | 3795.80 | 3787.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 14:00:00 | 3790.25 | 3795.80 | 3787.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 14:15:00 | 3781.35 | 3792.91 | 3786.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-09 14:30:00 | 3779.45 | 3792.91 | 3786.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-09 15:15:00 | 3781.00 | 3790.53 | 3786.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-10 09:15:00 | 3781.90 | 3790.53 | 3786.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 09:15:00 | 3773.60 | 3787.14 | 3784.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-10 10:00:00 | 3773.60 | 3787.14 | 3784.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2024-04-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-10 10:15:00 | 3763.35 | 3782.38 | 3782.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 12:15:00 | 3714.35 | 3753.41 | 3765.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-18 09:15:00 | 3575.60 | 3570.63 | 3615.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-18 09:30:00 | 3560.00 | 3570.63 | 3615.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 11:15:00 | 3610.00 | 3582.53 | 3613.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-18 11:30:00 | 3610.00 | 3582.53 | 3613.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 12:15:00 | 3591.05 | 3584.23 | 3611.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 13:15:00 | 3553.20 | 3584.23 | 3611.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 11:15:00 | 3584.10 | 3546.74 | 3560.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-22 13:15:00 | 3585.05 | 3562.08 | 3565.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-22 13:15:00 | 3593.05 | 3568.28 | 3568.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2024-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-22 13:15:00 | 3593.05 | 3568.28 | 3568.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-22 14:15:00 | 3614.05 | 3577.43 | 3572.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-23 12:15:00 | 3592.30 | 3593.28 | 3583.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-23 13:00:00 | 3592.30 | 3593.28 | 3583.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 3660.85 | 3637.00 | 3617.54 | EMA400 retest candle locked (from upside) |

### Cycle 64 — SELL (started 2024-04-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-26 14:15:00 | 3607.50 | 3622.63 | 3624.41 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2024-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-29 10:15:00 | 3648.00 | 3627.47 | 3626.03 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2024-04-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-30 14:15:00 | 3587.65 | 3625.04 | 3627.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-02 09:15:00 | 3582.35 | 3611.20 | 3620.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-02 10:15:00 | 3611.55 | 3611.27 | 3619.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-02 10:15:00 | 3611.55 | 3611.27 | 3619.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 10:15:00 | 3611.55 | 3611.27 | 3619.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 10:45:00 | 3618.10 | 3611.27 | 3619.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 11:15:00 | 3619.70 | 3612.96 | 3619.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 11:30:00 | 3620.15 | 3612.96 | 3619.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 12:15:00 | 3613.95 | 3613.16 | 3619.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 13:15:00 | 3616.80 | 3613.16 | 3619.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-02 13:15:00 | 3597.05 | 3609.94 | 3617.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-02 13:30:00 | 3611.45 | 3609.94 | 3617.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-03 09:15:00 | 3560.35 | 3596.61 | 3609.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-03 10:30:00 | 3549.85 | 3586.05 | 3603.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-08 09:15:00 | 3372.36 | 3442.68 | 3477.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-08 11:15:00 | 3459.40 | 3444.64 | 3472.29 | SL hit (close>ema200) qty=0.50 sl=3444.64 alert=retest2 |

### Cycle 67 — BUY (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 13:15:00 | 3368.05 | 3325.94 | 3320.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 3379.45 | 3336.64 | 3325.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 13:15:00 | 3395.60 | 3396.36 | 3376.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 13:30:00 | 3392.00 | 3396.36 | 3376.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 3441.05 | 3450.30 | 3438.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 15:00:00 | 3441.05 | 3450.30 | 3438.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 3435.05 | 3447.25 | 3437.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:15:00 | 3452.65 | 3447.25 | 3437.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:45:00 | 3456.80 | 3450.40 | 3440.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-03 09:15:00 | 3797.92 | 3720.90 | 3682.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 3419.75 | 3701.94 | 3722.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 3233.00 | 3483.80 | 3597.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 14:15:00 | 3417.85 | 3401.55 | 3504.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 15:00:00 | 3417.85 | 3401.55 | 3504.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 3516.90 | 3424.36 | 3497.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 10:00:00 | 3516.90 | 3424.36 | 3497.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 10:15:00 | 3528.90 | 3445.27 | 3499.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 12:00:00 | 3476.70 | 3451.56 | 3497.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-06 13:45:00 | 3485.30 | 3462.52 | 3495.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-07 13:15:00 | 3529.30 | 3503.66 | 3502.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — BUY (started 2024-06-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-07 13:15:00 | 3529.30 | 3503.66 | 3502.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 14:15:00 | 3536.70 | 3510.27 | 3505.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 14:15:00 | 3542.80 | 3543.08 | 3528.02 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 09:15:00 | 3585.05 | 3542.86 | 3529.29 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 3629.95 | 3678.19 | 3674.22 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 3629.95 | 3678.19 | 3674.22 | SL hit (close<ema400) qty=1.00 sl=3674.22 alert=retest1 |

### Cycle 70 — SELL (started 2024-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 10:15:00 | 3627.05 | 3667.96 | 3669.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 12:15:00 | 3616.50 | 3653.60 | 3662.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-20 14:15:00 | 3597.90 | 3596.76 | 3619.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-20 14:30:00 | 3608.80 | 3596.76 | 3619.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 3562.90 | 3547.28 | 3559.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 12:00:00 | 3562.90 | 3547.28 | 3559.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 12:15:00 | 3575.35 | 3552.90 | 3561.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 12:45:00 | 3576.80 | 3552.90 | 3561.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 13:15:00 | 3583.05 | 3558.93 | 3563.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 13:30:00 | 3581.60 | 3558.93 | 3563.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2024-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 15:15:00 | 3588.45 | 3569.48 | 3567.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 09:15:00 | 3602.40 | 3576.06 | 3570.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 09:15:00 | 3585.65 | 3594.30 | 3585.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 09:15:00 | 3585.65 | 3594.30 | 3585.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 3585.65 | 3594.30 | 3585.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:30:00 | 3581.15 | 3594.30 | 3585.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 3573.10 | 3590.06 | 3584.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:30:00 | 3568.30 | 3590.06 | 3584.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 11:15:00 | 3575.00 | 3587.05 | 3583.31 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 3558.60 | 3577.99 | 3579.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-28 14:15:00 | 3545.10 | 3561.81 | 3569.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-02 09:15:00 | 3540.75 | 3533.08 | 3545.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 09:15:00 | 3540.75 | 3533.08 | 3545.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 3540.75 | 3533.08 | 3545.33 | EMA400 retest candle locked (from downside) |

### Cycle 73 — BUY (started 2024-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 11:15:00 | 3617.30 | 3555.32 | 3553.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-09 09:15:00 | 3669.30 | 3633.46 | 3621.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 3645.00 | 3655.85 | 3642.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 09:15:00 | 3645.00 | 3655.85 | 3642.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 3645.00 | 3655.85 | 3642.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:00:00 | 3645.00 | 3655.85 | 3642.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 3628.50 | 3650.38 | 3640.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 3617.75 | 3650.38 | 3640.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 3649.00 | 3650.10 | 3641.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 09:15:00 | 3668.30 | 3646.64 | 3642.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-11 12:15:00 | 3622.05 | 3642.93 | 3642.42 | SL hit (close<static) qty=1.00 sl=3623.00 alert=retest2 |

### Cycle 74 — SELL (started 2024-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 13:15:00 | 3611.00 | 3636.55 | 3639.56 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2024-07-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 14:15:00 | 3643.40 | 3639.58 | 3639.08 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2024-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-16 10:15:00 | 3635.00 | 3642.34 | 3642.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 10:15:00 | 3607.05 | 3631.10 | 3636.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 11:15:00 | 3635.65 | 3632.01 | 3636.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 11:15:00 | 3635.65 | 3632.01 | 3636.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 3635.65 | 3632.01 | 3636.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 12:00:00 | 3635.65 | 3632.01 | 3636.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2024-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-18 12:15:00 | 3674.00 | 3640.41 | 3639.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-18 13:15:00 | 3674.80 | 3647.29 | 3643.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 15:15:00 | 3647.00 | 3648.62 | 3644.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-19 09:15:00 | 3649.70 | 3648.62 | 3644.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 3623.50 | 3643.59 | 3642.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 3623.50 | 3643.59 | 3642.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 3632.40 | 3641.35 | 3641.69 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2024-07-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-19 11:15:00 | 3653.30 | 3643.74 | 3642.75 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2024-07-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 12:15:00 | 3635.35 | 3642.06 | 3642.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 13:15:00 | 3626.95 | 3639.04 | 3640.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 10:15:00 | 3631.45 | 3630.27 | 3635.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 10:15:00 | 3631.45 | 3630.27 | 3635.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 3631.45 | 3630.27 | 3635.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 3640.55 | 3630.27 | 3635.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 3633.15 | 3630.85 | 3635.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:30:00 | 3639.65 | 3630.85 | 3635.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 3627.20 | 3630.12 | 3634.32 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2024-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 15:15:00 | 3651.00 | 3638.50 | 3637.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 09:15:00 | 3692.00 | 3649.20 | 3642.38 | Break + close above crossover candle high |

### Cycle 82 — SELL (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 11:15:00 | 3533.00 | 3635.83 | 3638.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 12:15:00 | 3521.30 | 3612.93 | 3627.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-25 09:15:00 | 3578.50 | 3540.03 | 3564.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 09:15:00 | 3578.50 | 3540.03 | 3564.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 3578.50 | 3540.03 | 3564.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 09:30:00 | 3586.65 | 3540.03 | 3564.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 10:15:00 | 3602.95 | 3552.62 | 3567.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 10:45:00 | 3595.35 | 3552.62 | 3567.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2024-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 13:15:00 | 3624.45 | 3583.71 | 3579.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 10:15:00 | 3636.75 | 3608.94 | 3594.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 09:15:00 | 3773.00 | 3775.98 | 3742.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 09:45:00 | 3779.00 | 3775.98 | 3742.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 3780.40 | 3799.38 | 3779.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:45:00 | 3782.20 | 3799.38 | 3779.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 13:15:00 | 3781.10 | 3795.72 | 3779.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 13:45:00 | 3779.45 | 3795.72 | 3779.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 3781.90 | 3792.96 | 3779.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:30:00 | 3783.05 | 3792.96 | 3779.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 15:15:00 | 3770.00 | 3788.37 | 3779.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:15:00 | 3706.55 | 3788.37 | 3779.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 3698.10 | 3770.31 | 3771.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 12:15:00 | 3678.25 | 3731.76 | 3751.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 3599.70 | 3578.15 | 3635.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 09:15:00 | 3616.20 | 3593.04 | 3616.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 3616.20 | 3593.04 | 3616.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:15:00 | 3644.50 | 3593.04 | 3616.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 3634.65 | 3601.37 | 3618.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:30:00 | 3647.55 | 3601.37 | 3618.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 11:15:00 | 3636.95 | 3608.48 | 3620.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 11:45:00 | 3632.75 | 3608.48 | 3620.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 3645.80 | 3623.68 | 3624.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 15:00:00 | 3645.80 | 3623.68 | 3624.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2024-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 15:15:00 | 3643.40 | 3627.62 | 3626.62 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 09:15:00 | 3581.75 | 3618.45 | 3622.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 12:15:00 | 3564.45 | 3600.74 | 3612.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 09:15:00 | 3593.65 | 3582.71 | 3598.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 09:15:00 | 3593.65 | 3582.71 | 3598.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 3593.65 | 3582.71 | 3598.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 12:00:00 | 3581.00 | 3583.15 | 3596.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 3577.30 | 3588.42 | 3594.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 11:45:00 | 3582.75 | 3585.44 | 3591.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 12:45:00 | 3581.50 | 3585.01 | 3590.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 14:15:00 | 3570.70 | 3581.83 | 3588.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-12 14:45:00 | 3578.80 | 3581.83 | 3588.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 3591.65 | 3582.38 | 3587.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 09:30:00 | 3595.55 | 3582.38 | 3587.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 3589.50 | 3583.80 | 3587.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:45:00 | 3590.90 | 3583.80 | 3587.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 11:15:00 | 3587.95 | 3584.63 | 3587.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 12:30:00 | 3580.90 | 3582.70 | 3586.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-20 11:15:00 | 3570.00 | 3563.79 | 3563.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2024-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 11:15:00 | 3570.00 | 3563.79 | 3563.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 12:15:00 | 3571.60 | 3565.35 | 3563.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 09:15:00 | 3588.50 | 3591.93 | 3583.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 10:00:00 | 3588.50 | 3591.93 | 3583.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 3617.50 | 3604.27 | 3594.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 11:30:00 | 3621.15 | 3608.98 | 3601.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-02 14:15:00 | 3683.10 | 3687.23 | 3687.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2024-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 14:15:00 | 3683.10 | 3687.23 | 3687.55 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2024-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-03 12:15:00 | 3704.20 | 3687.04 | 3686.72 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2024-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 09:15:00 | 3654.85 | 3682.17 | 3684.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-05 10:15:00 | 3621.65 | 3646.46 | 3661.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 10:15:00 | 3581.60 | 3580.01 | 3603.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-09 10:45:00 | 3579.90 | 3580.01 | 3603.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 3574.40 | 3577.92 | 3592.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 13:45:00 | 3567.65 | 3586.28 | 3591.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 09:45:00 | 3546.45 | 3565.73 | 3579.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 14:15:00 | 3631.70 | 3575.25 | 3577.88 | SL hit (close>static) qty=1.00 sl=3596.70 alert=retest2 |

### Cycle 91 — BUY (started 2024-09-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 15:15:00 | 3625.00 | 3585.20 | 3582.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 09:15:00 | 3639.75 | 3611.43 | 3599.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 09:15:00 | 3715.70 | 3716.26 | 3691.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-19 10:00:00 | 3715.70 | 3716.26 | 3691.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 11:15:00 | 3693.60 | 3709.56 | 3692.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 11:30:00 | 3680.75 | 3709.56 | 3692.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 3674.15 | 3702.48 | 3690.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:00:00 | 3674.15 | 3702.48 | 3690.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 3678.55 | 3697.69 | 3689.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 15:15:00 | 3689.00 | 3694.50 | 3688.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 12:15:00 | 3747.95 | 3779.93 | 3781.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2024-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 12:15:00 | 3747.95 | 3779.93 | 3781.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 09:15:00 | 3700.95 | 3755.50 | 3768.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 09:15:00 | 3713.20 | 3690.89 | 3709.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 09:15:00 | 3713.20 | 3690.89 | 3709.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 3713.20 | 3690.89 | 3709.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:00:00 | 3713.20 | 3690.89 | 3709.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 3675.15 | 3687.74 | 3706.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 11:15:00 | 3675.00 | 3687.74 | 3706.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 12:00:00 | 3674.10 | 3685.01 | 3703.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 15:15:00 | 3491.25 | 3548.30 | 3607.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-03 15:15:00 | 3490.39 | 3548.30 | 3607.40 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-04 11:15:00 | 3553.45 | 3535.89 | 3585.80 | SL hit (close>ema200) qty=0.50 sl=3535.89 alert=retest2 |

### Cycle 93 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 3552.00 | 3520.45 | 3517.90 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2024-10-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-09 14:15:00 | 3483.10 | 3512.29 | 3515.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 13:15:00 | 3478.70 | 3501.55 | 3508.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 11:15:00 | 3488.80 | 3485.17 | 3496.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-11 11:45:00 | 3489.00 | 3485.17 | 3496.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 13:15:00 | 3491.85 | 3485.65 | 3494.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 14:00:00 | 3491.85 | 3485.65 | 3494.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 14:15:00 | 3479.90 | 3484.50 | 3493.09 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2024-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 09:15:00 | 3548.60 | 3498.50 | 3498.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 12:15:00 | 3568.50 | 3526.24 | 3512.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 09:15:00 | 3532.90 | 3539.19 | 3524.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-15 10:00:00 | 3532.90 | 3539.19 | 3524.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 3545.80 | 3540.51 | 3525.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-15 10:30:00 | 3545.85 | 3540.51 | 3525.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 3558.50 | 3546.01 | 3535.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 09:30:00 | 3558.00 | 3546.01 | 3535.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 10:15:00 | 3539.65 | 3544.74 | 3535.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:45:00 | 3542.90 | 3544.74 | 3535.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 3523.05 | 3540.40 | 3534.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:45:00 | 3524.90 | 3540.40 | 3534.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 12:15:00 | 3522.55 | 3536.83 | 3533.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 12:30:00 | 3522.20 | 3536.83 | 3533.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 3531.00 | 3533.02 | 3532.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-17 09:15:00 | 3559.00 | 3533.02 | 3532.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-22 10:30:00 | 3531.30 | 3571.14 | 3570.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-22 11:15:00 | 3540.05 | 3564.93 | 3567.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — SELL (started 2024-10-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 11:15:00 | 3540.05 | 3564.93 | 3567.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 14:15:00 | 3510.05 | 3544.94 | 3557.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 15:15:00 | 3449.00 | 3448.33 | 3476.84 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-10-25 09:15:00 | 3406.00 | 3448.33 | 3476.84 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 3372.00 | 3351.08 | 3374.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 10:15:00 | 3371.15 | 3351.08 | 3374.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 10:15:00 | 3356.00 | 3352.07 | 3372.74 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-29 14:15:00 | 3382.00 | 3363.25 | 3371.84 | SL hit (close>ema400) qty=1.00 sl=3371.84 alert=retest1 |

### Cycle 97 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 3407.55 | 3382.07 | 3379.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 11:15:00 | 3428.80 | 3391.42 | 3383.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 3394.25 | 3397.67 | 3388.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 14:00:00 | 3394.25 | 3397.67 | 3388.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 12:15:00 | 3553.50 | 3575.13 | 3538.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 12:45:00 | 3533.40 | 3575.13 | 3538.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 3557.50 | 3570.13 | 3547.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-05 09:45:00 | 3562.80 | 3570.13 | 3547.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 11:15:00 | 3550.60 | 3563.84 | 3548.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 12:45:00 | 3557.75 | 3562.31 | 3549.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 13:15:00 | 3562.00 | 3562.31 | 3549.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 13:45:00 | 3559.25 | 3562.88 | 3550.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-12 09:15:00 | 3617.75 | 3632.91 | 3633.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2024-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 09:15:00 | 3617.75 | 3632.91 | 3633.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 13:15:00 | 3596.05 | 3622.87 | 3628.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-18 09:15:00 | 3536.05 | 3535.41 | 3557.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-18 09:15:00 | 3536.05 | 3535.41 | 3557.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 3536.05 | 3535.41 | 3557.44 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 3581.90 | 3559.96 | 3557.93 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2024-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 14:15:00 | 3497.55 | 3548.88 | 3553.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 3462.10 | 3525.49 | 3541.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-22 09:15:00 | 3511.10 | 3500.65 | 3517.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-22 10:00:00 | 3511.10 | 3500.65 | 3517.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 11:15:00 | 3549.00 | 3512.29 | 3520.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 12:00:00 | 3549.00 | 3512.29 | 3520.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 3560.95 | 3522.02 | 3523.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 13:00:00 | 3560.95 | 3522.02 | 3523.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 3580.05 | 3533.63 | 3528.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 14:15:00 | 3602.40 | 3547.38 | 3535.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 11:15:00 | 3702.85 | 3710.86 | 3660.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 12:00:00 | 3702.85 | 3710.86 | 3660.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 3706.60 | 3702.17 | 3688.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:30:00 | 3695.75 | 3702.17 | 3688.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 3686.85 | 3699.11 | 3688.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:30:00 | 3686.45 | 3699.11 | 3688.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 11:15:00 | 3687.20 | 3696.73 | 3687.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 11:30:00 | 3677.10 | 3696.73 | 3687.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 12:15:00 | 3670.75 | 3691.53 | 3686.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 13:00:00 | 3670.75 | 3691.53 | 3686.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 13:15:00 | 3663.15 | 3685.86 | 3684.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 14:00:00 | 3663.15 | 3685.86 | 3684.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2024-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 14:15:00 | 3672.20 | 3683.12 | 3683.17 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2024-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 09:15:00 | 3700.10 | 3683.94 | 3683.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 10:15:00 | 3738.40 | 3694.83 | 3688.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-02 09:15:00 | 3688.15 | 3710.92 | 3702.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 09:15:00 | 3688.15 | 3710.92 | 3702.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 3688.15 | 3710.92 | 3702.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:00:00 | 3688.15 | 3710.92 | 3702.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 3684.25 | 3705.58 | 3700.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:45:00 | 3682.25 | 3705.58 | 3700.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 3686.35 | 3701.74 | 3699.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 11:30:00 | 3682.35 | 3701.74 | 3699.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 3701.25 | 3700.35 | 3698.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 15:00:00 | 3701.25 | 3700.35 | 3698.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 3716.00 | 3703.48 | 3700.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-03 10:00:00 | 3718.75 | 3706.53 | 3702.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 10:15:00 | 3873.05 | 3902.79 | 3904.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 10:15:00 | 3873.05 | 3902.79 | 3904.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 12:15:00 | 3862.85 | 3889.47 | 3897.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 12:15:00 | 3867.00 | 3856.56 | 3873.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 12:15:00 | 3867.00 | 3856.56 | 3873.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 3867.00 | 3856.56 | 3873.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:00:00 | 3867.00 | 3856.56 | 3873.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 3879.15 | 3861.08 | 3874.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 13:45:00 | 3888.20 | 3861.08 | 3874.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 3888.00 | 3866.46 | 3875.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:30:00 | 3878.20 | 3866.46 | 3875.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 15:15:00 | 3889.25 | 3871.02 | 3876.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 09:15:00 | 3890.90 | 3871.02 | 3876.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 3865.85 | 3873.02 | 3876.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 12:00:00 | 3859.90 | 3870.40 | 3875.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:15:00 | 3856.05 | 3873.29 | 3875.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 09:15:00 | 3666.90 | 3718.44 | 3754.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 12:15:00 | 3663.25 | 3692.83 | 3732.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-24 09:15:00 | 3665.00 | 3648.11 | 3673.85 | SL hit (close>ema200) qty=0.50 sl=3648.11 alert=retest2 |

### Cycle 105 — BUY (started 2025-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 10:15:00 | 3657.00 | 3616.38 | 3613.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 12:15:00 | 3677.55 | 3634.45 | 3622.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 09:15:00 | 3683.30 | 3690.93 | 3668.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 10:00:00 | 3683.30 | 3690.93 | 3668.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 12:15:00 | 3679.10 | 3690.17 | 3673.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 12:30:00 | 3676.60 | 3690.17 | 3673.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 3664.60 | 3685.06 | 3672.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:00:00 | 3664.60 | 3685.06 | 3672.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 3663.25 | 3680.70 | 3672.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 3663.25 | 3680.70 | 3672.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 3660.00 | 3676.56 | 3670.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 3662.80 | 3676.56 | 3670.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 3641.75 | 3670.84 | 3669.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 3641.75 | 3670.84 | 3669.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 3620.00 | 3660.67 | 3664.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 14:15:00 | 3595.10 | 3631.67 | 3649.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 13:15:00 | 3632.65 | 3620.17 | 3634.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 13:15:00 | 3632.65 | 3620.17 | 3634.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 3632.65 | 3620.17 | 3634.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:00:00 | 3632.65 | 3620.17 | 3634.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 3641.40 | 3624.41 | 3634.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 15:00:00 | 3641.40 | 3624.41 | 3634.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 15:15:00 | 3641.00 | 3627.73 | 3635.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 09:15:00 | 3622.25 | 3627.73 | 3635.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 3513.25 | 3480.68 | 3497.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:00:00 | 3513.25 | 3480.68 | 3497.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 3508.05 | 3486.15 | 3498.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:45:00 | 3514.20 | 3486.15 | 3498.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 13:15:00 | 3494.15 | 3492.13 | 3498.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 13:30:00 | 3497.60 | 3492.13 | 3498.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 14:15:00 | 3503.10 | 3494.32 | 3498.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 15:00:00 | 3503.10 | 3494.32 | 3498.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 15:15:00 | 3505.65 | 3496.59 | 3499.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 09:15:00 | 3513.95 | 3496.59 | 3499.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 3489.00 | 3495.07 | 3498.44 | EMA400 retest candle locked (from downside) |

### Cycle 107 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 3525.00 | 3502.68 | 3501.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 13:15:00 | 3535.25 | 3511.79 | 3505.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 14:15:00 | 3503.00 | 3510.03 | 3505.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-16 14:15:00 | 3503.00 | 3510.03 | 3505.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 14:15:00 | 3503.00 | 3510.03 | 3505.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 15:00:00 | 3503.00 | 3510.03 | 3505.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 15:15:00 | 3515.00 | 3511.02 | 3506.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-17 09:15:00 | 3558.45 | 3511.02 | 3506.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-22 10:15:00 | 3532.20 | 3559.55 | 3561.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-01-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 10:15:00 | 3532.20 | 3559.55 | 3561.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 11:15:00 | 3512.40 | 3550.12 | 3557.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 11:15:00 | 3518.00 | 3515.76 | 3532.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 12:00:00 | 3518.00 | 3515.76 | 3532.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 3476.05 | 3504.41 | 3520.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:15:00 | 3466.85 | 3495.12 | 3510.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-27 10:15:00 | 3449.30 | 3482.03 | 3500.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-28 09:15:00 | 3436.45 | 3457.20 | 3475.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-30 09:15:00 | 3505.75 | 3453.46 | 3452.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 3505.75 | 3453.46 | 3452.25 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 11:15:00 | 3417.50 | 3451.64 | 3451.97 | EMA200 below EMA400 |

### Cycle 111 — BUY (started 2025-01-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 09:15:00 | 3561.95 | 3462.71 | 3455.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 09:15:00 | 3619.00 | 3554.78 | 3513.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 3512.00 | 3555.22 | 3521.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 3512.00 | 3555.22 | 3521.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 3512.00 | 3555.22 | 3521.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 3519.35 | 3555.22 | 3521.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 3414.05 | 3526.99 | 3511.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 12:30:00 | 3443.25 | 3526.99 | 3511.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 3415.95 | 3504.78 | 3503.08 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 14:15:00 | 3443.55 | 3492.53 | 3497.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 3313.75 | 3451.29 | 3477.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 3360.00 | 3338.65 | 3393.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 09:45:00 | 3369.00 | 3338.65 | 3393.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 10:15:00 | 3397.85 | 3350.49 | 3394.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 11:00:00 | 3397.85 | 3350.49 | 3394.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 11:15:00 | 3411.35 | 3362.66 | 3395.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-04 12:00:00 | 3411.35 | 3362.66 | 3395.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 3443.70 | 3415.65 | 3411.99 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 14:15:00 | 3382.70 | 3407.72 | 3409.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 09:15:00 | 3361.40 | 3395.05 | 3403.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 14:15:00 | 3331.70 | 3321.74 | 3339.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-10 15:00:00 | 3331.70 | 3321.74 | 3339.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 3323.85 | 3322.16 | 3337.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 09:15:00 | 3318.70 | 3322.16 | 3337.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 3301.15 | 3317.96 | 3334.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 12:00:00 | 3297.35 | 3311.79 | 3328.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 11:15:00 | 3289.80 | 3242.71 | 3236.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 3289.80 | 3242.71 | 3236.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 13:15:00 | 3290.50 | 3258.83 | 3245.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 11:15:00 | 3287.65 | 3302.70 | 3289.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 11:15:00 | 3287.65 | 3302.70 | 3289.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 3287.65 | 3302.70 | 3289.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 12:00:00 | 3287.65 | 3302.70 | 3289.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 3281.10 | 3298.38 | 3288.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 12:45:00 | 3286.40 | 3298.38 | 3288.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 3270.80 | 3292.87 | 3286.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:45:00 | 3272.20 | 3292.87 | 3286.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-02-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 15:15:00 | 3251.00 | 3279.11 | 3281.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 09:15:00 | 3225.70 | 3268.43 | 3276.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 09:15:00 | 3184.25 | 3178.22 | 3199.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 09:15:00 | 3184.25 | 3178.22 | 3199.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 3184.25 | 3178.22 | 3199.85 | EMA400 retest candle locked (from downside) |

### Cycle 117 — BUY (started 2025-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 13:15:00 | 3210.85 | 3202.19 | 3201.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 10:15:00 | 3224.80 | 3210.99 | 3205.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 3250.85 | 3256.12 | 3241.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 3250.85 | 3256.12 | 3241.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 3245.70 | 3253.42 | 3243.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 15:00:00 | 3245.70 | 3253.42 | 3243.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 3246.00 | 3251.94 | 3244.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 3218.00 | 3251.94 | 3244.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 3204.25 | 3242.40 | 3240.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 3204.25 | 3242.40 | 3240.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — SELL (started 2025-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 10:15:00 | 3207.35 | 3235.39 | 3237.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 13:15:00 | 3193.40 | 3218.63 | 3228.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 14:15:00 | 3196.65 | 3192.40 | 3205.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 15:00:00 | 3196.65 | 3192.40 | 3205.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 3175.25 | 3190.79 | 3202.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 10:15:00 | 3166.00 | 3190.79 | 3202.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 10:15:00 | 3228.00 | 3198.20 | 3198.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 10:15:00 | 3228.00 | 3198.20 | 3198.14 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 15:15:00 | 3189.00 | 3197.16 | 3198.17 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 09:15:00 | 3213.30 | 3200.39 | 3199.54 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-03-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 10:15:00 | 3187.50 | 3197.81 | 3198.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 11:15:00 | 3184.35 | 3195.12 | 3197.16 | Break + close below crossover candle low |

### Cycle 123 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 3242.80 | 3195.78 | 3195.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 10:15:00 | 3250.50 | 3206.72 | 3200.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 3289.60 | 3297.24 | 3268.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 09:45:00 | 3296.20 | 3297.24 | 3268.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 3457.85 | 3461.34 | 3441.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 09:45:00 | 3459.40 | 3461.34 | 3441.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 13:15:00 | 3448.05 | 3460.44 | 3447.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 13:45:00 | 3445.50 | 3460.44 | 3447.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 3450.50 | 3458.45 | 3447.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 14:45:00 | 3424.00 | 3458.45 | 3447.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 15:15:00 | 3454.45 | 3457.65 | 3448.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 09:15:00 | 3484.90 | 3457.65 | 3448.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-01 11:15:00 | 3441.50 | 3472.72 | 3476.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 11:15:00 | 3441.50 | 3472.72 | 3476.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-02 12:15:00 | 3419.60 | 3440.66 | 3454.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-03 09:15:00 | 3438.00 | 3432.13 | 3445.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 3438.00 | 3432.13 | 3445.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 3438.00 | 3432.13 | 3445.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:45:00 | 3442.20 | 3432.13 | 3445.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 11:15:00 | 3460.00 | 3438.95 | 3446.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 12:00:00 | 3460.00 | 3438.95 | 3446.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 12:15:00 | 3442.55 | 3439.67 | 3445.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 13:15:00 | 3440.00 | 3439.67 | 3445.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-04 11:15:00 | 3268.00 | 3352.07 | 3397.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-04-07 09:15:00 | 3096.00 | 3240.70 | 3322.22 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 125 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 3260.00 | 3143.66 | 3139.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 12:15:00 | 3265.50 | 3199.18 | 3168.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 13:15:00 | 3238.70 | 3245.87 | 3215.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 13:45:00 | 3237.00 | 3245.87 | 3215.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 3168.00 | 3225.35 | 3213.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:00:00 | 3168.00 | 3225.35 | 3213.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 3170.00 | 3214.28 | 3209.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:00:00 | 3193.70 | 3210.16 | 3208.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 11:15:00 | 3250.00 | 3275.24 | 3276.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 3250.00 | 3275.24 | 3276.78 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 3319.90 | 3283.36 | 3280.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 11:15:00 | 3333.60 | 3296.41 | 3286.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 12:15:00 | 3333.00 | 3336.78 | 3317.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-29 13:00:00 | 3333.00 | 3336.78 | 3317.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 3324.90 | 3331.83 | 3318.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-30 09:15:00 | 3334.90 | 3329.72 | 3318.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 3313.40 | 3334.28 | 3333.80 | SL hit (close<static) qty=1.00 sl=3315.00 alert=retest2 |

### Cycle 128 — SELL (started 2025-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 12:15:00 | 3315.70 | 3329.97 | 3331.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 09:15:00 | 3306.50 | 3324.84 | 3328.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-06 10:15:00 | 3326.70 | 3325.21 | 3328.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-06 11:00:00 | 3326.70 | 3325.21 | 3328.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 3329.60 | 3326.09 | 3328.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 11:45:00 | 3336.80 | 3326.09 | 3328.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 3346.20 | 3330.11 | 3330.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:00:00 | 3346.20 | 3330.11 | 3330.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 13:15:00 | 3333.00 | 3330.69 | 3330.57 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 3300.00 | 3325.44 | 3328.32 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 3352.90 | 3329.59 | 3327.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-09 09:15:00 | 3443.00 | 3352.44 | 3340.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 14:15:00 | 3563.80 | 3564.16 | 3518.33 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:15:00 | 3587.70 | 3565.37 | 3523.05 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 15:00:00 | 3578.50 | 3581.72 | 3551.99 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 3556.90 | 3575.84 | 3554.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:45:00 | 3558.90 | 3575.84 | 3554.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 3562.70 | 3573.21 | 3555.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:00:00 | 3600.80 | 3577.99 | 3560.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-19 12:15:00 | 3592.20 | 3602.03 | 3594.26 | SL hit (close<ema400) qty=1.00 sl=3594.26 alert=retest1 |

### Cycle 132 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 3579.40 | 3594.74 | 3594.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 09:15:00 | 3564.70 | 3583.57 | 3588.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 3586.00 | 3566.17 | 3574.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 3586.00 | 3566.17 | 3574.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 3586.00 | 3566.17 | 3574.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 3586.00 | 3566.17 | 3574.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 3612.80 | 3575.50 | 3577.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:45:00 | 3618.30 | 3575.50 | 3577.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 3599.90 | 3580.38 | 3579.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 3646.70 | 3602.94 | 3591.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 3610.60 | 3630.38 | 3614.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 3610.60 | 3630.38 | 3614.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 3610.60 | 3630.38 | 3614.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 3655.00 | 3630.10 | 3616.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 14:00:00 | 3655.20 | 3641.32 | 3631.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:15:00 | 3656.00 | 3642.82 | 3633.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 15:00:00 | 3657.20 | 3642.89 | 3637.34 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 3653.00 | 3664.26 | 3655.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 13:00:00 | 3684.50 | 3665.94 | 3658.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 14:45:00 | 3684.60 | 3671.61 | 3662.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 11:15:00 | 3644.80 | 3656.64 | 3657.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 11:15:00 | 3644.80 | 3656.64 | 3657.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 3629.50 | 3651.21 | 3654.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 09:15:00 | 3632.10 | 3631.75 | 3639.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 09:15:00 | 3632.10 | 3631.75 | 3639.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 3632.10 | 3631.75 | 3639.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:30:00 | 3635.30 | 3631.75 | 3639.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 3668.90 | 3638.53 | 3641.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:00:00 | 3668.90 | 3638.53 | 3641.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 12:15:00 | 3661.20 | 3643.06 | 3643.16 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 13:15:00 | 3647.30 | 3643.91 | 3643.53 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2025-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 15:15:00 | 3641.20 | 3643.18 | 3643.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 09:15:00 | 3636.80 | 3641.90 | 3642.67 | Break + close below crossover candle low |

### Cycle 137 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 3653.30 | 3644.18 | 3643.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 11:15:00 | 3660.60 | 3647.46 | 3645.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 13:15:00 | 3674.70 | 3678.24 | 3670.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 13:45:00 | 3677.70 | 3678.24 | 3670.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 3674.20 | 3677.37 | 3671.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 14:15:00 | 3690.80 | 3681.48 | 3675.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 09:15:00 | 3701.00 | 3682.78 | 3677.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 3655.10 | 3675.62 | 3674.91 | SL hit (close<static) qty=1.00 sl=3656.10 alert=retest2 |

### Cycle 138 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 3658.80 | 3672.25 | 3673.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 12:15:00 | 3637.00 | 3665.20 | 3670.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 3606.00 | 3591.90 | 3613.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 3606.00 | 3591.90 | 3613.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 3606.00 | 3591.90 | 3613.31 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2025-06-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 14:15:00 | 3629.90 | 3623.51 | 3623.36 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 09:15:00 | 3610.20 | 3621.40 | 3622.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 11:15:00 | 3601.10 | 3614.45 | 3617.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 3615.00 | 3606.97 | 3611.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 3615.00 | 3606.97 | 3611.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 3615.00 | 3606.97 | 3611.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:45:00 | 3623.00 | 3606.97 | 3611.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 3613.00 | 3608.17 | 3611.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:45:00 | 3617.60 | 3608.17 | 3611.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 11:15:00 | 3616.30 | 3609.80 | 3612.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:00:00 | 3616.30 | 3609.80 | 3612.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 3617.10 | 3611.26 | 3612.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:30:00 | 3611.60 | 3611.26 | 3612.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 3620.10 | 3613.03 | 3613.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:45:00 | 3636.30 | 3613.03 | 3613.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2025-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-19 14:15:00 | 3621.30 | 3614.68 | 3614.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 09:15:00 | 3637.70 | 3619.43 | 3616.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 3626.80 | 3643.22 | 3633.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 3626.80 | 3643.22 | 3633.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 3626.80 | 3643.22 | 3633.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:00:00 | 3626.80 | 3643.22 | 3633.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 3627.40 | 3640.05 | 3632.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:00:00 | 3627.40 | 3640.05 | 3632.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 3610.80 | 3634.20 | 3630.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 12:00:00 | 3610.80 | 3634.20 | 3630.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-23 12:15:00 | 3594.30 | 3626.22 | 3627.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-23 14:15:00 | 3586.50 | 3613.01 | 3620.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 3675.80 | 3620.13 | 3622.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 3675.80 | 3620.13 | 3622.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 3675.80 | 3620.13 | 3622.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:00:00 | 3675.80 | 3620.13 | 3622.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2025-06-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 10:15:00 | 3666.20 | 3629.34 | 3626.37 | EMA200 above EMA400 |

### Cycle 144 — SELL (started 2025-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 15:15:00 | 3612.00 | 3624.95 | 3626.52 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 09:15:00 | 3645.80 | 3626.23 | 3625.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 13:15:00 | 3657.00 | 3638.08 | 3631.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 14:15:00 | 3678.00 | 3679.85 | 3661.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 15:00:00 | 3678.00 | 3679.85 | 3661.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 12:15:00 | 3663.00 | 3680.31 | 3669.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:00:00 | 3663.00 | 3680.31 | 3669.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 3660.40 | 3676.33 | 3668.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-30 13:45:00 | 3661.50 | 3676.33 | 3668.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 3671.00 | 3674.62 | 3668.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 3667.90 | 3674.62 | 3668.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 3665.30 | 3672.75 | 3668.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:45:00 | 3673.80 | 3672.75 | 3668.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 3660.60 | 3670.32 | 3667.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:45:00 | 3663.80 | 3670.32 | 3667.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 3651.30 | 3666.52 | 3666.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:45:00 | 3649.10 | 3666.52 | 3666.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 12:15:00 | 3661.40 | 3665.49 | 3665.95 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 13:15:00 | 3669.90 | 3666.38 | 3666.31 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 14:15:00 | 3665.70 | 3666.24 | 3666.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 09:15:00 | 3646.90 | 3662.17 | 3664.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 3592.40 | 3590.26 | 3605.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 15:00:00 | 3592.40 | 3590.26 | 3605.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 3592.80 | 3591.86 | 3603.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 3600.90 | 3591.86 | 3603.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 3582.60 | 3584.27 | 3591.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:30:00 | 3589.30 | 3584.27 | 3591.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 3601.00 | 3587.62 | 3592.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:00:00 | 3601.00 | 3587.62 | 3592.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 3610.30 | 3592.15 | 3594.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 3610.30 | 3592.15 | 3594.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 3607.00 | 3595.12 | 3595.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 3579.50 | 3595.12 | 3595.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 13:15:00 | 3495.60 | 3485.91 | 3485.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2025-07-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 13:15:00 | 3495.60 | 3485.91 | 3485.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 14:15:00 | 3506.90 | 3490.11 | 3487.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 09:15:00 | 3482.60 | 3490.67 | 3488.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 3482.60 | 3490.67 | 3488.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 3482.60 | 3490.67 | 3488.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:45:00 | 3479.00 | 3490.67 | 3488.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 3473.00 | 3487.13 | 3486.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 11:00:00 | 3473.00 | 3487.13 | 3486.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — SELL (started 2025-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 11:15:00 | 3460.00 | 3481.71 | 3484.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 12:15:00 | 3457.20 | 3476.81 | 3481.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 3471.60 | 3471.54 | 3477.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 3471.60 | 3471.54 | 3477.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 3471.60 | 3471.54 | 3477.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 10:45:00 | 3463.70 | 3469.63 | 3475.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 12:45:00 | 3463.90 | 3467.61 | 3473.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 12:45:00 | 3465.10 | 3471.66 | 3473.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:30:00 | 3465.20 | 3470.71 | 3472.99 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 3461.40 | 3465.37 | 3469.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:00:00 | 3461.40 | 3465.37 | 3469.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 3442.50 | 3429.52 | 3440.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:00:00 | 3442.50 | 3429.52 | 3440.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 3445.30 | 3432.68 | 3440.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:30:00 | 3452.00 | 3432.68 | 3440.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 3472.80 | 3440.70 | 3443.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 3472.80 | 3440.70 | 3443.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-29 13:15:00 | 3486.70 | 3449.90 | 3447.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 151 — BUY (started 2025-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 13:15:00 | 3486.70 | 3449.90 | 3447.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 14:15:00 | 3495.60 | 3459.04 | 3452.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 3640.30 | 3641.68 | 3595.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 15:00:00 | 3640.30 | 3641.68 | 3595.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 3606.50 | 3623.10 | 3603.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:30:00 | 3597.00 | 3623.10 | 3603.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 3605.70 | 3619.62 | 3603.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:30:00 | 3600.30 | 3619.62 | 3603.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 3584.00 | 3612.49 | 3601.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 3584.00 | 3612.49 | 3601.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 3595.20 | 3609.03 | 3600.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:15:00 | 3598.90 | 3609.03 | 3600.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 3624.80 | 3610.89 | 3603.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 11:15:00 | 3630.50 | 3610.89 | 3603.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 13:45:00 | 3628.50 | 3619.06 | 3609.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 11:45:00 | 3627.00 | 3624.01 | 3615.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-06 15:00:00 | 3629.70 | 3635.79 | 3630.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 3633.00 | 3635.23 | 3630.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 3620.70 | 3635.23 | 3630.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 3614.70 | 3631.12 | 3629.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:00:00 | 3614.70 | 3631.12 | 3629.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-07 10:15:00 | 3609.80 | 3626.86 | 3627.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 3609.80 | 3626.86 | 3627.33 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2025-08-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 15:15:00 | 3641.80 | 3628.19 | 3627.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-08 10:15:00 | 3644.10 | 3632.40 | 3629.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 11:15:00 | 3631.70 | 3632.26 | 3629.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 11:15:00 | 3631.70 | 3632.26 | 3629.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 3631.70 | 3632.26 | 3629.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:00:00 | 3631.70 | 3632.26 | 3629.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 3630.20 | 3631.85 | 3629.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:15:00 | 3628.80 | 3631.85 | 3629.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 3614.70 | 3628.42 | 3628.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 14:00:00 | 3614.70 | 3628.42 | 3628.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — SELL (started 2025-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 14:15:00 | 3604.20 | 3623.57 | 3625.99 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 3652.50 | 3630.98 | 3628.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 15:15:00 | 3673.00 | 3648.46 | 3637.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 15:15:00 | 3679.80 | 3683.30 | 3665.23 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-13 09:15:00 | 3697.10 | 3683.30 | 3665.23 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 3671.30 | 3691.74 | 3681.99 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-08-14 09:15:00 | 3671.30 | 3691.74 | 3681.99 | SL hit (close<ema400) qty=1.00 sl=3681.99 alert=retest1 |

### Cycle 156 — SELL (started 2025-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 12:15:00 | 3665.50 | 3675.72 | 3676.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 09:15:00 | 3648.90 | 3669.70 | 3673.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-21 09:15:00 | 3633.50 | 3609.55 | 3621.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 3633.50 | 3609.55 | 3621.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 3633.50 | 3609.55 | 3621.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 10:00:00 | 3633.50 | 3609.55 | 3621.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 3634.60 | 3614.56 | 3622.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 11:45:00 | 3627.50 | 3616.65 | 3622.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 11:15:00 | 3613.00 | 3581.66 | 3580.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 157 — BUY (started 2025-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 11:15:00 | 3613.00 | 3581.66 | 3580.17 | EMA200 above EMA400 |

### Cycle 158 — SELL (started 2025-09-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 14:15:00 | 3578.20 | 3592.17 | 3592.98 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 15:15:00 | 3601.70 | 3589.55 | 3589.44 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 3577.70 | 3589.17 | 3589.79 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 15:15:00 | 3592.60 | 3590.58 | 3590.37 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 3579.20 | 3588.31 | 3589.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 3574.90 | 3585.63 | 3588.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 3559.30 | 3538.16 | 3546.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 3559.30 | 3538.16 | 3546.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 3559.30 | 3538.16 | 3546.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:30:00 | 3547.90 | 3540.27 | 3547.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 14:00:00 | 3548.60 | 3546.00 | 3548.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 10:00:00 | 3548.80 | 3548.80 | 3549.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 3582.80 | 3550.59 | 3548.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 3582.80 | 3550.59 | 3548.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 10:15:00 | 3588.50 | 3558.17 | 3552.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 14:15:00 | 3584.40 | 3588.41 | 3577.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 15:00:00 | 3584.40 | 3588.41 | 3577.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 14:15:00 | 3674.90 | 3683.46 | 3675.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 15:00:00 | 3674.90 | 3683.46 | 3675.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 15:15:00 | 3684.00 | 3683.56 | 3676.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 09:15:00 | 3662.70 | 3683.56 | 3676.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 3673.00 | 3681.45 | 3675.92 | EMA400 retest candle locked (from upside) |

### Cycle 164 — SELL (started 2025-09-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 12:15:00 | 3652.50 | 3669.63 | 3671.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 13:15:00 | 3649.40 | 3665.58 | 3669.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 3647.60 | 3644.35 | 3655.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 13:00:00 | 3647.60 | 3644.35 | 3655.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 3661.80 | 3647.84 | 3655.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:00:00 | 3661.80 | 3647.84 | 3655.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 3660.30 | 3650.33 | 3656.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:30:00 | 3659.40 | 3650.33 | 3656.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 3660.00 | 3652.27 | 3656.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 09:15:00 | 3649.00 | 3652.27 | 3656.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 09:15:00 | 3679.60 | 3657.73 | 3658.54 | SL hit (close>static) qty=1.00 sl=3665.60 alert=retest2 |

### Cycle 165 — BUY (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 10:15:00 | 3669.60 | 3660.11 | 3659.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-24 11:15:00 | 3699.20 | 3667.92 | 3663.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 14:15:00 | 3673.50 | 3678.42 | 3670.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-24 15:00:00 | 3673.50 | 3678.42 | 3670.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 15:15:00 | 3674.20 | 3677.58 | 3670.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 09:15:00 | 3691.90 | 3677.58 | 3670.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 3679.50 | 3677.96 | 3671.24 | EMA400 retest candle locked (from upside) |

### Cycle 166 — SELL (started 2025-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 14:15:00 | 3645.50 | 3665.28 | 3667.47 | EMA200 below EMA400 |

### Cycle 167 — BUY (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-26 09:15:00 | 3703.70 | 3670.52 | 3669.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-26 10:15:00 | 3747.00 | 3685.82 | 3676.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-29 09:15:00 | 3707.90 | 3723.50 | 3704.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 3707.90 | 3723.50 | 3704.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 3707.90 | 3723.50 | 3704.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 3695.40 | 3723.50 | 3704.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 3714.50 | 3721.70 | 3705.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:30:00 | 3705.90 | 3721.70 | 3705.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 11:15:00 | 3695.30 | 3716.42 | 3704.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 12:00:00 | 3695.30 | 3716.42 | 3704.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 12:15:00 | 3708.00 | 3714.74 | 3705.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 12:30:00 | 3698.40 | 3714.74 | 3705.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 3704.10 | 3712.61 | 3704.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 3704.10 | 3712.61 | 3704.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 3685.40 | 3707.17 | 3703.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 3685.40 | 3707.17 | 3703.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 3704.30 | 3706.59 | 3703.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 3674.70 | 3706.59 | 3703.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 168 — SELL (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 09:15:00 | 3655.80 | 3696.43 | 3698.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-01 10:15:00 | 3626.00 | 3659.37 | 3675.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 13:15:00 | 3666.00 | 3659.78 | 3671.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-01 14:00:00 | 3666.00 | 3659.78 | 3671.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 14:15:00 | 3670.10 | 3661.85 | 3671.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:45:00 | 3667.30 | 3661.85 | 3671.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 3685.00 | 3666.48 | 3672.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 3674.20 | 3666.48 | 3672.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 3680.30 | 3669.24 | 3673.37 | EMA400 retest candle locked (from downside) |

### Cycle 169 — BUY (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 11:15:00 | 3697.10 | 3676.42 | 3676.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 3730.90 | 3690.49 | 3682.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 13:15:00 | 3740.40 | 3740.99 | 3725.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 14:00:00 | 3740.40 | 3740.99 | 3725.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 3729.00 | 3738.60 | 3725.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 15:00:00 | 3729.00 | 3738.60 | 3725.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 3756.10 | 3740.80 | 3729.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 10:15:00 | 3763.10 | 3738.50 | 3732.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 13:15:00 | 3766.00 | 3753.60 | 3742.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 10:15:00 | 3767.80 | 3775.14 | 3765.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 12:30:00 | 3764.00 | 3768.63 | 3764.68 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 3759.00 | 3766.71 | 3764.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:45:00 | 3754.90 | 3766.71 | 3764.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 14:15:00 | 3768.90 | 3767.15 | 3764.59 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-14 11:15:00 | 3755.40 | 3763.38 | 3763.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 3755.40 | 3763.38 | 3763.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 3738.70 | 3758.44 | 3761.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 3786.00 | 3756.86 | 3758.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 3786.00 | 3756.86 | 3758.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 3786.00 | 3756.86 | 3758.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 3786.00 | 3756.86 | 3758.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 3818.30 | 3769.15 | 3764.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 11:15:00 | 3837.00 | 3782.72 | 3770.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 12:15:00 | 3843.00 | 3853.14 | 3834.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 13:00:00 | 3843.00 | 3853.14 | 3834.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 3837.00 | 3849.92 | 3834.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:30:00 | 3844.60 | 3849.92 | 3834.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 3840.00 | 3847.93 | 3835.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:45:00 | 3838.00 | 3847.93 | 3835.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 3906.20 | 3916.73 | 3905.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:45:00 | 3901.20 | 3916.73 | 3905.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 3905.00 | 3914.39 | 3905.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:15:00 | 3924.20 | 3914.39 | 3905.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 3955.50 | 3990.16 | 3994.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 172 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 3955.50 | 3990.16 | 3994.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 10:15:00 | 3946.00 | 3981.33 | 3990.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 14:15:00 | 3883.20 | 3881.69 | 3907.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 15:00:00 | 3883.20 | 3881.69 | 3907.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 3912.00 | 3887.82 | 3905.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 3911.00 | 3887.82 | 3905.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 3923.50 | 3894.95 | 3907.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:30:00 | 3923.40 | 3894.95 | 3907.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 173 — BUY (started 2025-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 15:15:00 | 3916.80 | 3914.38 | 3914.06 | EMA200 above EMA400 |

### Cycle 174 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 3902.70 | 3912.54 | 3913.31 | EMA200 below EMA400 |

### Cycle 175 — BUY (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 11:15:00 | 3921.90 | 3914.41 | 3914.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 12:15:00 | 3940.40 | 3919.61 | 3916.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 12:15:00 | 3985.00 | 3987.55 | 3971.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 13:00:00 | 3985.00 | 3987.55 | 3971.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 3980.40 | 3986.12 | 3972.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:45:00 | 3979.80 | 3986.12 | 3972.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 4009.40 | 3990.77 | 3975.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 14:45:00 | 3975.70 | 3990.77 | 3975.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 4002.00 | 4012.06 | 3998.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:15:00 | 3984.90 | 4012.06 | 3998.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 3988.20 | 4007.29 | 3997.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 3984.50 | 4007.29 | 3997.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 3992.50 | 4004.33 | 3996.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 11:30:00 | 3992.10 | 4004.33 | 3996.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 3999.80 | 4003.43 | 3997.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:30:00 | 3991.40 | 4003.43 | 3997.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 4009.40 | 4004.62 | 3998.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 11:30:00 | 4014.00 | 4001.58 | 3998.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:00:00 | 4012.00 | 4001.58 | 3998.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 12:45:00 | 4011.10 | 4004.67 | 4000.29 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-24 13:15:00 | 4014.00 | 4020.53 | 4020.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 176 — SELL (started 2025-11-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 13:15:00 | 4014.00 | 4020.53 | 4020.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 15:15:00 | 4006.50 | 4016.31 | 4018.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 4027.70 | 4018.59 | 4019.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 4027.70 | 4018.59 | 4019.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 4027.70 | 4018.59 | 4019.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:15:00 | 4029.70 | 4018.59 | 4019.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 4008.60 | 4016.59 | 4018.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:30:00 | 4020.80 | 4016.59 | 4018.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 177 — BUY (started 2025-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 11:15:00 | 4040.60 | 4021.39 | 4020.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 12:15:00 | 4052.00 | 4027.51 | 4023.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 14:15:00 | 3999.00 | 4023.04 | 4022.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 14:15:00 | 3999.00 | 4023.04 | 4022.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 3999.00 | 4023.04 | 4022.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 3999.00 | 4023.04 | 4022.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — SELL (started 2025-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 15:15:00 | 3982.20 | 4014.87 | 4018.55 | EMA200 below EMA400 |

### Cycle 179 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 4048.10 | 4025.39 | 4022.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 09:15:00 | 4104.70 | 4059.35 | 4042.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 14:15:00 | 4079.90 | 4083.90 | 4063.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 15:00:00 | 4079.90 | 4083.90 | 4063.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 4098.90 | 4086.28 | 4068.30 | EMA400 retest candle locked (from upside) |

### Cycle 180 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 4059.10 | 4067.51 | 4068.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 10:15:00 | 4055.40 | 4065.09 | 4067.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 10:15:00 | 4000.00 | 3999.14 | 4018.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:30:00 | 4004.50 | 3999.14 | 4018.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 4000.70 | 3996.12 | 4008.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:00:00 | 4000.70 | 3996.12 | 4008.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 4010.90 | 3999.08 | 4008.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:00:00 | 4010.90 | 3999.08 | 4008.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 4017.90 | 4002.84 | 4009.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:00:00 | 4017.90 | 4002.84 | 4009.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 4035.90 | 4009.45 | 4011.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 13:00:00 | 4035.90 | 4009.45 | 4011.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 181 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 4037.80 | 4015.12 | 4014.08 | EMA200 above EMA400 |

### Cycle 182 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 4003.00 | 4015.14 | 4015.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 3992.10 | 4008.45 | 4012.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 4012.40 | 4000.69 | 4006.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 11:15:00 | 4012.40 | 4000.69 | 4006.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 4012.40 | 4000.69 | 4006.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 4012.40 | 4000.69 | 4006.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 4008.60 | 4002.27 | 4006.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:15:00 | 3988.80 | 4003.22 | 4006.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 09:45:00 | 3991.80 | 4000.30 | 4004.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 11:15:00 | 4020.50 | 4007.01 | 4006.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 183 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 4020.50 | 4007.01 | 4006.65 | EMA200 above EMA400 |

### Cycle 184 — SELL (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 13:15:00 | 3996.80 | 4006.63 | 4006.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 15:15:00 | 3990.60 | 4001.13 | 4004.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 4003.20 | 4001.55 | 4003.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 4003.20 | 4001.55 | 4003.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 4003.20 | 4001.55 | 4003.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 09:30:00 | 3992.60 | 4001.55 | 4003.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 3996.90 | 4000.62 | 4003.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:30:00 | 4005.30 | 4000.62 | 4003.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 4007.00 | 4001.89 | 4003.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 4007.00 | 4001.89 | 4003.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 3992.40 | 4000.00 | 4002.61 | EMA400 retest candle locked (from downside) |

### Cycle 185 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 4077.30 | 4017.10 | 4009.84 | EMA200 above EMA400 |

### Cycle 186 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 4048.80 | 4059.10 | 4059.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 4031.20 | 4053.00 | 4056.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 11:15:00 | 4057.00 | 4052.27 | 4055.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 11:15:00 | 4057.00 | 4052.27 | 4055.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 11:15:00 | 4057.00 | 4052.27 | 4055.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:00:00 | 4057.00 | 4052.27 | 4055.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 12:15:00 | 4058.00 | 4053.41 | 4055.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 12:30:00 | 4060.10 | 4053.41 | 4055.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 4042.40 | 4051.21 | 4054.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:15:00 | 4031.20 | 4051.21 | 4054.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 4069.80 | 4048.42 | 4051.77 | SL hit (close>static) qty=1.00 sl=4059.60 alert=retest2 |

### Cycle 187 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 4077.00 | 4054.14 | 4054.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 11:15:00 | 4087.70 | 4060.85 | 4057.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 13:15:00 | 4071.40 | 4074.31 | 4068.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 14:00:00 | 4071.40 | 4074.31 | 4068.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 4078.20 | 4077.09 | 4072.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:45:00 | 4076.70 | 4077.09 | 4072.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 4062.10 | 4074.52 | 4072.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 15:00:00 | 4062.10 | 4074.52 | 4072.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 4070.00 | 4073.61 | 4072.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:45:00 | 4055.70 | 4071.09 | 4071.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 188 — SELL (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 10:15:00 | 4067.00 | 4070.27 | 4070.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 13:15:00 | 4054.90 | 4064.24 | 4067.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 15:15:00 | 4051.00 | 4047.39 | 4054.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 09:15:00 | 4052.50 | 4047.39 | 4054.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 4054.40 | 4048.79 | 4054.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:30:00 | 4055.30 | 4048.79 | 4054.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 4044.70 | 4047.97 | 4053.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 13:30:00 | 4037.00 | 4045.40 | 4050.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 10:30:00 | 4039.90 | 4040.16 | 4046.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 12:15:00 | 4058.00 | 4045.61 | 4047.88 | SL hit (close>static) qty=1.00 sl=4056.90 alert=retest2 |

### Cycle 189 — BUY (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 15:15:00 | 4058.00 | 4051.02 | 4050.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 10:15:00 | 4063.90 | 4053.45 | 4051.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 12:15:00 | 4161.80 | 4164.31 | 4142.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 12:45:00 | 4165.70 | 4164.31 | 4142.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 4140.00 | 4159.45 | 4141.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 4140.00 | 4159.45 | 4141.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 4149.20 | 4157.40 | 4142.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 10:00:00 | 4182.00 | 4161.29 | 4146.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 11:30:00 | 4165.00 | 4159.68 | 4148.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 09:15:00 | 4130.00 | 4147.31 | 4146.40 | SL hit (close<static) qty=1.00 sl=4137.00 alert=retest2 |

### Cycle 190 — SELL (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 10:15:00 | 4133.10 | 4144.47 | 4145.19 | EMA200 below EMA400 |

### Cycle 191 — BUY (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 15:15:00 | 4150.10 | 4145.45 | 4145.06 | EMA200 above EMA400 |

### Cycle 192 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 4124.70 | 4144.41 | 4145.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 12:15:00 | 4112.00 | 4137.93 | 4142.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 4005.70 | 4004.54 | 4040.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 13:30:00 | 4001.70 | 4004.54 | 4040.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 3920.00 | 3992.32 | 4026.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:15:00 | 3897.00 | 3968.66 | 4009.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 15:15:00 | 3804.50 | 3774.65 | 3773.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 193 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 3804.50 | 3774.65 | 3773.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 09:15:00 | 3888.00 | 3810.73 | 3794.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 3917.40 | 3943.68 | 3909.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 3917.40 | 3943.68 | 3909.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 3917.40 | 3943.68 | 3909.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 3917.40 | 3943.68 | 3909.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 3934.90 | 3941.92 | 3911.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 3878.60 | 3941.92 | 3911.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 3873.50 | 3928.24 | 3907.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:45:00 | 3879.60 | 3928.24 | 3907.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 3820.60 | 3906.71 | 3899.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 3820.60 | 3906.71 | 3899.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 194 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 3800.00 | 3885.37 | 3890.89 | EMA200 below EMA400 |

### Cycle 195 — BUY (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 12:15:00 | 3917.80 | 3894.04 | 3893.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 14:15:00 | 3925.50 | 3902.89 | 3897.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 4044.00 | 4065.00 | 4029.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:45:00 | 4043.00 | 4065.00 | 4029.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 4066.80 | 4063.56 | 4045.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 09:15:00 | 4095.50 | 4063.66 | 4053.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 13:15:00 | 4259.80 | 4338.16 | 4344.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 196 — SELL (started 2026-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 13:15:00 | 4259.80 | 4338.16 | 4344.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 4057.70 | 4240.10 | 4274.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 3950.30 | 3917.74 | 4010.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 09:45:00 | 3965.90 | 3917.74 | 4010.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 4023.70 | 3968.16 | 4002.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 4063.70 | 3968.16 | 4002.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 4027.00 | 3979.93 | 4004.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 3960.60 | 3979.93 | 4004.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 3762.57 | 3926.53 | 3964.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 3863.80 | 3850.21 | 3895.68 | SL hit (close>ema200) qty=0.50 sl=3850.21 alert=retest2 |

### Cycle 197 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 3617.70 | 3553.21 | 3550.47 | EMA200 above EMA400 |

### Cycle 198 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 3458.90 | 3545.95 | 3553.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 11:15:00 | 3445.00 | 3525.76 | 3543.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 3483.60 | 3477.90 | 3509.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 3483.60 | 3477.90 | 3509.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 3483.60 | 3477.90 | 3509.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 09:30:00 | 3502.90 | 3477.90 | 3509.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 3419.00 | 3366.77 | 3405.09 | EMA400 retest candle locked (from downside) |

### Cycle 199 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 3517.50 | 3431.97 | 3426.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 3620.70 | 3494.35 | 3458.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 3555.80 | 3596.14 | 3540.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:45:00 | 3554.80 | 3596.14 | 3540.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 3526.00 | 3569.19 | 3553.09 | EMA400 retest candle locked (from upside) |

### Cycle 200 — SELL (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 13:15:00 | 3510.80 | 3539.45 | 3542.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 3505.80 | 3532.72 | 3538.95 | Break + close below crossover candle low |

### Cycle 201 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 3611.10 | 3543.96 | 3542.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 11:15:00 | 3659.20 | 3578.66 | 3559.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 3484.60 | 3582.85 | 3572.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 3484.60 | 3582.85 | 3572.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 3484.60 | 3582.85 | 3572.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:00:00 | 3484.60 | 3582.85 | 3572.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 202 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 3483.40 | 3562.96 | 3564.77 | EMA200 below EMA400 |

### Cycle 203 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 3615.00 | 3569.40 | 3565.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 3684.40 | 3610.64 | 3588.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 3933.90 | 3937.83 | 3838.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:45:00 | 3930.30 | 3937.83 | 3838.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 3923.00 | 3940.96 | 3910.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:15:00 | 3946.10 | 3939.14 | 3914.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:00:00 | 3946.40 | 3940.59 | 3917.69 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 13:15:00 | 4034.60 | 4070.44 | 4071.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — SELL (started 2026-04-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 13:15:00 | 4034.60 | 4070.44 | 4071.84 | EMA200 below EMA400 |

### Cycle 205 — BUY (started 2026-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 12:15:00 | 4083.10 | 4071.93 | 4071.19 | EMA200 above EMA400 |

### Cycle 206 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 4031.00 | 4063.66 | 4067.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 12:15:00 | 4020.00 | 4046.56 | 4058.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 09:15:00 | 4039.80 | 4036.41 | 4048.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 09:15:00 | 4039.80 | 4036.41 | 4048.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 4039.80 | 4036.41 | 4048.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-23 10:00:00 | 4039.80 | 4036.41 | 4048.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 4031.40 | 4035.40 | 4047.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 09:45:00 | 4018.70 | 4039.86 | 4044.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 10:30:00 | 4018.80 | 4032.67 | 4041.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 4065.70 | 4024.78 | 4030.88 | SL hit (close>static) qty=1.00 sl=4048.40 alert=retest2 |

### Cycle 207 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 4065.10 | 4040.24 | 4037.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 12:15:00 | 4081.00 | 4048.39 | 4041.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-27 14:15:00 | 4051.40 | 4053.97 | 4045.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-27 15:00:00 | 4051.40 | 4053.97 | 4045.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 4078.00 | 4058.94 | 4049.12 | EMA400 retest candle locked (from upside) |

### Cycle 208 — SELL (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 14:15:00 | 4038.00 | 4046.02 | 4046.21 | EMA200 below EMA400 |

### Cycle 209 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 4082.30 | 4052.95 | 4049.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 10:15:00 | 4099.40 | 4062.24 | 4053.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 15:15:00 | 4083.10 | 4088.47 | 4072.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 09:15:00 | 4035.50 | 4088.47 | 4072.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 4021.70 | 4075.12 | 4068.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 4021.70 | 4075.12 | 4068.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 3997.60 | 4059.61 | 4061.65 | EMA200 below EMA400 |

### Cycle 211 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 4110.00 | 4055.82 | 4053.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 12:15:00 | 4117.10 | 4076.55 | 4063.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 09:15:00 | 4047.00 | 4081.14 | 4071.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 4047.00 | 4081.14 | 4071.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 4047.00 | 4081.14 | 4071.28 | EMA400 retest candle locked (from upside) |

### Cycle 212 — SELL (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 14:15:00 | 4059.60 | 4064.84 | 4065.54 | EMA200 below EMA400 |

### Cycle 213 — BUY (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 15:15:00 | 4075.00 | 4066.88 | 4066.40 | EMA200 above EMA400 |

### Cycle 214 — SELL (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 09:15:00 | 3919.50 | 4037.40 | 4053.05 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-05-24 11:45:00 | 2196.60 | 2023-05-26 09:15:00 | 2210.70 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2023-06-01 12:00:00 | 2210.00 | 2023-06-02 10:15:00 | 2218.95 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2023-06-01 13:45:00 | 2206.60 | 2023-06-02 10:15:00 | 2218.95 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2023-06-01 15:00:00 | 2209.45 | 2023-06-02 10:15:00 | 2218.95 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2023-06-13 12:00:00 | 2346.60 | 2023-06-26 10:15:00 | 2379.05 | STOP_HIT | 1.00 | 1.38% |
| BUY | retest2 | 2023-06-14 09:45:00 | 2348.15 | 2023-06-26 10:15:00 | 2379.05 | STOP_HIT | 1.00 | 1.32% |
| SELL | retest2 | 2023-06-27 14:15:00 | 2386.60 | 2023-06-28 09:15:00 | 2405.30 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2023-07-06 10:15:00 | 2493.30 | 2023-07-07 12:15:00 | 2453.60 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2023-07-12 14:00:00 | 2474.90 | 2023-07-12 14:15:00 | 2453.75 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2023-07-13 09:15:00 | 2470.30 | 2023-07-13 13:15:00 | 2453.30 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2023-07-18 09:15:00 | 2478.90 | 2023-08-02 11:15:00 | 2630.85 | STOP_HIT | 1.00 | 6.13% |
| BUY | retest2 | 2023-07-18 10:15:00 | 2477.05 | 2023-08-02 11:15:00 | 2630.85 | STOP_HIT | 1.00 | 6.21% |
| BUY | retest2 | 2023-07-18 15:15:00 | 2479.00 | 2023-08-02 11:15:00 | 2630.85 | STOP_HIT | 1.00 | 6.13% |
| BUY | retest2 | 2023-07-20 10:15:00 | 2481.00 | 2023-08-02 11:15:00 | 2630.85 | STOP_HIT | 1.00 | 6.04% |
| BUY | retest2 | 2023-07-20 12:15:00 | 2488.95 | 2023-08-02 11:15:00 | 2630.85 | STOP_HIT | 1.00 | 5.70% |
| BUY | retest2 | 2023-07-20 14:15:00 | 2489.00 | 2023-08-02 11:15:00 | 2630.85 | STOP_HIT | 1.00 | 5.70% |
| BUY | retest2 | 2023-07-20 15:00:00 | 2491.05 | 2023-08-02 11:15:00 | 2630.85 | STOP_HIT | 1.00 | 5.61% |
| SELL | retest2 | 2023-08-03 12:45:00 | 2618.60 | 2023-08-07 09:15:00 | 2651.70 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2023-08-04 12:30:00 | 2625.75 | 2023-08-07 09:15:00 | 2651.70 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2023-08-04 15:00:00 | 2624.40 | 2023-08-07 09:15:00 | 2651.70 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2023-09-05 09:15:00 | 2731.80 | 2023-09-12 09:15:00 | 3004.98 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-06 12:45:00 | 2717.50 | 2023-09-12 09:15:00 | 2989.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-06 14:30:00 | 2722.10 | 2023-09-12 09:15:00 | 2994.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-06 15:00:00 | 2730.10 | 2023-09-12 09:15:00 | 3003.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2023-09-18 09:45:00 | 2925.65 | 2023-09-20 11:15:00 | 2909.75 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2023-09-18 10:15:00 | 2927.85 | 2023-09-20 11:15:00 | 2909.75 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2023-09-18 10:45:00 | 2928.05 | 2023-09-20 11:15:00 | 2909.75 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2023-09-20 09:15:00 | 2938.50 | 2023-09-20 11:15:00 | 2909.75 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2023-10-05 09:15:00 | 3063.00 | 2023-10-12 13:15:00 | 3078.35 | STOP_HIT | 1.00 | 0.50% |
| BUY | retest2 | 2023-10-05 10:00:00 | 3047.55 | 2023-10-12 13:15:00 | 3078.35 | STOP_HIT | 1.00 | 1.01% |
| BUY | retest2 | 2023-10-05 10:45:00 | 3054.00 | 2023-10-12 13:15:00 | 3078.35 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest2 | 2023-10-20 09:15:00 | 3040.45 | 2023-10-26 09:15:00 | 2888.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-10-20 09:15:00 | 3040.45 | 2023-10-27 09:15:00 | 2884.75 | STOP_HIT | 0.50 | 5.12% |
| BUY | retest2 | 2023-11-06 09:15:00 | 2942.95 | 2023-11-23 09:15:00 | 3070.55 | STOP_HIT | 1.00 | 4.34% |
| SELL | retest2 | 2023-11-28 10:00:00 | 3043.60 | 2023-11-29 12:15:00 | 3061.40 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2023-11-28 14:30:00 | 3049.90 | 2023-11-29 12:15:00 | 3061.40 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2023-12-29 09:15:00 | 3527.05 | 2023-12-29 10:15:00 | 3497.95 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-01-01 09:45:00 | 3527.15 | 2024-01-01 12:15:00 | 3503.90 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-01-01 14:30:00 | 3525.75 | 2024-01-02 09:15:00 | 3483.55 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2024-01-09 09:15:00 | 3544.00 | 2024-01-11 13:15:00 | 3509.50 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-01-16 10:45:00 | 3585.15 | 2024-01-23 12:15:00 | 3589.50 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2024-01-16 11:45:00 | 3567.20 | 2024-01-23 12:15:00 | 3589.50 | STOP_HIT | 1.00 | 0.63% |
| BUY | retest2 | 2024-01-16 12:15:00 | 3571.00 | 2024-01-23 12:15:00 | 3589.50 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2024-01-16 13:15:00 | 3569.65 | 2024-01-23 12:15:00 | 3589.50 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2024-01-19 09:15:00 | 3628.50 | 2024-01-23 12:15:00 | 3589.50 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-01-23 11:15:00 | 3618.40 | 2024-01-23 12:15:00 | 3589.50 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2024-01-24 12:15:00 | 3572.90 | 2024-01-29 09:15:00 | 3669.70 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2024-01-24 13:30:00 | 3572.55 | 2024-01-29 09:15:00 | 3669.70 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2024-01-25 14:30:00 | 3569.30 | 2024-01-29 09:15:00 | 3669.70 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2024-02-13 15:15:00 | 3316.25 | 2024-02-16 09:15:00 | 3353.50 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2024-02-20 15:00:00 | 3358.50 | 2024-02-21 09:15:00 | 3339.25 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-03-01 09:15:00 | 3529.00 | 2024-03-12 14:15:00 | 3619.80 | STOP_HIT | 1.00 | 2.57% |
| SELL | retest2 | 2024-03-15 09:30:00 | 3565.65 | 2024-03-21 10:15:00 | 3587.15 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-03-21 10:15:00 | 3575.85 | 2024-03-21 10:15:00 | 3587.15 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2024-04-04 09:15:00 | 3790.25 | 2024-04-04 09:15:00 | 3745.40 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-04-18 13:15:00 | 3553.20 | 2024-04-22 13:15:00 | 3593.05 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2024-04-22 11:15:00 | 3584.10 | 2024-04-22 13:15:00 | 3593.05 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2024-04-22 13:15:00 | 3585.05 | 2024-04-22 13:15:00 | 3593.05 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2024-05-03 10:30:00 | 3549.85 | 2024-05-08 09:15:00 | 3372.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-03 10:30:00 | 3549.85 | 2024-05-08 11:15:00 | 3459.40 | STOP_HIT | 0.50 | 2.55% |
| BUY | retest2 | 2024-05-22 09:15:00 | 3452.65 | 2024-06-03 09:15:00 | 3797.92 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-05-22 09:45:00 | 3456.80 | 2024-06-03 09:15:00 | 3802.48 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-06 12:00:00 | 3476.70 | 2024-06-07 13:15:00 | 3529.30 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-06-06 13:45:00 | 3485.30 | 2024-06-07 13:15:00 | 3529.30 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest1 | 2024-06-11 09:15:00 | 3585.05 | 2024-06-19 09:15:00 | 3629.95 | STOP_HIT | 1.00 | 1.25% |
| BUY | retest2 | 2024-07-11 09:15:00 | 3668.30 | 2024-07-11 12:15:00 | 3622.05 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-08-09 12:00:00 | 3581.00 | 2024-08-20 11:15:00 | 3570.00 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2024-08-12 09:15:00 | 3577.30 | 2024-08-20 11:15:00 | 3570.00 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2024-08-12 11:45:00 | 3582.75 | 2024-08-20 11:15:00 | 3570.00 | STOP_HIT | 1.00 | 0.36% |
| SELL | retest2 | 2024-08-12 12:45:00 | 3581.50 | 2024-08-20 11:15:00 | 3570.00 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2024-08-13 12:30:00 | 3580.90 | 2024-08-20 11:15:00 | 3570.00 | STOP_HIT | 1.00 | 0.30% |
| BUY | retest2 | 2024-08-26 11:30:00 | 3621.15 | 2024-09-02 14:15:00 | 3683.10 | STOP_HIT | 1.00 | 1.71% |
| SELL | retest2 | 2024-09-11 13:45:00 | 3567.65 | 2024-09-12 14:15:00 | 3631.70 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2024-09-12 09:45:00 | 3546.45 | 2024-09-12 14:15:00 | 3631.70 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2024-09-19 15:15:00 | 3689.00 | 2024-09-26 12:15:00 | 3747.95 | STOP_HIT | 1.00 | 1.60% |
| SELL | retest2 | 2024-10-01 11:15:00 | 3675.00 | 2024-10-03 15:15:00 | 3491.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 12:00:00 | 3674.10 | 2024-10-03 15:15:00 | 3490.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 11:15:00 | 3675.00 | 2024-10-04 11:15:00 | 3553.45 | STOP_HIT | 0.50 | 3.31% |
| SELL | retest2 | 2024-10-01 12:00:00 | 3674.10 | 2024-10-04 11:15:00 | 3553.45 | STOP_HIT | 0.50 | 3.28% |
| BUY | retest2 | 2024-10-17 09:15:00 | 3559.00 | 2024-10-22 11:15:00 | 3540.05 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-10-22 10:30:00 | 3531.30 | 2024-10-22 11:15:00 | 3540.05 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest1 | 2024-10-25 09:15:00 | 3406.00 | 2024-10-29 14:15:00 | 3382.00 | STOP_HIT | 1.00 | 0.70% |
| BUY | retest2 | 2024-11-05 12:45:00 | 3557.75 | 2024-11-12 09:15:00 | 3617.75 | STOP_HIT | 1.00 | 1.69% |
| BUY | retest2 | 2024-11-05 13:15:00 | 3562.00 | 2024-11-12 09:15:00 | 3617.75 | STOP_HIT | 1.00 | 1.57% |
| BUY | retest2 | 2024-11-05 13:45:00 | 3559.25 | 2024-11-12 09:15:00 | 3617.75 | STOP_HIT | 1.00 | 1.64% |
| BUY | retest2 | 2024-12-03 10:00:00 | 3718.75 | 2024-12-12 10:15:00 | 3873.05 | STOP_HIT | 1.00 | 4.15% |
| SELL | retest2 | 2024-12-16 12:00:00 | 3859.90 | 2024-12-20 09:15:00 | 3666.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 09:15:00 | 3856.05 | 2024-12-20 12:15:00 | 3663.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 12:00:00 | 3859.90 | 2024-12-24 09:15:00 | 3665.00 | STOP_HIT | 0.50 | 5.05% |
| SELL | retest2 | 2024-12-17 09:15:00 | 3856.05 | 2024-12-24 09:15:00 | 3665.00 | STOP_HIT | 0.50 | 4.95% |
| BUY | retest2 | 2025-01-17 09:15:00 | 3558.45 | 2025-01-22 10:15:00 | 3532.20 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-01-24 14:15:00 | 3466.85 | 2025-01-30 09:15:00 | 3505.75 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2025-01-27 10:15:00 | 3449.30 | 2025-01-30 09:15:00 | 3505.75 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-01-28 09:15:00 | 3436.45 | 2025-01-30 09:15:00 | 3505.75 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-02-11 12:00:00 | 3297.35 | 2025-02-19 11:15:00 | 3289.80 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2025-03-12 10:15:00 | 3166.00 | 2025-03-13 10:15:00 | 3228.00 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2025-03-27 09:15:00 | 3484.90 | 2025-04-01 11:15:00 | 3441.50 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-04-03 13:15:00 | 3440.00 | 2025-04-04 11:15:00 | 3268.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 13:15:00 | 3440.00 | 2025-04-07 09:15:00 | 3096.00 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-04-17 12:00:00 | 3193.70 | 2025-04-25 11:15:00 | 3250.00 | STOP_HIT | 1.00 | 1.76% |
| BUY | retest2 | 2025-04-30 09:15:00 | 3334.90 | 2025-05-05 09:15:00 | 3313.40 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-05-05 11:00:00 | 3332.90 | 2025-05-05 12:15:00 | 3315.70 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-05-05 12:00:00 | 3331.70 | 2025-05-05 12:15:00 | 3315.70 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest1 | 2025-05-14 09:15:00 | 3587.70 | 2025-05-19 12:15:00 | 3592.20 | STOP_HIT | 1.00 | 0.13% |
| BUY | retest1 | 2025-05-14 15:00:00 | 3578.50 | 2025-05-19 12:15:00 | 3592.20 | STOP_HIT | 1.00 | 0.38% |
| BUY | retest2 | 2025-05-15 13:00:00 | 3600.80 | 2025-05-20 13:15:00 | 3579.40 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-05-19 14:45:00 | 3593.30 | 2025-05-20 13:15:00 | 3579.40 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-05-27 11:15:00 | 3655.00 | 2025-06-03 11:15:00 | 3644.80 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-05-28 14:00:00 | 3655.20 | 2025-06-03 11:15:00 | 3644.80 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-05-29 09:15:00 | 3656.00 | 2025-06-03 11:15:00 | 3644.80 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-05-29 15:00:00 | 3657.20 | 2025-06-03 11:15:00 | 3644.80 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-06-02 13:00:00 | 3684.50 | 2025-06-03 11:15:00 | 3644.80 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-06-02 14:45:00 | 3684.60 | 2025-06-03 11:15:00 | 3644.80 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2025-06-11 14:15:00 | 3690.80 | 2025-06-12 10:15:00 | 3655.10 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-06-12 09:15:00 | 3701.00 | 2025-06-12 10:15:00 | 3655.10 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-07-09 09:15:00 | 3579.50 | 2025-07-21 13:15:00 | 3495.60 | STOP_HIT | 1.00 | 2.34% |
| SELL | retest2 | 2025-07-23 10:45:00 | 3463.70 | 2025-07-29 13:15:00 | 3486.70 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-07-23 12:45:00 | 3463.90 | 2025-07-29 13:15:00 | 3486.70 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-07-24 12:45:00 | 3465.10 | 2025-07-29 13:15:00 | 3486.70 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-07-25 09:30:00 | 3465.20 | 2025-07-29 13:15:00 | 3486.70 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-08-04 11:15:00 | 3630.50 | 2025-08-07 10:15:00 | 3609.80 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-08-04 13:45:00 | 3628.50 | 2025-08-07 10:15:00 | 3609.80 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-08-05 11:45:00 | 3627.00 | 2025-08-07 10:15:00 | 3609.80 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-08-06 15:00:00 | 3629.70 | 2025-08-07 10:15:00 | 3609.80 | STOP_HIT | 1.00 | -0.55% |
| BUY | retest1 | 2025-08-13 09:15:00 | 3697.10 | 2025-08-14 09:15:00 | 3671.30 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-08-21 11:45:00 | 3627.50 | 2025-08-29 11:15:00 | 3613.00 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2025-09-10 10:30:00 | 3547.90 | 2025-09-12 09:15:00 | 3582.80 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-09-10 14:00:00 | 3548.60 | 2025-09-12 09:15:00 | 3582.80 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-09-11 10:00:00 | 3548.80 | 2025-09-12 09:15:00 | 3582.80 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-09-24 09:15:00 | 3649.00 | 2025-09-24 09:15:00 | 3679.60 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-10-09 10:15:00 | 3763.10 | 2025-10-14 11:15:00 | 3755.40 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2025-10-09 13:15:00 | 3766.00 | 2025-10-14 11:15:00 | 3755.40 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-10-13 10:15:00 | 3767.80 | 2025-10-14 11:15:00 | 3755.40 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-10-13 12:30:00 | 3764.00 | 2025-10-14 11:15:00 | 3755.40 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-10-27 09:15:00 | 3924.20 | 2025-11-04 09:15:00 | 3955.50 | STOP_HIT | 1.00 | 0.80% |
| BUY | retest2 | 2025-11-19 11:30:00 | 4014.00 | 2025-11-24 13:15:00 | 4014.00 | STOP_HIT | 1.00 | 0.00% |
| BUY | retest2 | 2025-11-19 12:00:00 | 4012.00 | 2025-11-24 13:15:00 | 4014.00 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-11-19 12:45:00 | 4011.10 | 2025-11-24 13:15:00 | 4014.00 | STOP_HIT | 1.00 | 0.07% |
| SELL | retest2 | 2025-12-09 14:15:00 | 3988.80 | 2025-12-10 11:15:00 | 4020.50 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-12-10 09:45:00 | 3991.80 | 2025-12-10 11:15:00 | 4020.50 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-12-18 14:15:00 | 4031.20 | 2025-12-19 09:15:00 | 4069.80 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-12-29 13:30:00 | 4037.00 | 2025-12-30 12:15:00 | 4058.00 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-12-30 10:30:00 | 4039.90 | 2025-12-30 12:15:00 | 4058.00 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2026-01-06 10:00:00 | 4182.00 | 2026-01-07 09:15:00 | 4130.00 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2026-01-06 11:30:00 | 4165.00 | 2026-01-07 09:15:00 | 4130.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2026-01-13 12:15:00 | 3897.00 | 2026-01-27 15:15:00 | 3804.50 | STOP_HIT | 1.00 | 2.37% |
| BUY | retest2 | 2026-02-09 09:15:00 | 4095.50 | 2026-02-24 13:15:00 | 4259.80 | STOP_HIT | 1.00 | 4.01% |
| SELL | retest2 | 2026-03-06 09:15:00 | 3960.60 | 2026-03-09 09:15:00 | 3762.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 09:15:00 | 3960.60 | 2026-03-10 09:15:00 | 3863.80 | STOP_HIT | 0.50 | 2.44% |
| BUY | retest2 | 2026-04-13 12:15:00 | 3946.10 | 2026-04-20 13:15:00 | 4034.60 | STOP_HIT | 1.00 | 2.24% |
| BUY | retest2 | 2026-04-13 13:00:00 | 3946.40 | 2026-04-20 13:15:00 | 4034.60 | STOP_HIT | 1.00 | 2.23% |
| SELL | retest2 | 2026-04-24 09:45:00 | 4018.70 | 2026-04-27 09:15:00 | 4065.70 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest2 | 2026-04-24 10:30:00 | 4018.80 | 2026-04-27 09:15:00 | 4065.70 | STOP_HIT | 1.00 | -1.17% |
