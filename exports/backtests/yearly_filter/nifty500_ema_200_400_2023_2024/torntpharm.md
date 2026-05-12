# Torrent Pharmaceuticals Ltd. (TORNTPHARM)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 4385.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 12 |
| ALERT1 | 9 |
| ALERT2 | 9 |
| ALERT2_SKIP | 7 |
| ALERT3 | 55 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 35 |
| PARTIAL | 2 |
| TARGET_HIT | 9 |
| STOP_HIT | 26 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 37 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 12 / 25
- **Target hits / Stop hits / Partials:** 9 / 26 / 2
- **Avg / median % per leg:** 1.58% / -0.88%
- **Sum % (uncompounded):** 58.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 8 | 38.1% | 8 | 13 | 0 | 2.55% | 53.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 8 | 38.1% | 8 | 13 | 0 | 2.55% | 53.6% |
| SELL (all) | 16 | 4 | 25.0% | 1 | 13 | 2 | 0.31% | 5.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 4 | 25.0% | 1 | 13 | 2 | 0.31% | 5.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 37 | 12 | 32.4% | 9 | 26 | 2 | 1.58% | 58.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-09-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 11:15:00 | 1852.85 | 1886.62 | 1886.74 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2023-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 13:15:00 | 1930.80 | 1887.13 | 1886.98 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-10-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 14:15:00 | 1864.00 | 1886.88 | 1886.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-05 09:15:00 | 1849.45 | 1886.25 | 1886.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-06 12:15:00 | 1885.75 | 1883.97 | 1885.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-06 12:15:00 | 1885.75 | 1883.97 | 1885.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 12:15:00 | 1885.75 | 1883.97 | 1885.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 12:30:00 | 1889.40 | 1883.97 | 1885.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 13:15:00 | 1889.65 | 1884.03 | 1885.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 14:00:00 | 1889.65 | 1884.03 | 1885.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-06 14:15:00 | 1891.45 | 1884.10 | 1885.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-06 15:00:00 | 1891.45 | 1884.10 | 1885.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-09 10:15:00 | 1883.45 | 1884.26 | 1885.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-09 11:45:00 | 1879.70 | 1884.43 | 1885.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-09 12:15:00 | 1900.90 | 1884.60 | 1885.66 | SL hit (close>static) qty=1.00 sl=1893.15 alert=retest2 |

### Cycle 4 — BUY (started 2023-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-16 13:15:00 | 1910.95 | 1886.60 | 1886.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-17 10:15:00 | 1928.60 | 1887.74 | 1887.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-20 14:15:00 | 1890.45 | 1894.27 | 1890.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-20 14:15:00 | 1890.45 | 1894.27 | 1890.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 14:15:00 | 1890.45 | 1894.27 | 1890.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-20 15:00:00 | 1890.45 | 1894.27 | 1890.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 15:15:00 | 1898.40 | 1894.31 | 1890.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-23 09:15:00 | 1908.15 | 1894.31 | 1890.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 09:15:00 | 1900.60 | 1894.38 | 1890.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-25 10:00:00 | 1913.75 | 1894.14 | 1890.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-26 11:15:00 | 1864.75 | 1895.71 | 1891.71 | SL hit (close<static) qty=1.00 sl=1875.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 12:15:00 | 3172.70 | 3307.54 | 3307.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 13:15:00 | 3156.80 | 3306.04 | 3306.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 14:15:00 | 3267.65 | 3233.56 | 3264.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 14:15:00 | 3267.65 | 3233.56 | 3264.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 3267.65 | 3233.56 | 3264.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 15:00:00 | 3267.65 | 3233.56 | 3264.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 3272.50 | 3233.95 | 3265.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 09:30:00 | 3244.45 | 3233.85 | 3264.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 14:45:00 | 3254.35 | 3234.47 | 3264.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 10:15:00 | 3300.00 | 3230.40 | 3259.78 | SL hit (close>static) qty=1.00 sl=3276.00 alert=retest2 |

### Cycle 6 — BUY (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 11:15:00 | 3384.75 | 3281.73 | 3281.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-18 10:15:00 | 3397.15 | 3302.58 | 3292.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 14:15:00 | 3354.25 | 3361.20 | 3332.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 15:00:00 | 3354.25 | 3361.20 | 3332.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 12:15:00 | 3336.80 | 3364.72 | 3336.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 13:00:00 | 3336.80 | 3364.72 | 3336.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 13:15:00 | 3332.90 | 3364.40 | 3336.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 13:30:00 | 3330.20 | 3364.40 | 3336.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 3329.45 | 3364.05 | 3336.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-08 15:00:00 | 3329.45 | 3364.05 | 3336.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 3322.00 | 3363.63 | 3336.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:15:00 | 3303.50 | 3363.63 | 3336.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 3321.85 | 3359.82 | 3335.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:45:00 | 3316.15 | 3359.82 | 3335.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-01-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 13:15:00 | 3180.00 | 3315.41 | 3315.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 14:15:00 | 3172.10 | 3313.98 | 3314.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-27 09:15:00 | 3333.00 | 3278.88 | 3295.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 09:15:00 | 3333.00 | 3278.88 | 3295.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 3333.00 | 3278.88 | 3295.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 09:45:00 | 3355.40 | 3278.88 | 3295.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 10:15:00 | 3341.95 | 3279.50 | 3295.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-27 11:00:00 | 3341.95 | 3279.50 | 3295.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 10:15:00 | 3304.80 | 3282.57 | 3296.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 11:00:00 | 3304.80 | 3282.57 | 3296.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 11:15:00 | 3307.60 | 3282.82 | 3296.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 12:00:00 | 3307.60 | 3282.82 | 3296.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 3309.10 | 3283.08 | 3296.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 3309.10 | 3283.08 | 3296.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 3330.05 | 3283.55 | 3296.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:00:00 | 3330.05 | 3283.55 | 3296.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 3301.00 | 3292.87 | 3300.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 10:00:00 | 3301.00 | 3292.87 | 3300.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 3269.00 | 3292.63 | 3300.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-01 11:30:00 | 3262.50 | 3292.22 | 3299.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-04 10:15:00 | 3099.38 | 3277.51 | 3291.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-07 11:15:00 | 3255.00 | 3252.80 | 3277.19 | SL hit (close>ema200) qty=0.50 sl=3252.80 alert=retest2 |

### Cycle 8 — BUY (started 2025-04-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-17 09:15:00 | 3214.40 | 3176.12 | 3175.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 14:15:00 | 3257.60 | 3178.70 | 3177.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 10:15:00 | 3194.50 | 3206.00 | 3192.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 10:15:00 | 3194.50 | 3206.00 | 3192.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 3194.50 | 3206.00 | 3192.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 3194.50 | 3206.00 | 3192.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 3231.80 | 3206.26 | 3192.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:30:00 | 3186.40 | 3206.26 | 3192.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 3261.40 | 3238.55 | 3213.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:00:00 | 3281.20 | 3238.97 | 3213.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 09:15:00 | 3156.30 | 3239.21 | 3215.51 | SL hit (close<static) qty=1.00 sl=3164.70 alert=retest2 |

### Cycle 9 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 3140.50 | 3208.67 | 3208.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 15:15:00 | 3126.50 | 3204.26 | 3206.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 10:15:00 | 3212.70 | 3190.58 | 3198.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 10:15:00 | 3212.70 | 3190.58 | 3198.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 3212.70 | 3190.58 | 3198.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:00:00 | 3212.70 | 3190.58 | 3198.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 3220.90 | 3190.88 | 3199.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:00:00 | 3220.90 | 3190.88 | 3199.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 3204.30 | 3192.54 | 3199.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:30:00 | 3200.80 | 3192.54 | 3199.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 3194.90 | 3192.56 | 3199.61 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 12:15:00 | 3254.40 | 3205.80 | 3205.66 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 11:15:00 | 3169.60 | 3205.62 | 3205.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 3157.00 | 3205.14 | 3205.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 10:15:00 | 3214.70 | 3199.74 | 3202.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 10:15:00 | 3214.70 | 3199.74 | 3202.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 3214.70 | 3199.74 | 3202.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 11:00:00 | 3214.70 | 3199.74 | 3202.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 3211.20 | 3199.85 | 3202.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 12:15:00 | 3209.40 | 3199.85 | 3202.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 12:45:00 | 3210.00 | 3200.04 | 3202.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 13:15:00 | 3229.30 | 3200.33 | 3202.79 | SL hit (close>static) qty=1.00 sl=3225.00 alert=retest2 |

### Cycle 12 — BUY (started 2025-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 12:15:00 | 3293.40 | 3205.21 | 3205.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 13:15:00 | 3326.70 | 3206.42 | 3205.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-28 09:15:00 | 3578.20 | 3583.11 | 3489.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-28 09:30:00 | 3586.80 | 3583.11 | 3489.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 3516.50 | 3587.97 | 3524.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:00:00 | 3516.50 | 3587.97 | 3524.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 3522.20 | 3587.32 | 3524.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 11:30:00 | 3516.60 | 3587.32 | 3524.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 12:15:00 | 3543.00 | 3586.88 | 3524.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 13:15:00 | 3545.10 | 3586.88 | 3524.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 12:15:00 | 3548.00 | 3583.85 | 3524.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 13:00:00 | 3552.70 | 3583.54 | 3525.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-17 10:15:00 | 3519.20 | 3581.05 | 3525.29 | SL hit (close<static) qty=1.00 sl=3519.90 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2023-10-09 11:45:00 | 1879.70 | 2023-10-09 12:15:00 | 1900.90 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2023-10-10 09:15:00 | 1882.65 | 2023-10-11 09:15:00 | 1901.95 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2023-10-10 12:00:00 | 1882.15 | 2023-10-11 09:15:00 | 1901.95 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2023-10-11 15:15:00 | 1873.10 | 2023-10-13 09:15:00 | 1902.15 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2023-10-25 10:00:00 | 1913.75 | 2023-10-26 11:15:00 | 1864.75 | STOP_HIT | 1.00 | -2.56% |
| BUY | retest2 | 2023-10-27 09:15:00 | 1930.85 | 2023-11-21 09:15:00 | 2123.93 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-26 09:30:00 | 3244.45 | 2024-11-29 10:15:00 | 3300.00 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-11-26 14:45:00 | 3254.35 | 2024-11-29 10:15:00 | 3300.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-02-01 11:30:00 | 3262.50 | 2025-02-04 10:15:00 | 3099.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-01 11:30:00 | 3262.50 | 2025-02-07 11:15:00 | 3255.00 | STOP_HIT | 0.50 | 0.23% |
| SELL | retest2 | 2025-02-07 13:45:00 | 3258.60 | 2025-02-11 12:15:00 | 3095.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 13:45:00 | 3258.60 | 2025-02-28 09:15:00 | 2932.74 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-21 12:15:00 | 3264.15 | 2025-04-04 15:15:00 | 3309.65 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-03-21 13:00:00 | 3248.60 | 2025-04-04 15:15:00 | 3309.65 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-04-11 13:45:00 | 3134.75 | 2025-04-15 09:15:00 | 3193.40 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-05-07 11:00:00 | 3281.20 | 2025-05-09 09:15:00 | 3156.30 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest2 | 2025-05-19 09:45:00 | 3291.90 | 2025-05-26 09:15:00 | 3157.20 | STOP_HIT | 1.00 | -4.09% |
| BUY | retest2 | 2025-05-21 09:15:00 | 3357.90 | 2025-05-26 09:15:00 | 3157.20 | STOP_HIT | 1.00 | -5.98% |
| BUY | retest2 | 2025-05-21 12:15:00 | 3288.70 | 2025-05-26 09:15:00 | 3157.20 | STOP_HIT | 1.00 | -4.00% |
| SELL | retest2 | 2025-06-25 12:15:00 | 3209.40 | 2025-06-25 13:15:00 | 3229.30 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-06-25 12:45:00 | 3210.00 | 2025-06-25 13:15:00 | 3229.30 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-06-26 09:15:00 | 3194.80 | 2025-06-26 14:15:00 | 3225.80 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2025-09-15 13:15:00 | 3545.10 | 2025-09-17 10:15:00 | 3519.20 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-09-16 12:15:00 | 3548.00 | 2025-09-17 10:15:00 | 3519.20 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-09-16 13:00:00 | 3552.70 | 2025-09-17 10:15:00 | 3519.20 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-09-17 12:30:00 | 3544.70 | 2025-09-26 10:15:00 | 3518.50 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-09-26 12:15:00 | 3554.10 | 2025-10-13 14:15:00 | 3518.80 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-10-03 09:45:00 | 3550.00 | 2025-10-13 14:15:00 | 3518.80 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-10-03 10:15:00 | 3552.50 | 2025-10-13 14:15:00 | 3518.80 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-10-08 12:15:00 | 3551.20 | 2025-10-16 13:15:00 | 3522.20 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-10-10 11:15:00 | 3547.50 | 2026-01-06 12:15:00 | 3909.51 | TARGET_HIT | 1.00 | 10.20% |
| BUY | retest2 | 2025-10-10 15:15:00 | 3545.70 | 2026-01-06 12:15:00 | 3905.00 | TARGET_HIT | 1.00 | 10.13% |
| BUY | retest2 | 2025-10-13 09:30:00 | 3542.40 | 2026-01-06 12:15:00 | 3907.75 | TARGET_HIT | 1.00 | 10.31% |
| BUY | retest2 | 2025-10-15 10:15:00 | 3548.90 | 2026-01-06 12:15:00 | 3906.32 | TARGET_HIT | 1.00 | 10.07% |
| BUY | retest2 | 2025-10-16 10:45:00 | 3543.20 | 2026-01-06 12:15:00 | 3903.79 | TARGET_HIT | 1.00 | 10.18% |
| BUY | retest2 | 2025-10-17 10:45:00 | 3556.10 | 2026-01-06 12:15:00 | 3911.71 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-06 10:45:00 | 3547.90 | 2026-01-06 12:15:00 | 3902.69 | TARGET_HIT | 1.00 | 10.00% |
