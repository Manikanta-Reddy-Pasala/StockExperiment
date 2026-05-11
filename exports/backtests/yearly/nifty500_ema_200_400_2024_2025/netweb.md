# Netweb Technologies India Ltd. (NETWEB)

## Backtest Summary

- **Window:** 2023-07-27 09:15:00 → 2026-05-08 15:15:00 (4792 bars)
- **Last close:** 4424.00
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
| ALERT2 | 6 |
| ALERT2_SKIP | 4 |
| ALERT3 | 30 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 40 |
| PARTIAL | 6 |
| TARGET_HIT | 5 |
| STOP_HIT | 35 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 46 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 29
- **Target hits / Stop hits / Partials:** 5 / 35 / 6
- **Avg / median % per leg:** -0.12% / -2.01%
- **Sum % (uncompounded):** -5.33%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 5 | 25.0% | 5 | 15 | 0 | 0.06% | 1.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 20 | 5 | 25.0% | 5 | 15 | 0 | 0.06% | 1.1% |
| SELL (all) | 26 | 12 | 46.2% | 0 | 20 | 6 | -0.25% | -6.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 26 | 12 | 46.2% | 0 | 20 | 6 | -0.25% | -6.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 46 | 17 | 37.0% | 5 | 35 | 6 | -0.12% | -5.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-01-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-15 15:15:00 | 2442.00 | 2725.72 | 2726.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 09:15:00 | 2373.70 | 2704.37 | 2715.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 14:15:00 | 1609.80 | 1588.89 | 1810.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-21 15:00:00 | 1609.80 | 1588.89 | 1810.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 1675.90 | 1505.99 | 1626.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 10:00:00 | 1675.90 | 1505.99 | 1626.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 1625.40 | 1507.18 | 1626.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 11:30:00 | 1617.00 | 1508.23 | 1626.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 09:30:00 | 1606.00 | 1513.21 | 1625.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 1536.15 | 1516.43 | 1623.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-07 09:15:00 | 1525.70 | 1516.43 | 1623.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-07 10:15:00 | 1524.80 | 1516.51 | 1623.05 | SL hit (close>ema200) qty=0.50 sl=1516.51 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 14:15:00 | 1894.20 | 1679.48 | 1679.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 09:15:00 | 1935.90 | 1684.28 | 1681.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 1841.50 | 1843.83 | 1779.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 09:45:00 | 1843.50 | 1843.83 | 1779.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 1793.90 | 1842.02 | 1780.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 14:15:00 | 1800.90 | 1834.35 | 1780.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-17 09:30:00 | 1799.00 | 1833.76 | 1780.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 13:15:00 | 1749.10 | 1827.34 | 1780.31 | SL hit (close<static) qty=1.00 sl=1750.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 14:15:00 | 3259.90 | 3283.65 | 3283.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 11:15:00 | 3206.80 | 3277.53 | 3280.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-05 10:15:00 | 3252.80 | 3211.58 | 3243.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 10:15:00 | 3252.80 | 3211.58 | 3243.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 3252.80 | 3211.58 | 3243.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:00:00 | 3252.80 | 3211.58 | 3243.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 3347.60 | 3212.94 | 3243.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:00:00 | 3347.60 | 3212.94 | 3243.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 3371.00 | 3219.05 | 3246.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 12:15:00 | 3358.60 | 3220.48 | 3246.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 13:15:00 | 3356.00 | 3221.89 | 3247.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 13:45:00 | 3354.90 | 3223.13 | 3247.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-06 15:15:00 | 3330.00 | 3224.52 | 3248.38 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 3278.70 | 3263.09 | 3265.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:15:00 | 3325.50 | 3263.09 | 3265.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 3287.00 | 3263.33 | 3265.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:15:00 | 3272.70 | 3263.33 | 3265.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 11:30:00 | 3283.50 | 3263.55 | 3265.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 15:15:00 | 3190.67 | 3261.47 | 3264.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 15:15:00 | 3188.20 | 3261.47 | 3264.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 15:15:00 | 3187.15 | 3261.47 | 3264.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-14 15:15:00 | 3163.50 | 3261.47 | 3264.76 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 3306.00 | 3261.91 | 3264.96 | SL hit (close>ema200) qty=0.50 sl=3261.91 alert=retest2 |

### Cycle 4 — BUY (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 10:15:00 | 3494.10 | 3269.87 | 3268.86 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2026-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 09:15:00 | 3133.00 | 3267.58 | 3268.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 3109.40 | 3253.51 | 3260.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 09:15:00 | 3217.10 | 3214.38 | 3238.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 3217.10 | 3214.38 | 3238.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 3217.10 | 3214.38 | 3238.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 12:15:00 | 3153.00 | 3219.67 | 3240.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 13:00:00 | 3152.50 | 3219.01 | 3239.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-05 09:15:00 | 3155.00 | 3218.34 | 3237.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 14:45:00 | 3148.90 | 3200.48 | 3224.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 3146.90 | 3176.68 | 3208.70 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-18 10:15:00 | 3289.10 | 3177.80 | 3209.10 | SL hit (close>static) qty=1.00 sl=3283.50 alert=retest2 |

### Cycle 6 — BUY (started 2026-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 14:15:00 | 3660.30 | 3237.74 | 3237.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 3681.90 | 3285.75 | 3262.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-05 09:15:00 | 3373.20 | 3402.70 | 3330.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 3373.20 | 3402.70 | 3330.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 3373.20 | 3402.70 | 3330.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:30:00 | 3342.90 | 3402.70 | 3330.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 3345.00 | 3402.12 | 3330.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 10:30:00 | 3338.10 | 3402.12 | 3330.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 11:15:00 | 3313.10 | 3401.24 | 3330.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:00:00 | 3313.10 | 3401.24 | 3330.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 3302.60 | 3400.25 | 3330.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:00:00 | 3302.60 | 3400.25 | 3330.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 3282.00 | 3392.91 | 3328.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:45:00 | 3270.00 | 3392.91 | 3328.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 3292.00 | 3368.58 | 3320.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 12:00:00 | 3292.00 | 3368.58 | 3320.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 3290.30 | 3367.80 | 3320.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 15:00:00 | 3319.00 | 3366.67 | 3319.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 10:15:00 | 3258.40 | 3364.78 | 3319.66 | SL hit (close<static) qty=1.00 sl=3281.00 alert=retest2 |

### Cycle 7 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 3176.60 | 3289.30 | 3289.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 3135.80 | 3282.16 | 3285.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 3341.30 | 3252.59 | 3269.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 3341.30 | 3252.59 | 3269.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 3341.30 | 3252.59 | 3269.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 14:45:00 | 3310.90 | 3257.03 | 3270.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:45:00 | 3292.80 | 3258.25 | 3271.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 3386.20 | 3265.03 | 3274.32 | SL hit (close>static) qty=1.00 sl=3374.00 alert=retest2 |

### Cycle 8 — BUY (started 2026-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 12:15:00 | 3562.90 | 3284.36 | 3283.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 3694.00 | 3296.52 | 3289.70 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-05 11:30:00 | 1617.00 | 2025-05-07 09:15:00 | 1536.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-06 09:30:00 | 1606.00 | 2025-05-07 09:15:00 | 1525.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-05 11:30:00 | 1617.00 | 2025-05-07 10:15:00 | 1524.80 | STOP_HIT | 0.50 | 5.70% |
| SELL | retest2 | 2025-05-06 09:30:00 | 1606.00 | 2025-05-07 10:15:00 | 1524.80 | STOP_HIT | 0.50 | 5.06% |
| SELL | retest2 | 2025-05-08 10:00:00 | 1607.20 | 2025-05-08 12:15:00 | 1688.00 | STOP_HIT | 1.00 | -5.03% |
| SELL | retest2 | 2025-05-08 15:15:00 | 1612.00 | 2025-05-12 09:15:00 | 1725.00 | STOP_HIT | 1.00 | -7.01% |
| BUY | retest2 | 2025-06-16 14:15:00 | 1800.90 | 2025-06-18 13:15:00 | 1749.10 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2025-06-17 09:30:00 | 1799.00 | 2025-06-18 13:15:00 | 1749.10 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2025-06-24 09:45:00 | 1796.70 | 2025-07-08 11:15:00 | 1783.60 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-06-25 09:15:00 | 1836.20 | 2025-07-11 14:15:00 | 1976.37 | TARGET_HIT | 1.00 | 7.63% |
| BUY | retest2 | 2025-07-08 09:15:00 | 1808.30 | 2025-07-11 14:15:00 | 1981.87 | TARGET_HIT | 1.00 | 9.60% |
| BUY | retest2 | 2025-07-08 14:30:00 | 1801.70 | 2025-07-11 14:15:00 | 1976.92 | TARGET_HIT | 1.00 | 9.73% |
| BUY | retest2 | 2025-07-09 13:30:00 | 1797.20 | 2025-07-11 14:15:00 | 1985.94 | TARGET_HIT | 1.00 | 10.50% |
| BUY | retest2 | 2025-07-09 14:30:00 | 1805.40 | 2025-07-30 13:15:00 | 2019.82 | TARGET_HIT | 1.00 | 11.88% |
| BUY | retest2 | 2025-11-07 09:45:00 | 3409.50 | 2025-11-18 09:15:00 | 3287.00 | STOP_HIT | 1.00 | -3.59% |
| BUY | retest2 | 2025-11-07 11:15:00 | 3408.90 | 2025-11-18 09:15:00 | 3287.00 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2025-11-07 15:15:00 | 3440.00 | 2025-11-18 09:15:00 | 3287.00 | STOP_HIT | 1.00 | -4.45% |
| BUY | retest2 | 2025-11-10 15:15:00 | 3408.00 | 2025-11-18 09:15:00 | 3287.00 | STOP_HIT | 1.00 | -3.55% |
| BUY | retest2 | 2025-11-17 11:45:00 | 3392.00 | 2025-11-18 09:15:00 | 3287.00 | STOP_HIT | 1.00 | -3.10% |
| BUY | retest2 | 2025-11-17 14:45:00 | 3393.20 | 2025-11-18 09:15:00 | 3287.00 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2025-11-20 09:15:00 | 3428.50 | 2025-11-21 09:15:00 | 3326.90 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2026-01-06 12:15:00 | 3358.60 | 2026-01-14 15:15:00 | 3190.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 13:15:00 | 3356.00 | 2026-01-14 15:15:00 | 3188.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 13:45:00 | 3354.90 | 2026-01-14 15:15:00 | 3187.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 15:15:00 | 3330.00 | 2026-01-14 15:15:00 | 3163.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-06 12:15:00 | 3358.60 | 2026-01-16 09:15:00 | 3306.00 | STOP_HIT | 0.50 | 1.57% |
| SELL | retest2 | 2026-01-06 13:15:00 | 3356.00 | 2026-01-16 09:15:00 | 3306.00 | STOP_HIT | 0.50 | 1.49% |
| SELL | retest2 | 2026-01-06 13:45:00 | 3354.90 | 2026-01-16 09:15:00 | 3306.00 | STOP_HIT | 0.50 | 1.46% |
| SELL | retest2 | 2026-01-06 15:15:00 | 3330.00 | 2026-01-16 09:15:00 | 3306.00 | STOP_HIT | 0.50 | 0.72% |
| SELL | retest2 | 2026-01-14 10:15:00 | 3272.70 | 2026-01-16 14:15:00 | 3348.10 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2026-01-14 11:30:00 | 3283.50 | 2026-01-16 14:15:00 | 3348.10 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-01-16 09:30:00 | 3283.40 | 2026-01-16 14:15:00 | 3348.10 | STOP_HIT | 1.00 | -1.97% |
| SELL | retest2 | 2026-01-16 11:30:00 | 3282.00 | 2026-01-16 14:15:00 | 3348.10 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2026-02-02 12:15:00 | 3153.00 | 2026-02-18 10:15:00 | 3289.10 | STOP_HIT | 1.00 | -4.32% |
| SELL | retest2 | 2026-02-02 13:00:00 | 3152.50 | 2026-02-18 10:15:00 | 3289.10 | STOP_HIT | 1.00 | -4.33% |
| SELL | retest2 | 2026-02-05 09:15:00 | 3155.00 | 2026-02-18 10:15:00 | 3289.10 | STOP_HIT | 1.00 | -4.25% |
| SELL | retest2 | 2026-02-11 14:45:00 | 3148.90 | 2026-02-18 10:15:00 | 3289.10 | STOP_HIT | 1.00 | -4.45% |
| BUY | retest2 | 2026-03-10 15:00:00 | 3319.00 | 2026-03-11 10:15:00 | 3258.40 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2026-03-17 14:45:00 | 3314.00 | 2026-03-19 09:15:00 | 3256.80 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2026-03-18 09:15:00 | 3352.30 | 2026-03-19 09:15:00 | 3256.80 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2026-03-20 09:15:00 | 3311.00 | 2026-03-23 09:15:00 | 3139.00 | STOP_HIT | 1.00 | -5.19% |
| BUY | retest2 | 2026-03-20 11:45:00 | 3335.20 | 2026-03-23 09:15:00 | 3139.00 | STOP_HIT | 1.00 | -5.88% |
| SELL | retest2 | 2026-04-08 14:45:00 | 3310.90 | 2026-04-10 09:15:00 | 3386.20 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2026-04-09 09:45:00 | 3292.80 | 2026-04-10 09:15:00 | 3386.20 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2026-04-13 09:15:00 | 3290.30 | 2026-04-15 09:15:00 | 3460.00 | STOP_HIT | 1.00 | -5.16% |
| SELL | retest2 | 2026-04-13 11:45:00 | 3310.00 | 2026-04-15 09:15:00 | 3460.00 | STOP_HIT | 1.00 | -4.53% |
