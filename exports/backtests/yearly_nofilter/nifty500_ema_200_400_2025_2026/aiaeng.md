# AIA Engineering Ltd. (AIAENG)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 3955.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 9 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 6 |
| PARTIAL | 0 |
| TARGET_HIT | 2 |
| STOP_HIT | 5 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 6 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 4
- **Target hits / Stop hits / Partials:** 2 / 4 / 0
- **Avg / median % per leg:** 1.20% / -2.92%
- **Sum % (uncompounded):** 7.17%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 2 | 2 | 100.0% | 2 | 0 | 0 | 10.00% | 20.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 2 | 2 | 100.0% | 2 | 0 | 0 | 10.00% | 20.0% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.21% | -12.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -3.21% | -12.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 6 | 2 | 33.3% | 2 | 4 | 0 | 1.20% | 7.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 12:15:00 | 3502.20 | 3286.12 | 3285.52 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-07-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-29 12:15:00 | 3176.60 | 3345.00 | 3345.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 3163.00 | 3328.80 | 3337.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 10:15:00 | 3108.40 | 3107.12 | 3171.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 10:30:00 | 3102.90 | 3107.12 | 3171.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 3148.70 | 3095.54 | 3143.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 3148.70 | 3095.54 | 3143.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 3129.00 | 3095.87 | 3143.11 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 11:15:00 | 3310.00 | 3172.80 | 3172.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 15:15:00 | 3315.00 | 3178.15 | 3175.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-03 10:15:00 | 3243.30 | 3243.56 | 3214.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-03 10:45:00 | 3240.20 | 3243.56 | 3214.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 3228.50 | 3253.33 | 3222.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:30:00 | 3239.60 | 3253.33 | 3222.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 3255.40 | 3253.35 | 3222.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 11:30:00 | 3263.70 | 3253.43 | 3222.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 15:15:00 | 3260.00 | 3253.39 | 3222.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-12 09:15:00 | 3590.07 | 3287.54 | 3242.83 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 12:15:00 | 3703.80 | 3818.26 | 3818.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 15:15:00 | 3678.00 | 3814.77 | 3816.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 3900.00 | 3795.94 | 3806.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 3900.00 | 3795.94 | 3806.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 3900.00 | 3795.94 | 3806.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:30:00 | 3890.50 | 3795.94 | 3806.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 3825.20 | 3796.23 | 3807.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:30:00 | 3863.00 | 3796.23 | 3807.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 3677.00 | 3665.92 | 3721.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 10:15:00 | 3673.70 | 3665.92 | 3721.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-08 15:15:00 | 3670.00 | 3666.58 | 3720.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:30:00 | 3655.00 | 3666.62 | 3720.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 13:15:00 | 3655.10 | 3667.18 | 3719.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 3780.90 | 3667.60 | 3718.97 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-10 09:15:00 | 3780.90 | 3667.60 | 3718.97 | SL hit (close>static) qty=1.00 sl=3778.00 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 11:15:00 | 4015.80 | 3759.10 | 3758.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-21 13:15:00 | 4044.80 | 3764.49 | 3761.29 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-11-07 11:30:00 | 3263.70 | 2025-11-12 09:15:00 | 3590.07 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-07 15:15:00 | 3260.00 | 2025-11-12 09:15:00 | 3586.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-08 10:15:00 | 3673.70 | 2026-04-10 09:15:00 | 3780.90 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2026-04-08 15:15:00 | 3670.00 | 2026-04-10 09:15:00 | 3780.90 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2026-04-09 09:30:00 | 3655.00 | 2026-04-10 09:15:00 | 3780.90 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2026-04-09 13:15:00 | 3655.10 | 2026-04-10 09:15:00 | 3780.90 | STOP_HIT | 1.00 | -3.44% |
