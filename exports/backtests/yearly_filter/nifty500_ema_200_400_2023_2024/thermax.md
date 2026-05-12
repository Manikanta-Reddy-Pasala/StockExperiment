# Thermax Ltd. (THERMAX)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 4707.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 10 |
| ALERT2 | 10 |
| ALERT2_SKIP | 5 |
| ALERT3 | 41 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 43 |
| PARTIAL | 6 |
| TARGET_HIT | 9 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 49 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 32
- **Target hits / Stop hits / Partials:** 9 / 34 / 6
- **Avg / median % per leg:** 0.80% / -1.85%
- **Sum % (uncompounded):** 38.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 16 | 5 | 31.2% | 5 | 11 | 0 | 0.99% | 15.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 16 | 5 | 31.2% | 5 | 11 | 0 | 0.99% | 15.8% |
| SELL (all) | 33 | 12 | 36.4% | 4 | 23 | 6 | 0.70% | 23.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 33 | 12 | 36.4% | 4 | 23 | 6 | 0.70% | 23.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 49 | 17 | 34.7% | 9 | 34 | 6 | 0.80% | 39.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2023-07-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 09:15:00 | 2409.90 | 2312.94 | 2312.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-20 14:15:00 | 2441.35 | 2317.52 | 2314.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 14:15:00 | 2727.45 | 2731.75 | 2617.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-15 15:00:00 | 2727.45 | 2731.75 | 2617.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 09:15:00 | 2841.35 | 2968.30 | 2849.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-26 09:45:00 | 2837.05 | 2968.30 | 2849.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-26 10:15:00 | 2885.25 | 2967.47 | 2850.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-26 11:30:00 | 2894.60 | 2966.51 | 2850.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-26 12:30:00 | 2890.00 | 2965.42 | 2850.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-10-27 09:15:00 | 2893.05 | 2962.45 | 2850.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-27 12:15:00 | 2830.30 | 2958.67 | 2850.79 | SL hit (close<static) qty=1.00 sl=2840.05 alert=retest2 |

### Cycle 2 — SELL (started 2023-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-28 15:15:00 | 2568.00 | 2826.07 | 2827.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-29 14:15:00 | 2513.95 | 2812.22 | 2820.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-15 09:15:00 | 2777.00 | 2742.89 | 2775.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-15 09:15:00 | 2777.00 | 2742.89 | 2775.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 09:15:00 | 2777.00 | 2742.89 | 2775.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-15 10:00:00 | 2777.00 | 2742.89 | 2775.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 10:15:00 | 2790.35 | 2743.36 | 2775.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-15 11:00:00 | 2790.35 | 2743.36 | 2775.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-15 11:15:00 | 2806.40 | 2743.99 | 2775.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-15 11:30:00 | 2819.95 | 2743.99 | 2775.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2023-12-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-21 10:15:00 | 3010.00 | 2802.16 | 2802.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-21 11:15:00 | 3036.15 | 2804.49 | 2803.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-18 09:15:00 | 3034.00 | 3059.24 | 2969.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-18 09:45:00 | 3017.90 | 3059.24 | 2969.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-24 09:15:00 | 3022.05 | 3059.78 | 2981.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-25 09:15:00 | 3098.45 | 3057.53 | 2982.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-02 13:30:00 | 3085.45 | 3081.88 | 3009.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-02-08 13:15:00 | 3408.30 | 3110.67 | 3033.93 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-08-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-09 11:15:00 | 4324.55 | 4946.64 | 4946.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-12 09:15:00 | 4211.00 | 4916.81 | 4931.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 09:15:00 | 4534.65 | 4499.67 | 4639.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 14:15:00 | 4644.60 | 4503.67 | 4637.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 4644.60 | 4503.67 | 4637.61 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 15:15:00 | 5236.20 | 4740.57 | 4738.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 14:15:00 | 5316.25 | 4768.54 | 4752.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 09:15:00 | 4896.05 | 4896.37 | 4828.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 09:15:00 | 4896.05 | 4896.37 | 4828.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 09:15:00 | 4896.05 | 4896.37 | 4828.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 09:30:00 | 4866.25 | 4896.37 | 4828.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 14:15:00 | 4972.20 | 5085.94 | 4981.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-30 15:00:00 | 4972.20 | 5085.94 | 4981.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 15:15:00 | 4930.00 | 5084.39 | 4981.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:15:00 | 4933.15 | 5084.39 | 4981.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 5032.05 | 5083.87 | 4981.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 09:15:00 | 5130.10 | 5063.02 | 4984.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 10:30:00 | 5110.70 | 5064.07 | 4985.41 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 13:15:00 | 5109.80 | 5064.89 | 4986.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 14:45:00 | 5111.00 | 5065.64 | 4987.77 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 5083.00 | 5089.75 | 5008.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 09:45:00 | 5084.70 | 5089.75 | 5008.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 11:15:00 | 4986.40 | 5088.44 | 5009.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 12:00:00 | 4986.40 | 5088.44 | 5009.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 12:15:00 | 4961.80 | 5087.18 | 5008.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 12:45:00 | 4949.80 | 5087.18 | 5008.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 13:15:00 | 4946.55 | 5085.78 | 5008.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 13:30:00 | 4955.40 | 5085.78 | 5008.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-13 14:15:00 | 4906.70 | 5084.00 | 5008.05 | SL hit (close<static) qty=1.00 sl=4915.55 alert=retest2 |

### Cycle 6 — SELL (started 2024-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 11:15:00 | 4403.70 | 4946.68 | 4948.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 09:15:00 | 4370.05 | 4724.99 | 4798.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-10 09:15:00 | 3420.00 | 3390.00 | 3675.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-10 09:45:00 | 3460.00 | 3390.00 | 3675.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 13:15:00 | 3575.80 | 3365.52 | 3594.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 15:00:00 | 3524.00 | 3367.10 | 3593.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-21 09:30:00 | 3530.50 | 3370.10 | 3593.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-21 14:30:00 | 3479.05 | 3378.01 | 3591.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 10:15:00 | 3664.70 | 3390.97 | 3587.90 | SL hit (close>static) qty=1.00 sl=3620.60 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 12:15:00 | 3592.10 | 3466.84 | 3466.57 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 12:15:00 | 3407.90 | 3469.93 | 3470.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 09:15:00 | 3401.40 | 3467.57 | 3468.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 3489.30 | 3457.55 | 3463.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 3489.30 | 3457.55 | 3463.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 3489.30 | 3457.55 | 3463.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 3489.30 | 3457.55 | 3463.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 3531.50 | 3458.29 | 3464.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:45:00 | 3529.30 | 3458.29 | 3464.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 3488.30 | 3460.47 | 3464.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:45:00 | 3486.40 | 3460.47 | 3464.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 12:15:00 | 3455.00 | 3460.41 | 3464.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 12:30:00 | 3459.10 | 3460.41 | 3464.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 3459.50 | 3460.33 | 3464.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 3459.50 | 3460.33 | 3464.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 3490.80 | 3460.54 | 3464.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 3490.80 | 3460.54 | 3464.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 3470.70 | 3460.64 | 3464.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:30:00 | 3467.90 | 3460.69 | 3464.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 13:15:00 | 3470.10 | 3453.92 | 3460.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 14:00:00 | 3470.10 | 3454.08 | 3460.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 14:45:00 | 3470.00 | 3454.22 | 3460.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 3442.40 | 3454.00 | 3460.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:45:00 | 3456.80 | 3454.00 | 3460.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 3438.20 | 3453.84 | 3460.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 11:45:00 | 3433.00 | 3453.60 | 3460.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 11:15:00 | 3475.00 | 3453.61 | 3460.06 | SL hit (close>static) qty=1.00 sl=3471.20 alert=retest2 |

### Cycle 9 — BUY (started 2025-07-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 15:15:00 | 3649.80 | 3466.47 | 3466.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 3811.20 | 3469.90 | 3467.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-04 11:15:00 | 3621.10 | 3677.00 | 3595.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 11:15:00 | 3621.10 | 3677.00 | 3595.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 3621.10 | 3677.00 | 3595.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:45:00 | 3589.50 | 3677.00 | 3595.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 3563.80 | 3675.14 | 3595.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 3563.80 | 3675.14 | 3595.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 3540.80 | 3673.80 | 3595.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 15:15:00 | 3526.50 | 3673.80 | 3595.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 11:15:00 | 3271.10 | 3535.76 | 3537.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-20 12:15:00 | 3261.60 | 3486.36 | 3510.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 12:15:00 | 3361.00 | 3360.79 | 3425.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 13:00:00 | 3361.00 | 3360.79 | 3425.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 3403.00 | 3358.85 | 3421.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 11:15:00 | 3377.90 | 3359.08 | 3421.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 13:00:00 | 3374.00 | 3359.50 | 3420.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 3209.01 | 3335.00 | 3386.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 3205.30 | 3335.00 | 3386.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 3231.60 | 3221.63 | 3291.11 | SL hit (close>ema200) qty=0.50 sl=3221.63 alert=retest2 |

### Cycle 11 — BUY (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 12:15:00 | 3185.60 | 2983.29 | 2982.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-13 14:15:00 | 3220.60 | 3061.77 | 3028.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 3069.10 | 3120.70 | 3066.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-23 10:00:00 | 3069.10 | 3120.70 | 3066.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 3077.90 | 3120.27 | 3066.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 11:15:00 | 3098.80 | 3120.27 | 3066.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 11:45:00 | 3100.10 | 3120.10 | 3066.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 11:00:00 | 3098.90 | 3119.10 | 3068.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-09 10:15:00 | 3408.68 | 3186.17 | 3120.01 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-10-26 11:30:00 | 2894.60 | 2023-10-27 12:15:00 | 2830.30 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2023-10-26 12:30:00 | 2890.00 | 2023-10-27 12:15:00 | 2830.30 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2023-10-27 09:15:00 | 2893.05 | 2023-10-27 12:15:00 | 2830.30 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2023-11-01 12:00:00 | 2889.00 | 2023-11-17 13:15:00 | 2844.50 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2023-11-07 11:00:00 | 2929.60 | 2023-11-17 13:15:00 | 2844.50 | STOP_HIT | 1.00 | -2.90% |
| BUY | retest2 | 2023-11-08 15:00:00 | 2943.90 | 2023-11-17 13:15:00 | 2844.50 | STOP_HIT | 1.00 | -3.38% |
| BUY | retest2 | 2023-11-09 10:45:00 | 2934.55 | 2023-11-17 14:15:00 | 2828.00 | STOP_HIT | 1.00 | -3.63% |
| BUY | retest2 | 2024-01-25 09:15:00 | 3098.45 | 2024-02-08 13:15:00 | 3408.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-02-02 13:30:00 | 3085.45 | 2024-02-08 13:15:00 | 3393.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-07 09:15:00 | 5130.10 | 2024-11-13 14:15:00 | 4906.70 | STOP_HIT | 1.00 | -4.35% |
| BUY | retest2 | 2024-11-07 10:30:00 | 5110.70 | 2024-11-13 14:15:00 | 4906.70 | STOP_HIT | 1.00 | -3.99% |
| BUY | retest2 | 2024-11-07 13:15:00 | 5109.80 | 2024-11-13 14:15:00 | 4906.70 | STOP_HIT | 1.00 | -3.97% |
| BUY | retest2 | 2024-11-07 14:45:00 | 5111.00 | 2024-11-13 14:15:00 | 4906.70 | STOP_HIT | 1.00 | -4.00% |
| SELL | retest2 | 2025-03-20 15:00:00 | 3524.00 | 2025-03-25 10:15:00 | 3664.70 | STOP_HIT | 1.00 | -3.99% |
| SELL | retest2 | 2025-03-21 09:30:00 | 3530.50 | 2025-03-25 10:15:00 | 3664.70 | STOP_HIT | 1.00 | -3.80% |
| SELL | retest2 | 2025-03-21 14:30:00 | 3479.05 | 2025-03-25 10:15:00 | 3664.70 | STOP_HIT | 1.00 | -5.34% |
| SELL | retest2 | 2025-03-25 11:30:00 | 3545.75 | 2025-03-27 14:15:00 | 3767.00 | STOP_HIT | 1.00 | -6.24% |
| SELL | retest2 | 2025-03-26 11:00:00 | 3485.80 | 2025-03-27 14:15:00 | 3767.00 | STOP_HIT | 1.00 | -8.07% |
| SELL | retest2 | 2025-04-04 09:15:00 | 3485.85 | 2025-04-07 09:15:00 | 3137.26 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-07 09:15:00 | 3328.55 | 2025-04-07 09:15:00 | 3162.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-07 09:15:00 | 3328.55 | 2025-04-17 10:15:00 | 3390.10 | STOP_HIT | 0.50 | -1.85% |
| SELL | retest2 | 2025-04-25 09:30:00 | 3479.10 | 2025-04-30 14:15:00 | 3305.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-25 14:45:00 | 3468.30 | 2025-04-30 14:15:00 | 3294.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-28 09:45:00 | 3449.10 | 2025-04-30 14:15:00 | 3276.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-25 09:30:00 | 3479.10 | 2025-05-09 09:15:00 | 3131.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-25 14:45:00 | 3468.30 | 2025-05-09 09:15:00 | 3121.47 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-04-28 09:45:00 | 3449.10 | 2025-05-09 09:15:00 | 3104.19 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-05-15 10:45:00 | 3469.20 | 2025-05-19 09:15:00 | 3551.40 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-05-15 12:00:00 | 3462.60 | 2025-05-19 09:15:00 | 3551.40 | STOP_HIT | 1.00 | -2.56% |
| SELL | retest2 | 2025-05-29 10:30:00 | 3410.30 | 2025-05-29 11:15:00 | 3440.60 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-05-29 11:00:00 | 3416.50 | 2025-05-29 11:15:00 | 3440.60 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2025-05-29 12:15:00 | 3411.10 | 2025-05-30 09:15:00 | 3441.80 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-05-29 15:00:00 | 3417.00 | 2025-05-30 09:15:00 | 3441.80 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-05-30 11:15:00 | 3410.30 | 2025-06-04 10:15:00 | 3474.90 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-05-30 15:00:00 | 3387.00 | 2025-06-04 10:15:00 | 3474.90 | STOP_HIT | 1.00 | -2.60% |
| SELL | retest2 | 2025-06-03 11:30:00 | 3414.90 | 2025-06-04 10:15:00 | 3474.90 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-06-03 12:45:00 | 3401.70 | 2025-06-04 10:15:00 | 3474.90 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-07-07 11:30:00 | 3467.90 | 2025-07-15 11:15:00 | 3475.00 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-07-11 13:15:00 | 3470.10 | 2025-07-16 09:15:00 | 3541.10 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-07-11 14:00:00 | 3470.10 | 2025-07-16 09:15:00 | 3541.10 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-07-11 14:45:00 | 3470.00 | 2025-07-16 09:15:00 | 3541.10 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2025-07-14 11:45:00 | 3433.00 | 2025-07-16 09:15:00 | 3541.10 | STOP_HIT | 1.00 | -3.15% |
| SELL | retest2 | 2025-09-10 11:15:00 | 3377.90 | 2025-09-26 09:15:00 | 3209.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-10 13:00:00 | 3374.00 | 2025-09-26 09:15:00 | 3205.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-10 11:15:00 | 3377.90 | 2025-10-23 10:15:00 | 3231.60 | STOP_HIT | 0.50 | 4.33% |
| SELL | retest2 | 2025-09-10 13:00:00 | 3374.00 | 2025-10-23 10:15:00 | 3231.60 | STOP_HIT | 0.50 | 4.22% |
| BUY | retest2 | 2026-03-23 11:15:00 | 3098.80 | 2026-04-09 10:15:00 | 3408.68 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-23 11:45:00 | 3100.10 | 2026-04-09 10:15:00 | 3410.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-24 11:00:00 | 3098.90 | 2026-04-09 10:15:00 | 3408.79 | TARGET_HIT | 1.00 | 10.00% |
