# Siemens Ltd. (SIEMENS)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 3838.00
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
| ALERT2_SKIP | 3 |
| ALERT3 | 20 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 8 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 8 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 9 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 7
- **Target hits / Stop hits / Partials:** 0 / 8 / 1
- **Avg / median % per leg:** -0.96% / -1.59%
- **Sum % (uncompounded):** -8.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.54% | -10.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -2.54% | -10.1% |
| SELL (all) | 5 | 2 | 40.0% | 0 | 4 | 1 | 0.29% | 1.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 5 | 2 | 40.0% | 0 | 4 | 1 | 0.29% | 1.5% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 9 | 2 | 22.2% | 0 | 8 | 1 | -0.96% | -8.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 3116.80 | 3150.00 | 3150.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 11:15:00 | 3091.60 | 3145.18 | 3147.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 3179.60 | 3112.87 | 3128.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 09:15:00 | 3179.60 | 3112.87 | 3128.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 3179.60 | 3112.87 | 3128.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:00:00 | 3179.60 | 3112.87 | 3128.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 3139.00 | 3113.13 | 3128.81 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 14:15:00 | 3166.10 | 3142.15 | 3142.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 14:15:00 | 3185.50 | 3143.88 | 3143.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 13:15:00 | 3196.30 | 3221.88 | 3188.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 14:00:00 | 3196.30 | 3221.88 | 3188.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 3153.50 | 3220.74 | 3188.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 3129.00 | 3220.74 | 3188.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 10:15:00 | 3142.40 | 3219.96 | 3187.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 10:45:00 | 3142.80 | 3219.96 | 3187.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 3170.00 | 3212.16 | 3185.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:00:00 | 3170.00 | 3212.16 | 3185.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 3175.40 | 3211.79 | 3185.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 11:45:00 | 3167.20 | 3211.79 | 3185.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 3176.90 | 3210.17 | 3185.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:00:00 | 3176.90 | 3210.17 | 3185.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 3158.90 | 3209.66 | 3185.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 11:45:00 | 3158.00 | 3209.66 | 3185.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 3145.00 | 3204.54 | 3184.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 13:00:00 | 3145.00 | 3204.54 | 3184.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 13:15:00 | 3105.00 | 3168.55 | 3168.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 09:15:00 | 3090.00 | 3166.46 | 3167.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 14:15:00 | 3134.80 | 3131.15 | 3146.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-07 15:00:00 | 3134.80 | 3131.15 | 3146.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 3130.90 | 3131.18 | 3146.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 10:30:00 | 3118.30 | 3130.85 | 3146.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 10:15:00 | 2962.39 | 3119.85 | 3139.82 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-29 11:15:00 | 3055.40 | 3014.78 | 3071.00 | SL hit (close>ema200) qty=0.50 sl=3014.78 alert=retest2 |

### Cycle 4 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 3184.00 | 3104.03 | 3103.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 10:15:00 | 3191.30 | 3108.01 | 3105.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 3104.50 | 3114.28 | 3109.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 14:15:00 | 3104.50 | 3114.28 | 3109.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 3104.50 | 3114.28 | 3109.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 3104.50 | 3114.28 | 3109.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 3110.10 | 3114.24 | 3109.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 3168.00 | 3114.24 | 3109.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-16 14:15:00 | 3150.50 | 3217.21 | 3178.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:45:00 | 3112.00 | 3208.83 | 3177.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-19 13:15:00 | 3090.10 | 3207.65 | 3177.22 | SL hit (close<static) qty=1.00 sl=3098.20 alert=retest2 |

### Cycle 5 — SELL (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 13:15:00 | 2963.70 | 3151.96 | 3152.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 2937.50 | 3149.83 | 3151.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 3173.10 | 3117.35 | 3133.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 3173.10 | 3117.35 | 3133.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 3173.10 | 3117.35 | 3133.57 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 11:15:00 | 3360.20 | 3149.48 | 3148.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 09:15:00 | 3504.10 | 3161.01 | 3154.41 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-01-08 10:30:00 | 3118.30 | 2026-01-12 10:15:00 | 2962.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 10:30:00 | 3118.30 | 2026-01-29 11:15:00 | 3055.40 | STOP_HIT | 0.50 | 2.02% |
| SELL | retest2 | 2026-02-09 09:15:00 | 3073.00 | 2026-02-11 14:15:00 | 3152.60 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2026-02-09 10:30:00 | 3110.20 | 2026-02-11 14:15:00 | 3152.60 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-02-09 11:00:00 | 3103.30 | 2026-02-11 14:15:00 | 3152.60 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2026-02-20 09:15:00 | 3168.00 | 2026-03-19 13:15:00 | 3090.10 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2026-03-16 14:15:00 | 3150.50 | 2026-03-19 13:15:00 | 3090.10 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2026-03-19 12:45:00 | 3112.00 | 2026-03-19 13:15:00 | 3090.10 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2026-03-20 09:15:00 | 3142.00 | 2026-03-23 09:15:00 | 2983.00 | STOP_HIT | 1.00 | -5.06% |
