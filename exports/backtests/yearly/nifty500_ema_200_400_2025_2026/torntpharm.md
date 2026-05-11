# Torrent Pharmaceuticals Ltd. (TORNTPHARM)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 4385.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 4 |
| ALERT1 | 3 |
| ALERT2 | 3 |
| ALERT2_SKIP | 2 |
| ALERT3 | 26 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 21 |
| PARTIAL | 0 |
| TARGET_HIT | 7 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 14
- **Target hits / Stop hits / Partials:** 7 / 14 / 0
- **Avg / median % per leg:** 2.28% / -0.74%
- **Sum % (uncompounded):** 47.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 18 | 7 | 38.9% | 7 | 11 | 0 | 2.78% | 50.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 18 | 7 | 38.9% | 7 | 11 | 0 | 2.78% | 50.0% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.73% | -2.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.73% | -2.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 21 | 7 | 33.3% | 7 | 14 | 0 | 2.28% | 47.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 3140.50 | 3208.67 | 3208.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 15:15:00 | 3126.50 | 3204.26 | 3206.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 10:15:00 | 3212.70 | 3190.58 | 3198.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 10:15:00 | 3212.70 | 3190.58 | 3198.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 3212.70 | 3190.58 | 3198.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:00:00 | 3212.70 | 3190.58 | 3198.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 3220.90 | 3190.88 | 3199.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:00:00 | 3220.90 | 3190.88 | 3199.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 3204.30 | 3192.54 | 3199.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:30:00 | 3200.80 | 3192.54 | 3199.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 3194.90 | 3192.56 | 3199.61 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 12:15:00 | 3254.40 | 3205.80 | 3205.66 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 11:15:00 | 3169.60 | 3205.62 | 3205.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 12:15:00 | 3157.00 | 3205.14 | 3205.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-25 10:15:00 | 3214.70 | 3199.74 | 3202.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-25 10:15:00 | 3214.70 | 3199.74 | 3202.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 3214.70 | 3199.74 | 3202.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 11:00:00 | 3214.70 | 3199.74 | 3202.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 3211.20 | 3199.85 | 3202.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 12:15:00 | 3209.40 | 3199.85 | 3202.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 12:45:00 | 3210.00 | 3200.04 | 3202.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-25 13:15:00 | 3229.30 | 3200.33 | 3202.79 | SL hit (close>static) qty=1.00 sl=3225.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 12:15:00 | 3293.40 | 3205.21 | 3205.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 13:15:00 | 3326.70 | 3206.42 | 3205.71 | Break + close above crossover candle high |
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
