# LT (LT)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 3978.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 7 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 4 |
| ALERT3 | 14 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 0 |
| TARGET_HIT | 4 |
| STOP_HIT | 15 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 18 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 4 / 14
- **Target hits / Stop hits / Partials:** 4 / 14 / 0
- **Avg / median % per leg:** 1.20% / -0.90%
- **Sum % (uncompounded):** 21.63%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 4 | 26.7% | 4 | 11 | 0 | 1.59% | 23.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 4 | 26.7% | 4 | 11 | 0 | 1.59% | 23.8% |
| SELL (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.72% | -2.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.72% | -2.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 18 | 4 | 22.2% | 4 | 14 | 0 | 1.20% | 21.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 3593.10 | 3351.74 | 3350.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 13:15:00 | 3639.00 | 3373.98 | 3362.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 3553.50 | 3578.39 | 3503.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 3553.50 | 3578.39 | 3503.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 3553.50 | 3578.39 | 3503.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 3576.10 | 3578.39 | 3503.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 12:15:00 | 3570.80 | 3578.05 | 3503.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 13:00:00 | 3568.50 | 3577.96 | 3504.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 10:15:00 | 3484.80 | 3581.03 | 3553.69 | SL hit (close<static) qty=1.00 sl=3485.90 alert=retest2 |

### Cycle 2 — SELL (started 2025-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 14:15:00 | 3422.00 | 3532.73 | 3532.98 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 13:15:00 | 3678.50 | 3533.26 | 3533.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 3715.00 | 3578.00 | 3558.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 09:15:00 | 3589.90 | 3601.96 | 3575.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 09:30:00 | 3583.90 | 3601.96 | 3575.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 3573.20 | 3602.53 | 3578.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 3572.50 | 3602.53 | 3578.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 3575.40 | 3602.26 | 3578.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:45:00 | 3576.90 | 3602.26 | 3578.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 3575.00 | 3601.99 | 3578.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:00:00 | 3575.00 | 3601.99 | 3578.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 3575.60 | 3601.73 | 3578.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:30:00 | 3573.60 | 3601.73 | 3578.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 10:15:00 | 3607.70 | 3597.59 | 3578.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 12:00:00 | 3613.00 | 3597.74 | 3578.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 12:30:00 | 3611.80 | 3597.77 | 3578.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 13:30:00 | 3611.90 | 3597.91 | 3578.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 14:00:00 | 3611.60 | 3597.91 | 3578.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 3566.00 | 3598.01 | 3579.88 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-02 13:15:00 | 3566.00 | 3598.01 | 3579.88 | SL hit (close<static) qty=1.00 sl=3570.20 alert=retest2 |

### Cycle 4 — SELL (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 11:15:00 | 3770.80 | 3950.99 | 3951.81 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2026-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 09:15:00 | 4044.00 | 3948.93 | 3948.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 09:15:00 | 4107.80 | 3964.27 | 3956.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 09:15:00 | 4057.70 | 4162.32 | 4081.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-02 09:15:00 | 4057.70 | 4162.32 | 4081.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 4057.70 | 4162.32 | 4081.08 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 3485.60 | 4023.27 | 4024.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 13:15:00 | 3474.90 | 4017.82 | 4021.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-06 14:15:00 | 3732.90 | 3727.11 | 3839.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 3971.00 | 3728.58 | 3835.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 3971.00 | 3728.58 | 3835.14 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 4071.00 | 3903.71 | 3903.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 4082.30 | 3922.45 | 3912.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 09:15:00 | 3919.50 | 3956.54 | 3932.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 09:15:00 | 3919.50 | 3956.54 | 3932.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 3919.50 | 3956.54 | 3932.45 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 12:15:00 | 3571.90 | 2025-05-14 10:15:00 | 3593.10 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-05-12 12:45:00 | 3568.10 | 2025-05-14 10:15:00 | 3593.10 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-05-13 09:15:00 | 3561.80 | 2025-05-14 10:15:00 | 3593.10 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-06-13 10:15:00 | 3576.10 | 2025-07-17 10:15:00 | 3484.80 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-06-13 12:15:00 | 3570.80 | 2025-07-17 10:15:00 | 3484.80 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-06-13 13:00:00 | 3568.50 | 2025-07-17 10:15:00 | 3484.80 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-08-29 12:00:00 | 3613.00 | 2025-09-02 13:15:00 | 3566.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-08-29 12:30:00 | 3611.80 | 2025-09-02 13:15:00 | 3566.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-08-29 13:30:00 | 3611.90 | 2025-09-02 13:15:00 | 3566.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2025-08-29 14:00:00 | 3611.60 | 2025-09-02 13:15:00 | 3566.00 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2025-09-03 09:45:00 | 3584.10 | 2025-09-05 14:15:00 | 3552.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-09-03 13:45:00 | 3582.40 | 2025-09-05 14:15:00 | 3552.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-09-04 12:30:00 | 3588.10 | 2025-09-05 14:15:00 | 3552.00 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-09-04 14:30:00 | 3589.20 | 2025-09-05 14:15:00 | 3552.00 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-09-12 11:00:00 | 3588.50 | 2025-10-23 11:15:00 | 3947.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-12 11:45:00 | 3593.00 | 2025-10-23 11:15:00 | 3952.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-15 09:15:00 | 3590.00 | 2025-10-23 11:15:00 | 3949.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-15 09:45:00 | 3601.40 | 2025-10-23 12:15:00 | 3961.54 | TARGET_HIT | 1.00 | 10.00% |
