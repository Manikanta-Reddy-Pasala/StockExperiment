# Navin Fluorine International Ltd. (NAVINFLUOR)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 7039.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 0 |
| ALERT3 | 39 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 35 |
| PARTIAL | 1 |
| TARGET_HIT | 16 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 36 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 18 / 18
- **Target hits / Stop hits / Partials:** 16 / 19 / 1
- **Avg / median % per leg:** 3.38% / 2.60%
- **Sum % (uncompounded):** 121.70%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 16 | 72.7% | 16 | 6 | 0 | 6.84% | 150.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 22 | 16 | 72.7% | 16 | 6 | 0 | 6.84% | 150.6% |
| SELL (all) | 14 | 2 | 14.3% | 0 | 13 | 1 | -2.06% | -28.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 2 | 14.3% | 0 | 13 | 1 | -2.06% | -28.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 36 | 18 | 50.0% | 16 | 19 | 1 | 3.38% | 121.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-08-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 15:15:00 | 3291.90 | 3497.94 | 3498.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 15:15:00 | 3280.00 | 3442.19 | 3468.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 3398.00 | 3383.15 | 3426.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 10:00:00 | 3398.00 | 3383.15 | 3426.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 3415.10 | 3335.62 | 3385.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-24 10:00:00 | 3415.10 | 3335.62 | 3385.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 3410.10 | 3336.36 | 3385.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-24 10:30:00 | 3408.00 | 3336.36 | 3385.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 12:15:00 | 3377.15 | 3337.22 | 3385.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 14:45:00 | 3373.00 | 3338.13 | 3385.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 09:15:00 | 3406.75 | 3339.22 | 3385.89 | SL hit (close>static) qty=1.00 sl=3390.75 alert=retest2 |

### Cycle 2 — BUY (started 2024-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-11 13:15:00 | 3474.85 | 3387.79 | 3387.72 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 3257.95 | 3387.38 | 3387.56 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 14:15:00 | 3475.30 | 3382.88 | 3382.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 09:15:00 | 3501.35 | 3384.89 | 3383.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 12:15:00 | 3485.00 | 3487.88 | 3446.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-16 13:00:00 | 3485.00 | 3487.88 | 3446.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 3407.25 | 3486.51 | 3448.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-19 09:30:00 | 3433.50 | 3486.51 | 3448.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 10:15:00 | 3398.05 | 3485.63 | 3448.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-19 11:00:00 | 3398.05 | 3485.63 | 3448.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2024-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 11:15:00 | 3200.00 | 3419.93 | 3420.65 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-01-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 12:15:00 | 3862.00 | 3417.54 | 3416.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 14:15:00 | 3914.35 | 3605.84 | 3535.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 3947.90 | 3950.16 | 3789.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 3947.90 | 3950.16 | 3789.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 14:15:00 | 3802.00 | 3947.26 | 3803.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-25 15:00:00 | 3802.00 | 3947.26 | 3803.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 15:15:00 | 3823.55 | 3946.03 | 3803.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 09:15:00 | 3794.00 | 3946.03 | 3803.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 3738.35 | 3943.97 | 3802.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 10:00:00 | 3738.35 | 3943.97 | 3802.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 10:15:00 | 3670.00 | 3941.24 | 3802.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 11:00:00 | 3670.00 | 3941.24 | 3802.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 3700.75 | 3927.34 | 3799.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-03 13:00:00 | 3842.00 | 3906.34 | 3794.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-03 15:00:00 | 3841.05 | 3904.44 | 3794.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-04 09:45:00 | 3822.30 | 3903.00 | 3795.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-10 09:15:00 | 4226.20 | 3939.40 | 3828.35 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2025-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 11:15:00 | 4644.00 | 4807.09 | 4807.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 13:15:00 | 4617.60 | 4803.51 | 4805.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 10:15:00 | 4726.00 | 4710.56 | 4751.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 11:00:00 | 4726.00 | 4710.56 | 4751.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 4740.50 | 4710.83 | 4750.43 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2025-10-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 15:15:00 | 5110.00 | 4783.00 | 4782.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 13:15:00 | 5151.60 | 4797.71 | 4789.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 5642.50 | 5695.26 | 5451.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 10:00:00 | 5642.50 | 5695.26 | 5451.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 5568.50 | 5687.09 | 5456.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:30:00 | 5505.00 | 5687.09 | 5456.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 5792.50 | 5845.07 | 5693.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 5724.50 | 5845.07 | 5693.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 5706.50 | 5842.08 | 5696.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 15:00:00 | 5706.50 | 5842.08 | 5696.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 5761.00 | 5896.52 | 5754.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:45:00 | 5762.50 | 5896.52 | 5754.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 5688.50 | 5894.45 | 5754.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:45:00 | 5645.00 | 5894.45 | 5754.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 11:15:00 | 5712.00 | 5892.64 | 5753.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 11:30:00 | 5615.00 | 5892.64 | 5753.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 5725.00 | 5886.68 | 5754.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 5725.00 | 5886.68 | 5754.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 5722.00 | 5885.05 | 5754.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:30:00 | 5739.00 | 5885.05 | 5754.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 5739.50 | 5883.60 | 5753.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:45:00 | 5711.50 | 5883.60 | 5753.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 5755.00 | 5885.82 | 5763.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 11:00:00 | 5755.00 | 5885.82 | 5763.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 5803.00 | 5885.00 | 5763.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 14:30:00 | 5829.50 | 5882.15 | 5763.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-03 09:15:00 | 6412.45 | 5925.86 | 5807.62 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-14 10:15:00 | 3278.70 | 2024-05-31 11:15:00 | 3219.85 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2024-05-30 10:30:00 | 3267.65 | 2024-05-31 11:15:00 | 3219.85 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-06-03 14:00:00 | 3334.60 | 2024-06-04 09:15:00 | 3209.90 | STOP_HIT | 1.00 | -3.74% |
| BUY | retest2 | 2024-06-05 12:00:00 | 3271.10 | 2024-06-14 09:15:00 | 3598.21 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-06 09:45:00 | 3317.50 | 2024-06-19 09:15:00 | 3649.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-13 12:45:00 | 3311.40 | 2024-08-13 14:15:00 | 3272.15 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-08-16 14:30:00 | 3310.95 | 2024-08-20 15:15:00 | 3291.90 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2024-08-19 11:00:00 | 3313.80 | 2024-08-20 15:15:00 | 3291.90 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2024-09-24 14:45:00 | 3373.00 | 2024-09-25 09:15:00 | 3406.75 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-09-25 11:45:00 | 3372.35 | 2024-09-27 09:15:00 | 3405.15 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-09-26 09:15:00 | 3360.95 | 2024-09-27 09:15:00 | 3405.15 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-09-26 10:30:00 | 3369.15 | 2024-09-27 09:15:00 | 3405.15 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-10-09 13:15:00 | 3366.85 | 2024-10-10 09:15:00 | 3418.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2024-10-09 14:30:00 | 3369.90 | 2024-10-10 09:15:00 | 3418.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-10-09 15:00:00 | 3361.90 | 2024-10-10 09:15:00 | 3418.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2024-10-14 09:15:00 | 3367.00 | 2024-10-16 13:15:00 | 3430.55 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2024-10-17 13:15:00 | 3303.00 | 2024-10-24 09:15:00 | 3517.00 | STOP_HIT | 1.00 | -6.48% |
| SELL | retest2 | 2024-10-18 09:15:00 | 3296.50 | 2024-10-24 09:15:00 | 3517.00 | STOP_HIT | 1.00 | -6.69% |
| SELL | retest2 | 2024-10-21 14:15:00 | 3311.70 | 2024-10-24 09:15:00 | 3517.00 | STOP_HIT | 1.00 | -6.20% |
| SELL | retest2 | 2024-10-21 15:00:00 | 3310.10 | 2024-10-24 09:15:00 | 3517.00 | STOP_HIT | 1.00 | -6.25% |
| SELL | retest2 | 2024-10-24 11:30:00 | 3451.70 | 2024-10-25 13:15:00 | 3279.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 11:30:00 | 3451.70 | 2024-10-28 10:15:00 | 3362.00 | STOP_HIT | 0.50 | 2.60% |
| BUY | retest2 | 2025-03-03 13:00:00 | 3842.00 | 2025-03-10 09:15:00 | 4226.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-03 15:00:00 | 3841.05 | 2025-03-10 09:15:00 | 4225.16 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-04 09:45:00 | 3822.30 | 2025-03-10 09:15:00 | 4204.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-07 14:30:00 | 3837.95 | 2025-04-15 09:15:00 | 4221.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-15 09:15:00 | 4171.70 | 2025-05-02 11:15:00 | 4588.87 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-27 14:30:00 | 5829.50 | 2026-02-03 09:15:00 | 6412.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 12:30:00 | 5822.00 | 2026-04-15 09:15:00 | 6404.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 13:15:00 | 5820.50 | 2026-04-15 09:15:00 | 6402.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-02 14:00:00 | 5825.00 | 2026-04-15 09:15:00 | 6407.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-09 12:30:00 | 6191.00 | 2026-04-29 09:15:00 | 6810.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-10 09:15:00 | 6149.00 | 2026-04-29 09:15:00 | 6763.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-10 10:00:00 | 6166.00 | 2026-04-29 09:15:00 | 6782.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-10 12:45:00 | 6157.50 | 2026-04-29 09:15:00 | 6773.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 10:30:00 | 6187.00 | 2026-04-29 09:15:00 | 6805.70 | TARGET_HIT | 1.00 | 10.00% |
