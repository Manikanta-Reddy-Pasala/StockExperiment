# Schaeffler India Ltd. (SCHAEFFLER)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 4226.20
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 41 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 17 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 16 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 21 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 16
- **Target hits / Stop hits / Partials:** 1 / 16 / 4
- **Avg / median % per leg:** -0.42% / -2.13%
- **Sum % (uncompounded):** -8.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 1 | 11.1% | 1 | 8 | 0 | -1.18% | -10.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 9 | 1 | 11.1% | 1 | 8 | 0 | -1.18% | -10.6% |
| SELL (all) | 12 | 4 | 33.3% | 0 | 8 | 4 | 0.15% | 1.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 12 | 4 | 33.3% | 0 | 8 | 4 | 0.15% | 1.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 21 | 5 | 23.8% | 1 | 16 | 4 | -0.42% | -8.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 11:15:00 | 3853.20 | 3943.90 | 3944.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 15:15:00 | 3841.50 | 3940.54 | 3942.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 3951.60 | 3936.02 | 3940.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 3951.60 | 3936.02 | 3940.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 3951.60 | 3936.02 | 3940.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:45:00 | 3959.00 | 3936.02 | 3940.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 3981.00 | 3936.46 | 3940.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:45:00 | 3981.40 | 3936.46 | 3940.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 4005.00 | 3938.56 | 3941.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 4004.50 | 3938.56 | 3941.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 3964.10 | 3942.06 | 3942.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:00:00 | 3964.10 | 3942.06 | 3942.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 3926.10 | 3941.90 | 3942.90 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-09-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 12:15:00 | 3996.00 | 3944.12 | 3943.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-15 14:15:00 | 4006.00 | 3945.15 | 3944.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 4012.80 | 4017.00 | 3986.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-26 14:00:00 | 4012.80 | 4017.00 | 3986.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 3992.70 | 4016.75 | 3986.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 15:00:00 | 3992.70 | 4016.75 | 3986.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 3995.30 | 4016.54 | 3986.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 3974.60 | 4016.12 | 3986.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 3966.90 | 4015.63 | 3986.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:30:00 | 3961.10 | 4015.63 | 3986.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 4049.90 | 4094.09 | 4039.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:45:00 | 4050.20 | 4094.09 | 4039.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 4028.70 | 4093.44 | 4039.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:00:00 | 4028.70 | 4093.44 | 4039.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 4025.60 | 4092.76 | 4039.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:00:00 | 4025.60 | 4092.76 | 4039.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 3914.50 | 4088.40 | 4038.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 3914.50 | 4088.40 | 4038.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 14:15:00 | 3948.60 | 4003.58 | 4003.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 09:15:00 | 3919.70 | 4002.21 | 4003.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 14:15:00 | 4013.40 | 3995.76 | 3999.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 14:15:00 | 4013.40 | 3995.76 | 3999.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 4013.40 | 3995.76 | 3999.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 4013.40 | 3995.76 | 3999.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 4023.00 | 3996.03 | 3999.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:15:00 | 4133.00 | 3996.03 | 3999.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 4204.40 | 3998.11 | 4000.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 10:00:00 | 4204.40 | 3998.11 | 4000.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 11:15:00 | 4286.00 | 4003.46 | 4003.41 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 3840.20 | 4024.60 | 4025.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 3800.80 | 3983.08 | 4002.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 14:15:00 | 3903.60 | 3901.73 | 3948.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-22 15:00:00 | 3903.60 | 3901.73 | 3948.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 3841.90 | 3860.08 | 3907.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 14:45:00 | 3890.00 | 3860.08 | 3907.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 3807.70 | 3859.49 | 3906.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 15:00:00 | 3755.80 | 3856.29 | 3903.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:15:00 | 3747.10 | 3845.99 | 3896.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:30:00 | 3774.00 | 3843.63 | 3894.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:15:00 | 3783.90 | 3841.65 | 3893.00 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 15:15:00 | 3568.01 | 3822.31 | 3875.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 15:15:00 | 3585.30 | 3822.31 | 3875.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 15:15:00 | 3594.70 | 3822.31 | 3875.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 3559.74 | 3819.93 | 3874.50 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 3835.90 | 3724.32 | 3803.91 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 3835.90 | 3724.32 | 3803.91 | SL hit (close>ema200) qty=0.50 sl=3724.32 alert=retest2 |

### Cycle 6 — BUY (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 12:15:00 | 4248.30 | 3838.74 | 3838.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 4289.00 | 3847.23 | 3842.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 3982.30 | 4028.11 | 3950.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 3927.90 | 4025.45 | 3952.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 3927.90 | 4025.45 | 3952.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 3927.90 | 4025.45 | 3952.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 3877.00 | 4023.98 | 3951.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:00:00 | 3877.00 | 4023.98 | 3951.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 3989.00 | 3990.81 | 3941.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 12:45:00 | 4000.00 | 3990.43 | 3941.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 09:45:00 | 3997.40 | 3992.30 | 3943.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 4079.10 | 3990.83 | 3944.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 12:00:00 | 4017.50 | 3999.81 | 3951.16 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 3954.90 | 3999.02 | 3951.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 15:00:00 | 3954.90 | 3999.02 | 3951.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 15:15:00 | 3937.00 | 3998.40 | 3951.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 3990.80 | 3998.40 | 3951.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 3919.40 | 3997.61 | 3951.25 | SL hit (close<static) qty=1.00 sl=3922.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2026-01-09 15:00:00 | 3755.80 | 2026-01-20 15:15:00 | 3568.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 09:15:00 | 3747.10 | 2026-01-20 15:15:00 | 3585.30 | PARTIAL | 0.50 | 4.32% |
| SELL | retest2 | 2026-01-13 11:30:00 | 3774.00 | 2026-01-20 15:15:00 | 3594.70 | PARTIAL | 0.50 | 4.75% |
| SELL | retest2 | 2026-01-13 15:15:00 | 3783.90 | 2026-01-21 09:15:00 | 3559.74 | PARTIAL | 0.50 | 5.92% |
| SELL | retest2 | 2026-01-09 15:00:00 | 3755.80 | 2026-02-03 09:15:00 | 3835.90 | STOP_HIT | 0.50 | -2.13% |
| SELL | retest2 | 2026-01-13 09:15:00 | 3747.10 | 2026-02-03 09:15:00 | 3835.90 | STOP_HIT | 0.50 | -2.37% |
| SELL | retest2 | 2026-01-13 11:30:00 | 3774.00 | 2026-02-03 09:15:00 | 3835.90 | STOP_HIT | 0.50 | -1.64% |
| SELL | retest2 | 2026-01-13 15:15:00 | 3783.90 | 2026-02-03 09:15:00 | 3835.90 | STOP_HIT | 0.50 | -1.37% |
| SELL | retest2 | 2026-02-13 09:15:00 | 3730.70 | 2026-02-16 10:15:00 | 3829.20 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2026-02-13 15:00:00 | 3767.60 | 2026-02-16 10:15:00 | 3829.20 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2026-02-16 15:15:00 | 3767.00 | 2026-02-18 14:15:00 | 3885.30 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2026-02-18 10:00:00 | 3762.30 | 2026-02-18 14:15:00 | 3885.30 | STOP_HIT | 1.00 | -3.27% |
| BUY | retest2 | 2026-03-18 12:45:00 | 4000.00 | 2026-03-24 09:15:00 | 3919.40 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2026-03-19 09:45:00 | 3997.40 | 2026-03-24 10:15:00 | 3904.10 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-03-20 09:15:00 | 4079.10 | 2026-03-24 10:15:00 | 3904.10 | STOP_HIT | 1.00 | -4.29% |
| BUY | retest2 | 2026-03-23 12:00:00 | 4017.50 | 2026-03-24 10:15:00 | 3904.10 | STOP_HIT | 1.00 | -2.82% |
| BUY | retest2 | 2026-03-24 09:15:00 | 3990.80 | 2026-03-24 10:15:00 | 3904.10 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2026-03-24 14:00:00 | 3978.30 | 2026-03-30 09:15:00 | 3865.00 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2026-04-01 10:00:00 | 3978.20 | 2026-04-01 10:15:00 | 3882.20 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2026-04-08 12:00:00 | 3970.50 | 2026-04-08 13:15:00 | 3902.00 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2026-04-16 14:30:00 | 3972.50 | 2026-04-27 10:15:00 | 4369.75 | TARGET_HIT | 1.00 | 10.00% |
