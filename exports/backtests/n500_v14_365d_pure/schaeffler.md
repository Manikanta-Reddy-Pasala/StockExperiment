# Schaeffler India Ltd. (SCHAEFFLER)

## Backtest Summary

- **Window:** 2024-04-05 09:15:00 → 2026-05-08 15:15:00 (3605 bars)
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
| ALERT1 | 4 |
| ALERT2 | 5 |
| ALERT2_SKIP | 3 |
| ALERT3 | 43 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 25 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 24 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 29 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 5 / 24
- **Target hits / Stop hits / Partials:** 1 / 24 / 4
- **Avg / median % per leg:** -0.83% / -2.02%
- **Sum % (uncompounded):** -23.96%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 1 | 7.7% | 1 | 12 | 0 | -1.28% | -16.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 1 | 7.7% | 1 | 12 | 0 | -1.28% | -16.6% |
| SELL (all) | 16 | 4 | 25.0% | 0 | 12 | 4 | -0.46% | -7.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 4 | 25.0% | 0 | 12 | 4 | -0.46% | -7.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 29 | 5 | 17.2% | 1 | 24 | 4 | -0.83% | -24.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 11:15:00 | 3867.00 | 3970.51 | 3970.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 13:15:00 | 3858.00 | 3968.31 | 3969.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 3951.60 | 3935.94 | 3951.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 3951.60 | 3935.94 | 3951.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 3951.60 | 3935.94 | 3951.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:45:00 | 3959.00 | 3935.94 | 3951.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 10:15:00 | 3981.00 | 3936.39 | 3952.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 10:45:00 | 3981.40 | 3936.39 | 3952.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 11:15:00 | 4001.40 | 3937.03 | 3952.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 12:00:00 | 4001.40 | 3937.03 | 3952.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 12:15:00 | 3960.60 | 3937.27 | 3952.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 14:30:00 | 3952.90 | 3937.70 | 3952.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 15:00:00 | 3952.30 | 3937.70 | 3952.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 4015.20 | 3940.57 | 3953.52 | SL hit (close>static) qty=1.00 sl=4008.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-11 12:15:00 | 4015.20 | 3940.57 | 3953.52 | SL hit (close>static) qty=1.00 sl=4008.70 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 09:45:00 | 3952.10 | 3941.77 | 3953.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 11:15:00 | 3954.60 | 3942.00 | 3953.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 3926.10 | 3941.84 | 3953.78 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-16 09:15:00 | 4071.50 | 3946.81 | 3955.62 | SL hit (close>static) qty=1.00 sl=4008.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-16 09:15:00 | 4071.50 | 3946.81 | 3955.62 | SL hit (close>static) qty=1.00 sl=4008.70 alert=retest2 |

### Cycle 2 — BUY (started 2025-09-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 15:15:00 | 4129.00 | 3964.19 | 3964.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 4146.40 | 3966.00 | 3964.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 13:15:00 | 4012.80 | 4016.96 | 3993.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 13:15:00 | 4012.80 | 4016.96 | 3993.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 13:15:00 | 4012.80 | 4016.96 | 3993.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 14:00:00 | 4012.80 | 4016.96 | 3993.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 3992.70 | 4016.72 | 3993.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 15:00:00 | 3992.70 | 4016.72 | 3993.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 3995.30 | 4016.51 | 3993.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:30:00 | 3974.60 | 4016.08 | 3993.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 3966.90 | 4015.60 | 3993.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-29 10:30:00 | 3961.10 | 4015.60 | 3993.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 4049.90 | 4094.07 | 4044.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:45:00 | 4050.20 | 4094.07 | 4044.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 4028.70 | 4093.42 | 4044.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 13:00:00 | 4028.70 | 4093.42 | 4044.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 4025.60 | 4092.75 | 4044.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:00:00 | 4025.60 | 4092.75 | 4044.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-10-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 14:15:00 | 3869.90 | 4010.42 | 4010.91 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-11-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 14:15:00 | 4294.00 | 4011.71 | 4010.91 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 09:15:00 | 3860.90 | 4028.20 | 4029.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 10:15:00 | 3800.80 | 3983.08 | 4003.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 14:15:00 | 3903.60 | 3901.73 | 3949.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-22 15:00:00 | 3903.60 | 3901.73 | 3949.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 3807.70 | 3859.49 | 3907.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 15:00:00 | 3755.80 | 3856.29 | 3904.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:15:00 | 3747.10 | 3845.99 | 3897.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:30:00 | 3774.00 | 3843.63 | 3895.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:15:00 | 3783.90 | 3841.65 | 3893.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 15:15:00 | 3568.01 | 3822.31 | 3876.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 15:15:00 | 3585.30 | 3822.31 | 3876.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 15:15:00 | 3594.70 | 3822.31 | 3876.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 3559.74 | 3819.93 | 3875.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 3835.90 | 3724.32 | 3804.29 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 3835.90 | 3724.32 | 3804.29 | SL hit (close>ema200) qty=0.50 sl=3724.32 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 3835.90 | 3724.32 | 3804.29 | SL hit (close>ema200) qty=0.50 sl=3724.32 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 3835.90 | 3724.32 | 3804.29 | SL hit (close>ema200) qty=0.50 sl=3724.32 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 3835.90 | 3724.32 | 3804.29 | SL hit (close>ema200) qty=0.50 sl=3724.32 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-02-03 10:00:00 | 3835.90 | 3724.32 | 3804.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 10:15:00 | 3845.90 | 3725.53 | 3804.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 11:15:00 | 3853.70 | 3725.53 | 3804.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 14:15:00 | 3849.90 | 3740.10 | 3807.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-04 15:00:00 | 3849.90 | 3740.10 | 3807.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 15:15:00 | 3839.60 | 3741.09 | 3807.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:15:00 | 3769.60 | 3741.09 | 3807.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 3812.00 | 3742.25 | 3807.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 12:45:00 | 3812.00 | 3742.25 | 3807.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 3826.00 | 3743.09 | 3807.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 13:45:00 | 3823.50 | 3743.09 | 3807.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 3847.60 | 3744.13 | 3807.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:45:00 | 3843.10 | 3744.13 | 3807.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 3791.90 | 3745.21 | 3807.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:15:00 | 3795.40 | 3745.21 | 3807.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 3784.50 | 3745.60 | 3807.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 3730.70 | 3781.91 | 3817.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 15:00:00 | 3767.60 | 3780.14 | 3815.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-16 10:15:00 | 3829.20 | 3781.05 | 3815.68 | SL hit (close>static) qty=1.00 sl=3814.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-16 10:15:00 | 3829.20 | 3781.05 | 3815.68 | SL hit (close>static) qty=1.00 sl=3814.90 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 15:15:00 | 3767.00 | 3782.36 | 3815.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-18 10:00:00 | 3762.30 | 3782.74 | 3814.39 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 3885.30 | 3782.74 | 3813.61 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-18 14:15:00 | 3885.30 | 3782.74 | 3813.61 | SL hit (close>static) qty=1.00 sl=3814.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-18 14:15:00 | 3885.30 | 3782.74 | 3813.61 | SL hit (close>static) qty=1.00 sl=3814.90 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 3885.30 | 3782.74 | 3813.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 3900.00 | 3783.91 | 3814.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 3871.00 | 3783.91 | 3814.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 3838.00 | 3789.69 | 3815.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:00:00 | 3838.00 | 3789.69 | 3815.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 3866.50 | 3790.45 | 3816.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:00:00 | 3866.50 | 3790.45 | 3816.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2026-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 12:15:00 | 4248.30 | 3838.74 | 3838.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 14:15:00 | 4289.00 | 3847.23 | 3842.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 3982.30 | 4028.11 | 3950.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 3927.90 | 4025.45 | 3952.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 3927.90 | 4025.45 | 3952.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 3927.90 | 4025.45 | 3952.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 3877.00 | 4023.98 | 3951.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:00:00 | 3877.00 | 4023.98 | 3951.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 3989.00 | 3990.81 | 3941.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-18 12:45:00 | 4000.00 | 3990.43 | 3941.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 09:45:00 | 3997.40 | 3992.30 | 3943.70 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 4079.10 | 3990.83 | 3944.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-23 12:00:00 | 4017.50 | 3999.81 | 3951.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 14:15:00 | 3954.90 | 3999.02 | 3951.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 15:00:00 | 3954.90 | 3999.02 | 3951.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 15:15:00 | 3937.00 | 3998.40 | 3951.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 3990.80 | 3998.40 | 3951.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 3919.40 | 3997.61 | 3951.36 | SL hit (close<static) qty=1.00 sl=3922.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 10:15:00 | 3904.10 | 3996.68 | 3951.13 | SL hit (close<static) qty=1.00 sl=3906.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 10:15:00 | 3904.10 | 3996.68 | 3951.13 | SL hit (close<static) qty=1.00 sl=3906.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 10:15:00 | 3904.10 | 3996.68 | 3951.13 | SL hit (close<static) qty=1.00 sl=3906.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 10:15:00 | 3904.10 | 3996.68 | 3951.13 | SL hit (close<static) qty=1.00 sl=3906.90 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 14:00:00 | 3978.30 | 3995.08 | 3950.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 3865.00 | 3998.38 | 3956.37 | SL hit (close<static) qty=1.00 sl=3922.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 10:00:00 | 3978.20 | 3989.73 | 3953.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-01 10:15:00 | 3882.20 | 3988.66 | 3953.05 | SL hit (close<static) qty=1.00 sl=3922.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 12:00:00 | 3970.50 | 3937.83 | 3930.64 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 12:15:00 | 3934.80 | 3937.80 | 3930.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-08 12:30:00 | 3944.00 | 3937.80 | 3930.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 13:15:00 | 3902.00 | 3937.45 | 3930.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-08 13:15:00 | 3902.00 | 3937.45 | 3930.51 | SL hit (close<static) qty=1.00 sl=3922.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2026-04-08 13:45:00 | 3902.30 | 3937.45 | 3930.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 14:15:00 | 3886.90 | 3936.94 | 3930.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-08 15:00:00 | 3886.90 | 3936.94 | 3930.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 3897.40 | 3930.22 | 3927.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-10 11:00:00 | 3897.40 | 3930.22 | 3927.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 3918.00 | 3928.50 | 3926.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 13:00:00 | 3918.00 | 3928.50 | 3926.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 3952.90 | 3928.75 | 3926.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 14:30:00 | 3972.50 | 3929.10 | 3926.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-27 10:15:00 | 4369.75 | 4021.26 | 3977.17 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-07-29 14:15:00 | 4041.00 | 2025-08-08 12:15:00 | 3955.20 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-08-07 14:45:00 | 4007.70 | 2025-08-08 12:15:00 | 3955.20 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-08-21 09:15:00 | 4028.00 | 2025-08-22 09:15:00 | 3966.70 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-08-21 14:00:00 | 4008.00 | 2025-08-22 09:15:00 | 3966.70 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-09-10 14:30:00 | 3952.90 | 2025-09-11 12:15:00 | 4015.20 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-09-10 15:00:00 | 3952.30 | 2025-09-11 12:15:00 | 4015.20 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-09-12 09:45:00 | 3952.10 | 2025-09-16 09:15:00 | 4071.50 | STOP_HIT | 1.00 | -3.02% |
| SELL | retest2 | 2025-09-12 11:15:00 | 3954.60 | 2025-09-16 09:15:00 | 4071.50 | STOP_HIT | 1.00 | -2.96% |
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
