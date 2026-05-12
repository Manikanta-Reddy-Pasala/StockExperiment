# Tube Investments of India Ltd. (TIINDIA)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3709 bars)
- **Last close:** 3032.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 154 |
| ALERT1 | 109 |
| ALERT2 | 108 |
| ALERT2_SKIP | 51 |
| ALERT3 | 298 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 9 |
| ENTRY2 | 132 |
| PARTIAL | 27 |
| TARGET_HIT | 18 |
| STOP_HIT | 123 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 168 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 75 / 93
- **Target hits / Stop hits / Partials:** 18 / 123 / 27
- **Avg / median % per leg:** 1.59% / -0.54%
- **Sum % (uncompounded):** 267.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 71 | 30 | 42.3% | 13 | 53 | 5 | 1.74% | 123.3% |
| BUY @ 2nd Alert (retest1) | 14 | 14 | 100.0% | 0 | 9 | 5 | 3.60% | 50.4% |
| BUY @ 3rd Alert (retest2) | 57 | 16 | 28.1% | 13 | 44 | 0 | 1.28% | 72.8% |
| SELL (all) | 97 | 45 | 46.4% | 5 | 70 | 22 | 1.48% | 143.7% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 97 | 45 | 46.4% | 5 | 70 | 22 | 1.48% | 143.7% |
| retest1 (combined) | 14 | 14 | 100.0% | 0 | 9 | 5 | 3.60% | 50.4% |
| retest2 (combined) | 154 | 61 | 39.6% | 18 | 114 | 22 | 1.41% | 216.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 09:15:00 | 3785.00 | 3911.85 | 3926.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-14 13:15:00 | 3780.00 | 3842.63 | 3884.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-15 12:15:00 | 3797.50 | 3790.49 | 3834.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-15 13:00:00 | 3797.50 | 3790.49 | 3834.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 09:15:00 | 3779.05 | 3787.86 | 3819.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 11:00:00 | 3745.00 | 3779.29 | 3812.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 12:30:00 | 3745.05 | 3764.20 | 3799.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-16 15:15:00 | 3751.00 | 3754.90 | 3788.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-21 09:15:00 | 3734.00 | 3792.29 | 3792.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 3726.45 | 3695.07 | 3712.05 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-05-23 11:15:00 | 3787.95 | 3723.24 | 3722.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 11:15:00 | 3787.95 | 3723.24 | 3722.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 14:15:00 | 3817.15 | 3761.46 | 3741.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 14:15:00 | 3812.90 | 3818.18 | 3786.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-24 15:00:00 | 3812.90 | 3818.18 | 3786.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 14:15:00 | 3800.00 | 3817.14 | 3800.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 15:00:00 | 3800.00 | 3817.14 | 3800.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 15:15:00 | 3804.00 | 3814.51 | 3801.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 09:15:00 | 3831.40 | 3814.51 | 3801.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 11:15:00 | 3779.25 | 3799.71 | 3797.13 | SL hit (close<static) qty=1.00 sl=3782.20 alert=retest2 |

### Cycle 3 — SELL (started 2024-05-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 12:15:00 | 3765.80 | 3792.93 | 3794.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 13:15:00 | 3737.05 | 3781.75 | 3789.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-28 14:15:00 | 3807.30 | 3786.86 | 3790.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 14:15:00 | 3807.30 | 3786.86 | 3790.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 3807.30 | 3786.86 | 3790.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-28 15:00:00 | 3807.30 | 3786.86 | 3790.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 15:15:00 | 3780.15 | 3785.52 | 3789.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 09:30:00 | 3772.55 | 3779.89 | 3786.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-31 09:15:00 | 3583.92 | 3708.12 | 3728.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-31 09:15:00 | 3710.50 | 3708.12 | 3728.85 | SL hit (close>static) qty=0.50 sl=3708.12 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 13:15:00 | 3772.30 | 3705.81 | 3705.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 14:15:00 | 3804.55 | 3725.56 | 3714.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 3695.00 | 3758.60 | 3736.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 3695.00 | 3758.60 | 3736.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 3695.00 | 3758.60 | 3736.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 3695.00 | 3758.60 | 3736.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 3700.00 | 3746.88 | 3732.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:15:00 | 3674.80 | 3746.88 | 3732.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 12:15:00 | 3655.05 | 3728.52 | 3725.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 12:30:00 | 3620.25 | 3728.52 | 3725.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2024-06-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 13:15:00 | 3650.55 | 3712.92 | 3718.90 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 14:15:00 | 3819.70 | 3734.28 | 3728.06 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-06-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-05 09:15:00 | 3624.65 | 3722.87 | 3724.61 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 3858.90 | 3733.45 | 3722.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 09:15:00 | 3936.95 | 3848.67 | 3797.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 3931.50 | 3951.09 | 3907.83 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 09:15:00 | 3989.90 | 3951.09 | 3907.83 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 10:30:00 | 3999.50 | 3967.29 | 3923.00 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 09:15:00 | 4189.40 | 4057.15 | 3990.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-12 09:15:00 | 4199.48 | 4057.15 | 3990.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2024-06-12 13:15:00 | 4099.15 | 4102.40 | 4037.76 | SL hit (close<ema200) qty=0.50 sl=4102.40 alert=retest1 |

### Cycle 9 — SELL (started 2024-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 14:15:00 | 4241.40 | 4299.98 | 4300.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 09:15:00 | 4110.00 | 4246.79 | 4275.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 11:15:00 | 4107.50 | 4090.18 | 4156.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-24 12:00:00 | 4107.50 | 4090.18 | 4156.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 4260.80 | 4124.30 | 4166.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 13:00:00 | 4260.80 | 4124.30 | 4166.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 13:15:00 | 4216.25 | 4142.69 | 4170.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 13:30:00 | 4240.00 | 4142.69 | 4170.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 4192.65 | 4182.30 | 4184.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 10:45:00 | 4199.85 | 4182.30 | 4184.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2024-06-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 11:15:00 | 4214.70 | 4188.78 | 4187.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 11:15:00 | 4225.00 | 4204.21 | 4196.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 09:15:00 | 4197.05 | 4214.86 | 4206.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 09:15:00 | 4197.05 | 4214.86 | 4206.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 09:15:00 | 4197.05 | 4214.86 | 4206.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 10:00:00 | 4197.05 | 4214.86 | 4206.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 4203.60 | 4212.60 | 4206.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-27 11:00:00 | 4203.60 | 4212.60 | 4206.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2024-06-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 13:15:00 | 4165.00 | 4197.36 | 4200.23 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2024-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 11:15:00 | 4248.40 | 4207.14 | 4203.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 14:15:00 | 4252.80 | 4221.87 | 4211.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 09:15:00 | 4203.00 | 4222.60 | 4214.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 09:15:00 | 4203.00 | 4222.60 | 4214.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 4203.00 | 4222.60 | 4214.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 12:15:00 | 4256.60 | 4222.07 | 4215.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 10:15:00 | 4203.00 | 4213.03 | 4214.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2024-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 10:15:00 | 4203.00 | 4213.03 | 4214.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 12:15:00 | 4158.40 | 4201.69 | 4208.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 14:15:00 | 4212.25 | 4164.23 | 4176.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 14:15:00 | 4212.25 | 4164.23 | 4176.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 4212.25 | 4164.23 | 4176.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 15:00:00 | 4212.25 | 4164.23 | 4176.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 15:15:00 | 4220.00 | 4175.39 | 4180.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 09:15:00 | 4184.35 | 4175.39 | 4180.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 14:15:00 | 4338.55 | 4210.43 | 4194.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2024-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-04 14:15:00 | 4338.55 | 4210.43 | 4194.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 14:15:00 | 4464.55 | 4348.81 | 4315.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 09:15:00 | 4371.70 | 4399.50 | 4347.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-11 09:30:00 | 4389.65 | 4399.50 | 4347.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 4301.40 | 4371.52 | 4346.87 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2024-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-11 14:15:00 | 4239.75 | 4330.33 | 4331.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-11 15:15:00 | 4228.95 | 4310.05 | 4322.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 09:15:00 | 4130.00 | 4107.51 | 4161.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-16 09:30:00 | 4134.45 | 4107.51 | 4161.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 4066.10 | 4087.26 | 4123.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 10:15:00 | 4064.40 | 4087.26 | 4123.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 12:45:00 | 4060.90 | 4074.29 | 4108.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 14:45:00 | 4053.30 | 4075.38 | 4102.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 09:15:00 | 4061.45 | 4077.11 | 4100.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 4019.55 | 4065.59 | 4093.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 10:45:00 | 3994.90 | 4052.89 | 4085.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 11:30:00 | 4004.95 | 4045.44 | 4078.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 12:45:00 | 3996.15 | 4037.40 | 4072.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 14:15:00 | 4001.35 | 4031.57 | 4066.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 4044.20 | 4025.40 | 4050.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 4065.75 | 4025.40 | 4050.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 4039.75 | 4028.27 | 4049.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:30:00 | 4045.80 | 4028.27 | 4049.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 4048.90 | 4032.40 | 4049.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:45:00 | 4047.45 | 4032.40 | 4049.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 4038.90 | 4033.70 | 4048.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:30:00 | 4067.95 | 4033.70 | 4048.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 4119.85 | 4050.93 | 4055.28 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-07-22 14:15:00 | 4119.85 | 4050.93 | 4055.28 | SL hit (close>static) qty=1.00 sl=4095.80 alert=retest2 |

### Cycle 16 — BUY (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 09:15:00 | 4093.45 | 4064.88 | 4061.20 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 12:15:00 | 4003.55 | 4052.93 | 4056.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 09:15:00 | 3932.70 | 3971.81 | 4003.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-25 11:15:00 | 4038.70 | 3983.78 | 4003.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 11:15:00 | 4038.70 | 3983.78 | 4003.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 4038.70 | 3983.78 | 4003.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 11:45:00 | 4033.50 | 3983.78 | 4003.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 4038.60 | 3994.74 | 4006.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 12:45:00 | 4077.00 | 3994.74 | 4006.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 4031.75 | 4012.84 | 4012.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:15:00 | 4123.95 | 4012.84 | 4012.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 4155.95 | 4041.46 | 4025.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 13:15:00 | 4163.55 | 4101.61 | 4062.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 14:15:00 | 4153.80 | 4162.05 | 4121.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 15:00:00 | 4153.80 | 4162.05 | 4121.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 15:15:00 | 4134.00 | 4156.44 | 4122.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 09:15:00 | 4224.00 | 4156.44 | 4122.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 14:15:00 | 4161.30 | 4169.77 | 4144.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 09:15:00 | 4156.95 | 4154.58 | 4150.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 10:15:00 | 4124.00 | 4145.81 | 4147.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 10:15:00 | 4124.00 | 4145.81 | 4147.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 11:15:00 | 4116.65 | 4139.98 | 4144.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-05 14:15:00 | 4003.85 | 3970.34 | 4023.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-05 15:00:00 | 4003.85 | 3970.34 | 4023.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 15:15:00 | 4067.00 | 3989.67 | 4027.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:00:00 | 3979.00 | 4019.66 | 4032.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 13:15:00 | 4062.25 | 4039.88 | 4037.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2024-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 13:15:00 | 4062.25 | 4039.88 | 4037.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 14:15:00 | 4108.65 | 4053.64 | 4043.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 10:15:00 | 4055.50 | 4066.06 | 4052.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 10:15:00 | 4055.50 | 4066.06 | 4052.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 10:15:00 | 4055.50 | 4066.06 | 4052.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 11:00:00 | 4055.50 | 4066.06 | 4052.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 11:15:00 | 4023.65 | 4057.57 | 4050.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 12:00:00 | 4023.65 | 4057.57 | 4050.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 4030.25 | 4052.11 | 4048.44 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2024-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 13:15:00 | 4003.85 | 4042.46 | 4044.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 15:15:00 | 3999.40 | 4018.75 | 4028.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 09:15:00 | 3997.85 | 3952.24 | 3977.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 09:15:00 | 3997.85 | 3952.24 | 3977.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 3997.85 | 3952.24 | 3977.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:00:00 | 3997.85 | 3952.24 | 3977.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 4026.50 | 3967.09 | 3982.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 11:00:00 | 4026.50 | 3967.09 | 3982.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2024-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 13:15:00 | 4013.40 | 3992.42 | 3991.62 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2024-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 14:15:00 | 3970.20 | 3987.98 | 3989.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 09:15:00 | 3918.85 | 3970.16 | 3981.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 3992.95 | 3948.17 | 3959.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 3992.95 | 3948.17 | 3959.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 3992.95 | 3948.17 | 3959.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 09:45:00 | 3981.00 | 3948.17 | 3959.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 3990.00 | 3956.54 | 3962.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 11:00:00 | 3990.00 | 3956.54 | 3962.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2024-08-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 11:15:00 | 4019.30 | 3969.09 | 3967.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 09:15:00 | 4042.85 | 4001.13 | 3985.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 13:15:00 | 3990.00 | 4004.48 | 3992.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 13:15:00 | 3990.00 | 4004.48 | 3992.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 13:15:00 | 3990.00 | 4004.48 | 3992.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 13:30:00 | 3983.25 | 4004.48 | 3992.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 14:15:00 | 3985.90 | 4000.76 | 3992.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 14:30:00 | 3982.95 | 4000.76 | 3992.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 4006.00 | 3996.86 | 3992.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:00:00 | 4006.00 | 3996.86 | 3992.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 12:15:00 | 4056.00 | 4060.32 | 4037.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 12:30:00 | 4036.25 | 4060.32 | 4037.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 14:15:00 | 4056.80 | 4060.60 | 4041.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 14:30:00 | 4040.05 | 4060.60 | 4041.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 15:15:00 | 4054.90 | 4059.46 | 4042.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 09:15:00 | 4090.50 | 4059.46 | 4042.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:30:00 | 4076.95 | 4084.04 | 4080.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 10:15:00 | 4055.70 | 4113.14 | 4118.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 4055.70 | 4113.14 | 4118.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 11:15:00 | 4017.35 | 4093.98 | 4109.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 14:15:00 | 4049.65 | 4025.98 | 4050.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 14:15:00 | 4049.65 | 4025.98 | 4050.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 4049.65 | 4025.98 | 4050.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 15:00:00 | 4049.65 | 4025.98 | 4050.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 4034.00 | 4027.59 | 4049.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:15:00 | 4074.70 | 4027.59 | 4049.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 4060.20 | 4034.11 | 4050.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:30:00 | 4079.05 | 4034.11 | 4050.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 10:15:00 | 4062.70 | 4039.83 | 4051.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 10:45:00 | 4069.45 | 4039.83 | 4051.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 4076.95 | 4047.25 | 4053.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 12:00:00 | 4076.95 | 4047.25 | 4053.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2024-09-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 12:15:00 | 4105.45 | 4058.89 | 4058.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 09:15:00 | 4128.65 | 4088.42 | 4073.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 14:15:00 | 4073.95 | 4102.77 | 4088.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 14:15:00 | 4073.95 | 4102.77 | 4088.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 4073.95 | 4102.77 | 4088.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 4073.95 | 4102.77 | 4088.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 4088.00 | 4099.81 | 4088.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 4085.00 | 4099.81 | 4088.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 4082.00 | 4096.25 | 4087.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 12:30:00 | 4108.45 | 4096.87 | 4089.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 15:00:00 | 4124.20 | 4102.56 | 4093.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-05 14:30:00 | 4113.90 | 4094.62 | 4094.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-05 15:15:00 | 4065.00 | 4088.69 | 4091.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2024-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 15:15:00 | 4065.00 | 4088.69 | 4091.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 10:15:00 | 4009.25 | 4070.94 | 4083.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 3925.95 | 3916.00 | 3962.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 11:15:00 | 3962.00 | 3926.53 | 3959.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 11:15:00 | 3962.00 | 3926.53 | 3959.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 12:00:00 | 3962.00 | 3926.53 | 3959.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 12:15:00 | 3960.15 | 3933.25 | 3959.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 12:45:00 | 3955.10 | 3933.25 | 3959.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 13:15:00 | 3939.90 | 3934.58 | 3957.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-10 15:00:00 | 3921.30 | 3931.92 | 3954.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 13:30:00 | 3928.00 | 3935.02 | 3946.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 10:15:00 | 3931.55 | 3932.32 | 3941.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-13 09:15:00 | 4025.10 | 3955.09 | 3947.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2024-09-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 09:15:00 | 4025.10 | 3955.09 | 3947.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 12:15:00 | 4119.00 | 4010.06 | 3976.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 12:15:00 | 4035.30 | 4064.18 | 4029.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 13:00:00 | 4035.30 | 4064.18 | 4029.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 13:15:00 | 4025.80 | 4056.50 | 4028.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-16 14:00:00 | 4025.80 | 4056.50 | 4028.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 14:15:00 | 4046.35 | 4054.47 | 4030.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 09:15:00 | 4112.00 | 4050.28 | 4030.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 10:30:00 | 4068.95 | 4098.66 | 4077.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 13:15:00 | 4022.50 | 4063.64 | 4065.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2024-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 13:15:00 | 4022.50 | 4063.64 | 4065.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 09:15:00 | 4002.50 | 4039.37 | 4052.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 4030.60 | 3998.48 | 4019.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 4030.60 | 3998.48 | 4019.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 4030.60 | 3998.48 | 4019.04 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2024-09-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 12:15:00 | 4102.05 | 4037.34 | 4033.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 14:15:00 | 4369.85 | 4108.83 | 4066.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 11:15:00 | 4257.50 | 4258.58 | 4197.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-24 12:00:00 | 4257.50 | 4258.58 | 4197.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 4200.10 | 4238.54 | 4210.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 4200.10 | 4238.54 | 4210.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 4199.95 | 4230.82 | 4209.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:15:00 | 4194.90 | 4230.82 | 4209.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2024-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 13:15:00 | 4142.95 | 4192.59 | 4195.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 09:15:00 | 4123.80 | 4166.50 | 4181.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 14:15:00 | 4132.40 | 4130.66 | 4155.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 15:00:00 | 4132.40 | 4130.66 | 4155.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 4108.00 | 4126.12 | 4150.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:30:00 | 4162.60 | 4132.84 | 4151.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 4184.00 | 4143.07 | 4154.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:00:00 | 4184.00 | 4143.07 | 4154.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 11:15:00 | 4187.90 | 4152.04 | 4157.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 11:45:00 | 4200.00 | 4152.04 | 4157.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2024-09-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 13:15:00 | 4184.00 | 4164.82 | 4162.94 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 4133.75 | 4158.61 | 4160.28 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2024-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 09:15:00 | 4225.00 | 4170.51 | 4165.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 10:15:00 | 4250.30 | 4186.47 | 4173.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-01 11:15:00 | 4271.10 | 4272.21 | 4235.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-01 11:45:00 | 4262.35 | 4272.21 | 4235.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 4287.40 | 4299.10 | 4265.06 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2024-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 13:15:00 | 4171.40 | 4240.75 | 4246.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 09:15:00 | 4130.65 | 4201.03 | 4225.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-07 14:15:00 | 4022.75 | 4021.27 | 4088.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-07 15:00:00 | 4022.75 | 4021.27 | 4088.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 4065.90 | 4034.90 | 4074.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:45:00 | 4058.35 | 4034.90 | 4074.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 4049.60 | 4033.13 | 4063.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 4049.60 | 4033.13 | 4063.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 4057.90 | 4038.09 | 4062.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 4135.05 | 4038.09 | 4062.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 4208.50 | 4072.17 | 4075.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 4208.50 | 4072.17 | 4075.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 4297.15 | 4117.16 | 4096.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 14:15:00 | 4329.30 | 4267.98 | 4230.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-17 09:15:00 | 4432.40 | 4437.93 | 4361.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-17 10:00:00 | 4432.40 | 4437.93 | 4361.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 4490.95 | 4454.30 | 4406.28 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2024-10-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 15:15:00 | 4378.05 | 4431.78 | 4434.05 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2024-10-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-22 10:15:00 | 4447.00 | 4436.23 | 4435.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-22 11:15:00 | 4489.00 | 4446.78 | 4440.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-23 10:15:00 | 4454.45 | 4465.16 | 4454.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-23 10:15:00 | 4454.45 | 4465.16 | 4454.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 4454.45 | 4465.16 | 4454.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:45:00 | 4453.30 | 4465.16 | 4454.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 4452.90 | 4462.71 | 4454.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 11:30:00 | 4461.50 | 4462.71 | 4454.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 4461.50 | 4462.47 | 4455.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-23 13:15:00 | 4473.25 | 4462.47 | 4455.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-29 10:15:00 | 4510.85 | 4605.55 | 4607.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2024-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 10:15:00 | 4510.85 | 4605.55 | 4607.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-30 14:15:00 | 4449.40 | 4515.00 | 4547.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-31 14:15:00 | 4504.50 | 4467.52 | 4501.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 14:15:00 | 4504.50 | 4467.52 | 4501.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 14:15:00 | 4504.50 | 4467.52 | 4501.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-31 15:00:00 | 4504.50 | 4467.52 | 4501.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 15:15:00 | 4470.00 | 4468.02 | 4498.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-01 18:00:00 | 4513.00 | 4477.01 | 4499.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-01 18:15:00 | 4442.00 | 4470.01 | 4494.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-04 09:15:00 | 4495.75 | 4470.01 | 4494.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 4437.05 | 4463.42 | 4489.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 10:15:00 | 4409.85 | 4463.42 | 4489.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 14:15:00 | 4399.95 | 4429.62 | 4462.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-04 14:45:00 | 4395.95 | 4412.55 | 4451.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 09:15:00 | 4189.36 | 4350.23 | 4415.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 09:15:00 | 4179.95 | 4350.23 | 4415.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-05 09:15:00 | 4176.15 | 4350.23 | 4415.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-11-08 09:15:00 | 3968.87 | 4003.33 | 4087.76 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 40 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 3597.00 | 3515.18 | 3511.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 13:15:00 | 3634.45 | 3539.04 | 3522.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 13:15:00 | 3560.05 | 3571.73 | 3551.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-21 14:00:00 | 3560.05 | 3571.73 | 3551.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 14:15:00 | 3544.55 | 3566.29 | 3551.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 15:00:00 | 3544.55 | 3566.29 | 3551.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 15:15:00 | 3530.00 | 3559.03 | 3549.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-22 09:15:00 | 3564.55 | 3559.03 | 3549.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 3547.95 | 3556.82 | 3549.03 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2024-11-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 12:15:00 | 3499.00 | 3538.89 | 3542.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-22 14:15:00 | 3471.70 | 3519.48 | 3532.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-26 09:15:00 | 3465.00 | 3461.19 | 3490.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 09:15:00 | 3465.00 | 3461.19 | 3490.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 3465.00 | 3461.19 | 3490.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 09:15:00 | 3376.55 | 3454.16 | 3474.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 09:15:00 | 3550.30 | 3423.61 | 3436.65 | SL hit (close>static) qty=1.00 sl=3524.75 alert=retest2 |

### Cycle 42 — BUY (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 10:15:00 | 3613.30 | 3461.55 | 3452.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 09:15:00 | 3679.90 | 3571.43 | 3554.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 15:15:00 | 3607.20 | 3611.73 | 3586.95 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 09:45:00 | 3631.90 | 3614.49 | 3590.46 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 10:30:00 | 3631.70 | 3618.17 | 3594.31 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 11:00:00 | 3632.90 | 3618.17 | 3594.31 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-12-04 12:30:00 | 3636.45 | 3623.33 | 3600.90 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 3697.90 | 3712.71 | 3681.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 09:30:00 | 3712.00 | 3694.51 | 3684.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-09 12:30:00 | 3723.40 | 3708.57 | 3693.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-10 12:15:00 | 3685.50 | 3715.04 | 3706.06 | SL hit (close<ema400) qty=1.00 sl=3706.06 alert=retest1 |

### Cycle 43 — SELL (started 2024-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 12:15:00 | 3689.40 | 3703.97 | 3704.01 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2024-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 13:15:00 | 3714.80 | 3706.14 | 3704.99 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 14:15:00 | 3690.10 | 3702.93 | 3703.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 09:15:00 | 3679.70 | 3698.61 | 3701.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 13:15:00 | 3645.00 | 3639.00 | 3660.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 14:00:00 | 3645.00 | 3639.00 | 3660.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 3696.50 | 3655.96 | 3663.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:00:00 | 3696.50 | 3655.96 | 3663.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 3713.15 | 3667.40 | 3667.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:00:00 | 3713.15 | 3667.40 | 3667.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2024-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 11:15:00 | 3731.55 | 3680.23 | 3673.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-16 12:15:00 | 3751.00 | 3694.38 | 3680.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 12:15:00 | 3705.05 | 3725.48 | 3707.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 12:15:00 | 3705.05 | 3725.48 | 3707.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 3705.05 | 3725.48 | 3707.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 12:30:00 | 3704.90 | 3725.48 | 3707.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 3683.30 | 3717.04 | 3705.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:00:00 | 3683.30 | 3717.04 | 3705.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 3692.40 | 3712.11 | 3704.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:30:00 | 3677.80 | 3712.11 | 3704.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 3674.95 | 3707.55 | 3703.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-18 10:15:00 | 3678.15 | 3707.55 | 3703.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2024-12-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 10:15:00 | 3663.75 | 3698.79 | 3700.30 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-12-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 12:15:00 | 3730.95 | 3706.38 | 3703.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-18 14:15:00 | 3746.05 | 3719.58 | 3710.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 09:15:00 | 3711.25 | 3723.89 | 3714.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 09:15:00 | 3711.25 | 3723.89 | 3714.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 3711.25 | 3723.89 | 3714.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 10:30:00 | 3726.55 | 3722.27 | 3714.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 11:30:00 | 3732.65 | 3725.40 | 3716.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 12:15:00 | 3627.45 | 3716.58 | 3722.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2024-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 12:15:00 | 3627.45 | 3716.58 | 3722.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 3610.85 | 3685.04 | 3706.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 10:15:00 | 3667.00 | 3663.50 | 3689.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 11:00:00 | 3667.00 | 3663.50 | 3689.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 3603.00 | 3575.94 | 3603.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 14:00:00 | 3603.00 | 3575.94 | 3603.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 3645.85 | 3589.92 | 3607.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 3645.85 | 3589.92 | 3607.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 3615.00 | 3594.94 | 3608.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 11:30:00 | 3597.05 | 3604.99 | 3610.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 14:45:00 | 3594.90 | 3603.00 | 3608.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-30 09:30:00 | 3570.10 | 3597.26 | 3604.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 14:15:00 | 3590.50 | 3546.83 | 3563.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 3584.55 | 3554.37 | 3565.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 15:15:00 | 3553.00 | 3554.37 | 3565.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 15:15:00 | 3561.00 | 3557.98 | 3561.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 09:15:00 | 3622.35 | 3571.34 | 3567.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — BUY (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 09:15:00 | 3622.35 | 3571.34 | 3567.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 11:15:00 | 3662.10 | 3598.31 | 3580.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 12:15:00 | 3621.55 | 3636.20 | 3616.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 12:45:00 | 3623.20 | 3636.20 | 3616.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 13:15:00 | 3603.80 | 3629.72 | 3614.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 14:00:00 | 3603.80 | 3629.72 | 3614.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 3611.00 | 3625.98 | 3614.63 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 3522.90 | 3595.00 | 3602.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 12:15:00 | 3514.20 | 3570.36 | 3589.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 3479.20 | 3468.02 | 3497.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 14:00:00 | 3479.20 | 3468.02 | 3497.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 3528.40 | 3480.10 | 3499.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 15:00:00 | 3528.40 | 3480.10 | 3499.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 3497.85 | 3483.65 | 3499.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:30:00 | 3527.55 | 3500.50 | 3505.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2025-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 10:15:00 | 3625.20 | 3525.44 | 3516.76 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-01-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 12:15:00 | 3502.60 | 3532.23 | 3532.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 13:15:00 | 3478.00 | 3521.39 | 3527.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 3340.00 | 3325.68 | 3399.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 10:00:00 | 3340.00 | 3325.68 | 3399.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 15:15:00 | 3344.00 | 3316.67 | 3360.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 09:15:00 | 3302.15 | 3316.67 | 3360.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 12:15:00 | 3303.75 | 3303.76 | 3342.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 12:45:00 | 3303.80 | 3301.12 | 3338.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-16 09:15:00 | 3380.55 | 3313.11 | 3331.10 | SL hit (close>static) qty=1.00 sl=3362.00 alert=retest2 |

### Cycle 54 — BUY (started 2025-01-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 09:15:00 | 3389.60 | 3335.69 | 3334.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 11:15:00 | 3431.25 | 3384.22 | 3365.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 10:15:00 | 3399.25 | 3416.20 | 3393.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 10:15:00 | 3399.25 | 3416.20 | 3393.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 3399.25 | 3416.20 | 3393.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 3399.25 | 3416.20 | 3393.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 3411.20 | 3415.20 | 3395.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:30:00 | 3401.20 | 3415.20 | 3395.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 12:15:00 | 3398.50 | 3411.86 | 3395.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 12:30:00 | 3394.65 | 3411.86 | 3395.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 13:15:00 | 3386.50 | 3406.79 | 3394.60 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2025-01-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-22 09:15:00 | 3282.00 | 3376.50 | 3383.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 10:15:00 | 3271.40 | 3355.48 | 3373.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 3315.20 | 3282.93 | 3320.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 3315.20 | 3282.93 | 3320.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 3315.20 | 3282.93 | 3320.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 3319.35 | 3282.93 | 3320.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 3357.05 | 3297.75 | 3324.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:00:00 | 3357.05 | 3297.75 | 3324.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 11:15:00 | 3337.40 | 3305.68 | 3325.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 11:30:00 | 3371.55 | 3305.68 | 3325.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 12:15:00 | 3331.45 | 3310.83 | 3325.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 12:30:00 | 3335.35 | 3310.83 | 3325.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 3357.00 | 3320.07 | 3328.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 14:00:00 | 3357.00 | 3320.07 | 3328.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 3361.45 | 3328.34 | 3331.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 15:00:00 | 3361.45 | 3328.34 | 3331.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 3230.20 | 3159.23 | 3191.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:45:00 | 3234.15 | 3159.23 | 3191.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 3203.25 | 3168.03 | 3192.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:30:00 | 3241.65 | 3168.03 | 3192.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 3193.40 | 3173.10 | 3192.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 11:45:00 | 3194.15 | 3173.10 | 3192.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 3190.35 | 3176.55 | 3192.75 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 3249.55 | 3200.30 | 3199.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 12:15:00 | 3277.55 | 3232.35 | 3215.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 10:15:00 | 3279.90 | 3300.28 | 3276.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 10:15:00 | 3279.90 | 3300.28 | 3276.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 10:15:00 | 3279.90 | 3300.28 | 3276.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:00:00 | 3279.90 | 3300.28 | 3276.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 3271.65 | 3294.56 | 3276.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 3252.00 | 3294.56 | 3276.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 3245.05 | 3284.66 | 3273.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 3245.05 | 3284.66 | 3273.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 14:15:00 | 3174.95 | 3253.41 | 3260.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-03 09:15:00 | 3137.50 | 3217.52 | 3242.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 09:15:00 | 3099.85 | 3098.38 | 3155.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-04 09:15:00 | 3099.85 | 3098.38 | 3155.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 09:15:00 | 3099.85 | 3098.38 | 3155.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-04 11:45:00 | 3073.80 | 3098.79 | 3146.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-05 09:15:00 | 3058.85 | 3093.12 | 3128.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-07 09:15:00 | 2920.11 | 2968.70 | 3009.58 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-07 13:15:00 | 2905.91 | 2941.11 | 2982.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-11 12:15:00 | 2766.42 | 2820.97 | 2879.24 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 58 — BUY (started 2025-02-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 11:15:00 | 2673.75 | 2633.59 | 2628.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 2702.05 | 2666.70 | 2648.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 2726.80 | 2734.68 | 2699.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 2726.80 | 2734.68 | 2699.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 2691.55 | 2726.05 | 2699.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 2691.55 | 2726.05 | 2699.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 11:15:00 | 2696.15 | 2720.07 | 2698.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 12:45:00 | 2709.50 | 2715.66 | 2698.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 15:15:00 | 2681.00 | 2700.97 | 2695.46 | SL hit (close<static) qty=1.00 sl=2683.90 alert=retest2 |

### Cycle 59 — SELL (started 2025-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 09:15:00 | 2608.20 | 2682.41 | 2687.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 10:15:00 | 2592.05 | 2637.02 | 2657.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 11:15:00 | 2484.10 | 2465.53 | 2506.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 12:00:00 | 2484.10 | 2465.53 | 2506.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 2525.50 | 2477.53 | 2508.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:00:00 | 2525.50 | 2477.53 | 2508.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 2571.40 | 2496.30 | 2513.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 14:00:00 | 2571.40 | 2496.30 | 2513.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2025-03-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 14:15:00 | 2648.85 | 2526.81 | 2526.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 10:15:00 | 2702.95 | 2598.48 | 2562.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 10:15:00 | 2670.35 | 2685.03 | 2635.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-05 10:45:00 | 2673.65 | 2685.03 | 2635.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 09:15:00 | 2684.20 | 2685.83 | 2657.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 09:30:00 | 2689.00 | 2685.83 | 2657.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 10:15:00 | 2674.25 | 2683.51 | 2659.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 11:00:00 | 2674.25 | 2683.51 | 2659.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 11:15:00 | 2622.85 | 2671.38 | 2655.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 12:00:00 | 2622.85 | 2671.38 | 2655.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 12:15:00 | 2649.35 | 2666.97 | 2655.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 14:00:00 | 2655.15 | 2664.61 | 2655.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 14:30:00 | 2657.90 | 2661.80 | 2654.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-06 15:00:00 | 2650.55 | 2661.80 | 2654.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-07 09:15:00 | 2657.30 | 2658.64 | 2653.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 2845.20 | 2740.68 | 2711.11 | EMA400 retest candle locked (from upside) |
| Target hit | 2025-03-11 10:15:00 | 2920.67 | 2775.34 | 2729.56 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2025-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-17 13:15:00 | 2839.00 | 2864.42 | 2867.11 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-19 11:15:00 | 2864.10 | 2861.26 | 2860.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 09:15:00 | 2905.05 | 2871.11 | 2865.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 13:15:00 | 2874.95 | 2888.73 | 2882.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-21 13:15:00 | 2874.95 | 2888.73 | 2882.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 2874.95 | 2888.73 | 2882.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 13:45:00 | 2874.95 | 2888.73 | 2882.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 2874.85 | 2885.95 | 2881.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 15:00:00 | 2874.85 | 2885.95 | 2881.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 15:15:00 | 2872.00 | 2883.16 | 2880.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 09:15:00 | 2892.65 | 2883.16 | 2880.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 13:15:00 | 2876.00 | 2880.56 | 2880.44 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-24 13:15:00 | 2877.15 | 2879.88 | 2880.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2025-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-24 13:15:00 | 2877.15 | 2879.88 | 2880.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-24 14:15:00 | 2856.45 | 2875.19 | 2877.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 12:15:00 | 2732.05 | 2725.29 | 2757.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 12:30:00 | 2735.75 | 2725.29 | 2757.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 2780.00 | 2735.87 | 2756.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 2780.00 | 2735.87 | 2756.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 2749.90 | 2738.68 | 2756.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:15:00 | 2813.85 | 2738.68 | 2756.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 2795.45 | 2750.03 | 2759.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:30:00 | 2813.65 | 2750.03 | 2759.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 2780.05 | 2756.04 | 2761.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:30:00 | 2800.55 | 2756.04 | 2761.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 13:15:00 | 2766.75 | 2765.15 | 2764.97 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 10:15:00 | 2717.95 | 2755.82 | 2760.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 14:15:00 | 2696.20 | 2726.72 | 2744.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-02 10:15:00 | 2736.00 | 2713.58 | 2732.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-02 10:15:00 | 2736.00 | 2713.58 | 2732.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 2736.00 | 2713.58 | 2732.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:00:00 | 2736.00 | 2713.58 | 2732.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 2706.75 | 2712.22 | 2730.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 12:15:00 | 2700.20 | 2712.22 | 2730.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 10:00:00 | 2697.00 | 2713.46 | 2724.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 12:00:00 | 2692.50 | 2708.15 | 2719.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 2565.19 | 2630.36 | 2666.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 2562.15 | 2630.36 | 2666.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-04-07 09:15:00 | 2557.88 | 2630.36 | 2666.56 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-04-08 10:15:00 | 2589.80 | 2585.12 | 2617.37 | SL hit (close>ema200) qty=0.50 sl=2585.12 alert=retest2 |

### Cycle 66 — BUY (started 2025-04-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 12:15:00 | 2575.60 | 2553.12 | 2552.99 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-16 10:15:00 | 2533.60 | 2552.36 | 2553.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-16 13:15:00 | 2508.40 | 2536.76 | 2545.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-21 09:15:00 | 2511.70 | 2505.48 | 2518.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-21 09:15:00 | 2511.70 | 2505.48 | 2518.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 2511.70 | 2505.48 | 2518.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 09:45:00 | 2518.40 | 2505.48 | 2518.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 10:15:00 | 2544.90 | 2513.36 | 2520.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 10:45:00 | 2540.90 | 2513.36 | 2520.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 11:15:00 | 2538.50 | 2518.39 | 2522.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 11:30:00 | 2536.40 | 2518.39 | 2522.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2025-04-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 12:15:00 | 2560.40 | 2526.79 | 2525.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 13:15:00 | 2580.40 | 2537.51 | 2530.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 11:15:00 | 2652.70 | 2657.93 | 2633.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-24 12:00:00 | 2652.70 | 2657.93 | 2633.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 2645.60 | 2654.16 | 2635.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 13:45:00 | 2644.60 | 2654.16 | 2635.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 2590.00 | 2642.35 | 2634.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 2590.00 | 2642.35 | 2634.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 2586.90 | 2631.26 | 2630.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 11:00:00 | 2586.90 | 2631.26 | 2630.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 2582.30 | 2621.47 | 2626.21 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 14:15:00 | 2643.60 | 2618.29 | 2616.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 15:15:00 | 2650.00 | 2624.63 | 2619.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-02 11:15:00 | 2878.60 | 2879.36 | 2820.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-02 12:00:00 | 2878.60 | 2879.36 | 2820.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 2913.90 | 2945.77 | 2923.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 15:00:00 | 2913.90 | 2945.77 | 2923.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 2895.70 | 2935.75 | 2920.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 2928.80 | 2935.75 | 2920.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 2960.30 | 2932.78 | 2921.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:30:00 | 2967.80 | 2940.23 | 2925.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 12:00:00 | 2970.00 | 2940.23 | 2925.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 14:00:00 | 2966.20 | 2947.67 | 2931.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 14:30:00 | 2964.70 | 2949.12 | 2934.00 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 2951.40 | 2948.60 | 2936.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 10:45:00 | 2995.00 | 2954.22 | 2940.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-08 11:45:00 | 2963.80 | 2953.77 | 2941.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-08 13:15:00 | 2907.90 | 2942.52 | 2938.05 | SL hit (close<static) qty=1.00 sl=2917.20 alert=retest2 |

### Cycle 71 — SELL (started 2025-05-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 14:15:00 | 2889.00 | 2931.82 | 2933.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 15:15:00 | 2869.00 | 2919.25 | 2927.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 15:15:00 | 2865.70 | 2865.07 | 2888.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-12 09:15:00 | 2929.60 | 2865.07 | 2888.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 2930.00 | 2878.06 | 2892.43 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 2953.40 | 2901.85 | 2900.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 2968.90 | 2930.75 | 2915.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 10:15:00 | 3028.00 | 3029.87 | 3001.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 11:00:00 | 3028.00 | 3029.87 | 3001.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 2990.00 | 3019.81 | 3001.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 13:00:00 | 2990.00 | 3019.81 | 3001.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 2960.90 | 3008.03 | 2998.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:00:00 | 2960.90 | 3008.03 | 2998.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — SELL (started 2025-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 14:15:00 | 2902.00 | 2986.82 | 2989.40 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2025-05-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 12:15:00 | 3034.50 | 2988.97 | 2987.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 13:15:00 | 3056.40 | 3002.46 | 2993.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 3079.90 | 3115.47 | 3079.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 3079.90 | 3115.47 | 3079.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 3079.90 | 3115.47 | 3079.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:00:00 | 3079.90 | 3115.47 | 3079.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 3052.30 | 3102.84 | 3076.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:00:00 | 3052.30 | 3102.84 | 3076.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 3041.00 | 3090.47 | 3073.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:30:00 | 3051.40 | 3090.47 | 3073.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 2994.50 | 3061.76 | 3062.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 09:15:00 | 2985.60 | 3031.54 | 3047.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 09:15:00 | 3033.20 | 3014.80 | 3028.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 09:15:00 | 3033.20 | 3014.80 | 3028.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 3033.20 | 3014.80 | 3028.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 10:00:00 | 3033.20 | 3014.80 | 3028.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 3005.20 | 3012.88 | 3026.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 11:15:00 | 2990.00 | 3012.88 | 3026.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 12:45:00 | 2990.00 | 3003.08 | 3019.46 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 13:15:00 | 3046.00 | 3011.24 | 3013.31 | SL hit (close>static) qty=1.00 sl=3036.30 alert=retest2 |

### Cycle 76 — BUY (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 14:15:00 | 3044.40 | 3017.87 | 3016.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 12:15:00 | 3059.00 | 3035.57 | 3026.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 3010.00 | 3037.74 | 3031.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 3010.00 | 3037.74 | 3031.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 3010.00 | 3037.74 | 3031.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 3010.00 | 3037.74 | 3031.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 3005.20 | 3031.23 | 3028.73 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 11:15:00 | 3010.00 | 3026.98 | 3027.03 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2025-05-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 12:15:00 | 3034.40 | 3028.47 | 3027.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 14:15:00 | 3046.90 | 3032.13 | 3029.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 12:15:00 | 3029.50 | 3039.05 | 3034.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-28 12:15:00 | 3029.50 | 3039.05 | 3034.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 12:15:00 | 3029.50 | 3039.05 | 3034.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:00:00 | 3029.50 | 3039.05 | 3034.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 13:15:00 | 3032.40 | 3037.72 | 3034.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 13:30:00 | 3029.60 | 3037.72 | 3034.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 14:15:00 | 3016.70 | 3033.51 | 3032.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 15:15:00 | 3016.80 | 3033.51 | 3032.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2025-05-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 15:15:00 | 3016.80 | 3030.17 | 3031.33 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 11:15:00 | 3041.80 | 3032.76 | 3032.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 12:15:00 | 3056.20 | 3037.45 | 3034.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 14:15:00 | 3061.30 | 3063.89 | 3053.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 15:00:00 | 3061.30 | 3063.89 | 3053.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 3035.10 | 3058.13 | 3051.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:15:00 | 3055.50 | 3058.13 | 3051.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 3061.10 | 3058.73 | 3052.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 3077.40 | 3054.44 | 3052.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 11:15:00 | 3040.00 | 3051.35 | 3051.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 11:15:00 | 3040.00 | 3051.35 | 3051.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 3036.50 | 3048.38 | 3050.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 3055.30 | 3043.96 | 3047.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 3055.30 | 3043.96 | 3047.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 3055.30 | 3043.96 | 3047.02 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 3080.90 | 3051.35 | 3050.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 09:15:00 | 3104.00 | 3075.91 | 3064.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 14:15:00 | 3082.30 | 3082.31 | 3072.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 3061.50 | 3077.91 | 3072.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 3061.50 | 3077.91 | 3072.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:00:00 | 3061.50 | 3077.91 | 3072.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 3047.80 | 3071.89 | 3070.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:00:00 | 3047.80 | 3071.89 | 3070.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 11:15:00 | 3030.50 | 3063.61 | 3066.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-09 12:15:00 | 3025.80 | 3050.81 | 3058.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-10 09:15:00 | 3083.40 | 3052.54 | 3056.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 09:15:00 | 3083.40 | 3052.54 | 3056.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 3083.40 | 3052.54 | 3056.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:00:00 | 3083.40 | 3052.54 | 3056.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 3073.20 | 3056.67 | 3057.84 | EMA400 retest candle locked (from downside) |

### Cycle 84 — BUY (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 11:15:00 | 3080.40 | 3061.42 | 3059.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 13:15:00 | 3087.60 | 3069.05 | 3063.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 10:15:00 | 3059.90 | 3077.75 | 3070.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 10:15:00 | 3059.90 | 3077.75 | 3070.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 3059.90 | 3077.75 | 3070.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:00:00 | 3059.90 | 3077.75 | 3070.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 3078.40 | 3077.88 | 3071.24 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 3035.40 | 3061.51 | 3064.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 3012.00 | 3045.44 | 3056.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 3001.80 | 2960.26 | 2981.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 10:15:00 | 3001.80 | 2960.26 | 2981.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 3001.80 | 2960.26 | 2981.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:00:00 | 3001.80 | 2960.26 | 2981.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 11:15:00 | 3000.70 | 2968.35 | 2983.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 11:30:00 | 3010.80 | 2968.35 | 2983.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 2980.10 | 2981.18 | 2985.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 2993.00 | 2981.18 | 2985.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 2950.20 | 2974.98 | 2982.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 10:30:00 | 2944.90 | 2969.02 | 2979.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 14:15:00 | 2922.50 | 2906.09 | 2905.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 2922.50 | 2906.09 | 2905.54 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 15:15:00 | 2901.20 | 2905.11 | 2905.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-23 09:15:00 | 2900.80 | 2904.25 | 2904.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 2900.20 | 2882.25 | 2890.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 2900.20 | 2882.25 | 2890.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 2900.20 | 2882.25 | 2890.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 10:00:00 | 2900.20 | 2882.25 | 2890.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 2906.00 | 2887.00 | 2891.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:00:00 | 2906.00 | 2887.00 | 2891.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 2906.30 | 2890.86 | 2892.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 12:00:00 | 2906.30 | 2890.86 | 2892.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 2866.10 | 2872.40 | 2882.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 09:30:00 | 2885.00 | 2872.40 | 2882.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 10:15:00 | 2885.60 | 2875.04 | 2882.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 11:00:00 | 2885.60 | 2875.04 | 2882.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 11:15:00 | 2840.60 | 2868.15 | 2878.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 12:15:00 | 2836.30 | 2868.15 | 2878.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 13:45:00 | 2835.50 | 2857.52 | 2871.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 09:15:00 | 2917.00 | 2869.31 | 2873.44 | SL hit (close>static) qty=1.00 sl=2888.40 alert=retest2 |

### Cycle 88 — BUY (started 2025-06-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-26 10:15:00 | 2916.00 | 2878.65 | 2877.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 11:15:00 | 2951.90 | 2893.30 | 2884.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 12:15:00 | 3081.60 | 3095.66 | 3064.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-01 13:00:00 | 3081.60 | 3095.66 | 3064.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 3063.80 | 3085.87 | 3069.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 3063.80 | 3085.87 | 3069.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 3051.10 | 3078.92 | 3068.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:00:00 | 3051.10 | 3078.92 | 3068.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2025-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 12:15:00 | 3005.00 | 3056.87 | 3059.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 13:15:00 | 2986.00 | 3042.69 | 3052.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 13:15:00 | 2981.00 | 2964.49 | 2984.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-04 14:00:00 | 2981.00 | 2964.49 | 2984.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 2993.90 | 2970.37 | 2985.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 2993.90 | 2970.37 | 2985.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 2979.10 | 2972.12 | 2984.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 2999.00 | 2972.12 | 2984.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 2972.00 | 2972.10 | 2983.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:30:00 | 2994.80 | 2972.10 | 2983.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 2910.00 | 2947.12 | 2964.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:30:00 | 2904.40 | 2936.16 | 2958.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 12:45:00 | 2906.80 | 2922.04 | 2947.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 2904.60 | 2922.49 | 2941.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 10:45:00 | 2904.90 | 2917.75 | 2935.80 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 2915.60 | 2903.82 | 2918.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 09:45:00 | 2914.60 | 2903.82 | 2918.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 2911.20 | 2905.30 | 2918.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-10 11:15:00 | 2900.90 | 2905.30 | 2918.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 12:15:00 | 2920.10 | 2890.97 | 2887.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 2920.10 | 2890.97 | 2887.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 13:15:00 | 2931.80 | 2899.13 | 2891.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 14:15:00 | 2937.10 | 2937.64 | 2919.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 15:00:00 | 2937.10 | 2937.64 | 2919.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 15:15:00 | 2925.00 | 2939.85 | 2931.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 09:15:00 | 2935.90 | 2939.85 | 2931.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 2937.00 | 2939.28 | 2932.02 | EMA400 retest candle locked (from upside) |

### Cycle 91 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 2920.00 | 2928.67 | 2929.56 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 09:15:00 | 2952.00 | 2933.33 | 2931.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 09:15:00 | 2981.80 | 2946.33 | 2939.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 14:15:00 | 2946.90 | 2956.19 | 2948.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 14:15:00 | 2946.90 | 2956.19 | 2948.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 14:15:00 | 2946.90 | 2956.19 | 2948.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-22 15:00:00 | 2946.90 | 2956.19 | 2948.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 15:15:00 | 2952.00 | 2955.35 | 2948.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:15:00 | 2953.50 | 2955.35 | 2948.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 2950.40 | 2954.36 | 2948.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 10:30:00 | 2975.60 | 2956.47 | 2950.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 09:15:00 | 2919.90 | 2948.91 | 2949.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2025-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 09:15:00 | 2919.90 | 2948.91 | 2949.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 2901.80 | 2933.20 | 2940.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 2902.90 | 2902.26 | 2918.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 10:00:00 | 2902.90 | 2902.26 | 2918.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 12:15:00 | 2870.00 | 2843.34 | 2867.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:00:00 | 2870.00 | 2843.34 | 2867.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 2853.30 | 2845.33 | 2866.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 13:30:00 | 2864.20 | 2845.33 | 2866.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 2870.70 | 2850.41 | 2866.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:30:00 | 2873.00 | 2850.41 | 2866.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 2873.20 | 2854.97 | 2867.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 2884.00 | 2854.97 | 2867.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 2915.30 | 2878.40 | 2876.70 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2025-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 13:15:00 | 2841.00 | 2872.98 | 2875.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 2818.50 | 2857.34 | 2866.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 14:15:00 | 2843.80 | 2841.05 | 2853.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-31 15:00:00 | 2843.80 | 2841.05 | 2853.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 15:15:00 | 2840.00 | 2840.84 | 2852.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 2820.70 | 2840.84 | 2852.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 09:15:00 | 2913.20 | 2835.55 | 2839.93 | SL hit (close>static) qty=1.00 sl=2860.00 alert=retest2 |

### Cycle 96 — BUY (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 10:15:00 | 2904.00 | 2849.24 | 2845.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 13:15:00 | 2959.00 | 2916.99 | 2897.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 13:15:00 | 2938.00 | 2938.64 | 2921.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-07 13:30:00 | 2935.90 | 2938.64 | 2921.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 2912.00 | 2936.19 | 2926.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:00:00 | 2912.00 | 2936.19 | 2926.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 2914.70 | 2931.89 | 2925.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:30:00 | 2906.20 | 2931.89 | 2925.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2025-08-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 13:15:00 | 2906.80 | 2920.38 | 2920.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 2883.10 | 2912.92 | 2917.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 2919.00 | 2909.61 | 2914.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 2919.00 | 2909.61 | 2914.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 2919.00 | 2909.61 | 2914.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:30:00 | 2911.40 | 2909.61 | 2914.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 2926.30 | 2912.95 | 2915.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:00:00 | 2926.30 | 2912.95 | 2915.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 2925.00 | 2915.36 | 2916.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:15:00 | 2930.20 | 2915.36 | 2916.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — BUY (started 2025-08-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 12:15:00 | 2936.90 | 2919.67 | 2918.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 2981.00 | 2931.93 | 2924.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 11:15:00 | 3055.00 | 3063.85 | 3047.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 12:00:00 | 3055.00 | 3063.85 | 3047.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 14:15:00 | 3049.20 | 3065.86 | 3052.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 15:00:00 | 3049.20 | 3065.86 | 3052.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 3051.10 | 3062.91 | 3052.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 09:15:00 | 3092.70 | 3062.91 | 3052.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 15:15:00 | 3099.50 | 3126.04 | 3129.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 3099.50 | 3126.04 | 3129.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 12:15:00 | 3089.90 | 3118.68 | 3125.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 11:15:00 | 3093.10 | 3090.48 | 3105.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 12:00:00 | 3093.10 | 3090.48 | 3105.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 3100.00 | 3092.39 | 3104.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 12:45:00 | 3097.40 | 3092.39 | 3104.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 3006.40 | 2974.15 | 2992.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 3009.60 | 2974.15 | 2992.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 3036.80 | 2986.68 | 2996.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 10:45:00 | 3038.00 | 2986.68 | 2996.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 3072.10 | 3003.76 | 3003.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 12:15:00 | 3113.00 | 3025.61 | 3013.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 11:15:00 | 3052.50 | 3087.29 | 3057.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 11:15:00 | 3052.50 | 3087.29 | 3057.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 11:15:00 | 3052.50 | 3087.29 | 3057.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 12:00:00 | 3052.50 | 3087.29 | 3057.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 3095.10 | 3088.85 | 3060.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 12:30:00 | 3068.60 | 3088.85 | 3060.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 3065.00 | 3084.08 | 3061.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:30:00 | 3059.20 | 3084.08 | 3061.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 3048.80 | 3077.02 | 3060.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:00:00 | 3048.80 | 3077.02 | 3060.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 3050.00 | 3071.62 | 3059.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:15:00 | 3064.00 | 3071.62 | 3059.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 10:15:00 | 3022.80 | 3058.40 | 3055.20 | SL hit (close<static) qty=1.00 sl=3043.30 alert=retest2 |

### Cycle 101 — SELL (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 11:15:00 | 3028.80 | 3052.48 | 3052.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 12:15:00 | 3006.30 | 3043.24 | 3048.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 14:15:00 | 3044.80 | 3037.64 | 3044.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 14:15:00 | 3044.80 | 3037.64 | 3044.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 3044.80 | 3037.64 | 3044.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 15:00:00 | 3044.80 | 3037.64 | 3044.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 3050.00 | 3040.11 | 3045.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 3053.10 | 3040.11 | 3045.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 3062.70 | 3044.63 | 3046.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:00:00 | 3062.70 | 3044.63 | 3046.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — BUY (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 11:15:00 | 3050.70 | 3048.30 | 3048.25 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 12:15:00 | 3025.90 | 3043.82 | 3046.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 3003.40 | 3035.74 | 3042.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 14:15:00 | 2991.80 | 2991.36 | 3010.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-05 14:30:00 | 2990.40 | 2991.36 | 3010.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 2980.00 | 2988.50 | 3006.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 11:15:00 | 2968.50 | 2985.64 | 3003.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 11:15:00 | 3017.90 | 2992.09 | 3004.50 | SL hit (close>static) qty=1.00 sl=3006.90 alert=retest2 |

### Cycle 104 — BUY (started 2025-09-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 14:15:00 | 3070.80 | 3013.69 | 3011.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 15:15:00 | 3086.00 | 3028.15 | 3018.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 15:15:00 | 3207.00 | 3209.21 | 3178.53 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 09:15:00 | 3221.00 | 3209.21 | 3178.53 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 10:15:00 | 3218.00 | 3209.51 | 3181.46 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-12 15:00:00 | 3224.80 | 3212.29 | 3193.41 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 09:15:00 | 3382.05 | 3336.19 | 3282.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 09:15:00 | 3378.90 | 3336.19 | 3282.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-16 09:15:00 | 3386.04 | 3336.19 | 3282.47 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-09-16 15:15:00 | 3376.50 | 3378.99 | 3331.55 | SL hit (close<ema200) qty=0.50 sl=3378.99 alert=retest1 |

### Cycle 105 — SELL (started 2025-09-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 13:15:00 | 3328.90 | 3355.50 | 3358.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 14:15:00 | 3323.20 | 3349.04 | 3355.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 11:15:00 | 3377.10 | 3353.34 | 3355.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 11:15:00 | 3377.10 | 3353.34 | 3355.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 3377.10 | 3353.34 | 3355.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:00:00 | 3377.10 | 3353.34 | 3355.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 3354.60 | 3353.59 | 3355.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 13:15:00 | 3345.50 | 3353.59 | 3355.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-22 15:15:00 | 3340.00 | 3353.10 | 3354.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 09:45:00 | 3350.50 | 3352.12 | 3353.91 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 10:45:00 | 3347.30 | 3349.22 | 3352.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 3351.20 | 3349.61 | 3352.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:00:00 | 3351.20 | 3349.61 | 3352.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 3351.80 | 3350.05 | 3352.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:30:00 | 3363.00 | 3350.05 | 3352.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 3321.40 | 3344.32 | 3349.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 14:15:00 | 3316.90 | 3344.32 | 3349.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 11:45:00 | 3310.80 | 3315.01 | 3330.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 13:00:00 | 3317.30 | 3315.47 | 3329.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 3178.22 | 3221.58 | 3261.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 3173.00 | 3221.58 | 3261.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 3182.97 | 3221.58 | 3261.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 09:15:00 | 3179.93 | 3221.58 | 3261.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 10:15:00 | 3151.05 | 3206.86 | 3250.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 10:15:00 | 3151.43 | 3206.86 | 3250.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-26 11:15:00 | 3145.26 | 3191.99 | 3240.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-30 15:15:00 | 3110.80 | 3103.26 | 3130.74 | SL hit (close>ema200) qty=0.50 sl=3103.26 alert=retest2 |

### Cycle 106 — BUY (started 2025-10-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 15:15:00 | 3130.00 | 3111.58 | 3110.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 10:15:00 | 3166.00 | 3125.81 | 3117.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 09:15:00 | 3182.70 | 3196.21 | 3172.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 10:00:00 | 3182.70 | 3196.21 | 3172.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 3183.20 | 3193.60 | 3173.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:15:00 | 3173.80 | 3193.60 | 3173.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 3164.00 | 3187.68 | 3172.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:00:00 | 3164.00 | 3187.68 | 3172.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 3165.60 | 3183.27 | 3171.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 12:30:00 | 3157.20 | 3183.27 | 3171.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 15:15:00 | 3136.90 | 3163.52 | 3164.81 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2025-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 11:15:00 | 3172.20 | 3166.53 | 3165.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 12:15:00 | 3183.30 | 3169.89 | 3167.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 15:15:00 | 3176.10 | 3176.67 | 3171.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-10 09:15:00 | 3171.00 | 3176.67 | 3171.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 3169.40 | 3175.22 | 3171.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:00:00 | 3169.40 | 3175.22 | 3171.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 3161.00 | 3172.37 | 3170.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:45:00 | 3146.60 | 3172.37 | 3170.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 11:15:00 | 3156.20 | 3169.14 | 3169.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 09:15:00 | 3124.00 | 3155.18 | 3162.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 3119.00 | 3106.11 | 3119.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 3119.00 | 3106.11 | 3119.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 3119.00 | 3106.11 | 3119.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:00:00 | 3119.00 | 3106.11 | 3119.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 3157.90 | 3116.47 | 3122.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 3157.90 | 3116.47 | 3122.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 3155.40 | 3124.25 | 3125.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 3152.20 | 3124.25 | 3125.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 110 — BUY (started 2025-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 12:15:00 | 3142.70 | 3127.94 | 3127.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 15:15:00 | 3161.10 | 3140.07 | 3133.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 10:15:00 | 3168.30 | 3174.20 | 3161.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 10:45:00 | 3166.90 | 3174.20 | 3161.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 3152.00 | 3169.76 | 3160.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:00:00 | 3152.00 | 3169.76 | 3160.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 3122.80 | 3160.37 | 3157.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 3122.80 | 3160.37 | 3157.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 3130.00 | 3154.29 | 3154.69 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 09:15:00 | 3192.90 | 3156.44 | 3155.11 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 10:15:00 | 3139.60 | 3153.07 | 3153.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 11:15:00 | 3122.90 | 3147.04 | 3150.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 3136.80 | 3134.79 | 3142.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 3136.80 | 3134.79 | 3142.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 3136.80 | 3134.79 | 3142.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 3141.00 | 3134.79 | 3142.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 3184.80 | 3144.10 | 3145.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 3184.80 | 3144.10 | 3145.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 3184.30 | 3152.14 | 3148.83 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 09:15:00 | 3152.10 | 3165.39 | 3166.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 13:15:00 | 3123.00 | 3148.79 | 3157.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 3159.30 | 3146.92 | 3154.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 3159.30 | 3146.92 | 3154.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 3159.30 | 3146.92 | 3154.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:45:00 | 3161.10 | 3146.92 | 3154.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 3150.00 | 3147.54 | 3153.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 11:15:00 | 3141.80 | 3147.54 | 3153.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 13:00:00 | 3145.10 | 3147.16 | 3152.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:15:00 | 3140.90 | 3135.03 | 3139.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 14:15:00 | 2984.71 | 3021.72 | 3041.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 14:15:00 | 2987.84 | 3021.72 | 3041.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-04 15:15:00 | 2983.86 | 3014.38 | 3036.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-06 09:15:00 | 3041.30 | 3019.76 | 3036.58 | SL hit (close>ema200) qty=0.50 sl=3019.76 alert=retest2 |

### Cycle 116 — BUY (started 2025-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 15:15:00 | 2995.00 | 2982.79 | 2982.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 09:15:00 | 3027.90 | 2991.81 | 2986.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 3032.20 | 3042.85 | 3026.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 3032.20 | 3042.85 | 3026.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 3049.00 | 3044.08 | 3028.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 3067.40 | 3044.08 | 3028.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 11:00:00 | 3052.80 | 3045.94 | 3031.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:45:00 | 3052.30 | 3047.98 | 3035.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 13:15:00 | 3057.00 | 3047.98 | 3035.44 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 3051.70 | 3048.72 | 3036.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 14:00:00 | 3051.70 | 3048.72 | 3036.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 3030.50 | 3075.54 | 3064.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 3030.50 | 3075.54 | 3064.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 3035.50 | 3067.53 | 3061.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:30:00 | 3032.20 | 3067.53 | 3061.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 3043.20 | 3059.41 | 3058.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:45:00 | 3044.20 | 3059.41 | 3058.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-18 13:15:00 | 3047.10 | 3056.95 | 3057.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 13:15:00 | 3047.10 | 3056.95 | 3057.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 3024.10 | 3044.39 | 3051.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 14:15:00 | 3036.90 | 3036.02 | 3043.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 14:30:00 | 3038.90 | 3036.02 | 3043.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 3039.50 | 3034.96 | 3041.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-20 11:30:00 | 3031.80 | 3032.15 | 3039.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-21 15:15:00 | 2880.21 | 2927.53 | 2971.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-24 14:15:00 | 2913.10 | 2898.32 | 2933.79 | SL hit (close>ema200) qty=0.50 sl=2898.32 alert=retest2 |

### Cycle 118 — BUY (started 2025-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 10:15:00 | 2671.00 | 2638.83 | 2637.93 | EMA200 above EMA400 |

### Cycle 119 — SELL (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 09:15:00 | 2613.30 | 2637.25 | 2640.02 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 15:15:00 | 2646.20 | 2639.46 | 2639.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 2654.60 | 2642.49 | 2640.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 11:15:00 | 2639.30 | 2643.40 | 2641.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 11:15:00 | 2639.30 | 2643.40 | 2641.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 2639.30 | 2643.40 | 2641.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:00:00 | 2639.30 | 2643.40 | 2641.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 12:15:00 | 2645.20 | 2643.76 | 2641.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 12:30:00 | 2633.50 | 2643.76 | 2641.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 13:15:00 | 2651.90 | 2645.39 | 2642.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 13:30:00 | 2644.30 | 2645.39 | 2642.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 2660.10 | 2648.33 | 2644.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:30:00 | 2643.80 | 2648.33 | 2644.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 2627.90 | 2643.71 | 2642.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:30:00 | 2620.60 | 2643.71 | 2642.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — SELL (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 10:15:00 | 2635.10 | 2641.99 | 2642.10 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 15:15:00 | 2654.50 | 2642.26 | 2641.59 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 2631.00 | 2640.01 | 2640.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 11:15:00 | 2616.70 | 2633.27 | 2637.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 2628.60 | 2626.85 | 2632.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 2628.60 | 2626.85 | 2632.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 2628.60 | 2626.85 | 2632.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:30:00 | 2651.50 | 2626.85 | 2632.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 09:15:00 | 2589.80 | 2611.32 | 2621.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 13:30:00 | 2582.00 | 2598.29 | 2611.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:00:00 | 2583.20 | 2589.59 | 2603.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 10:45:00 | 2578.30 | 2586.25 | 2600.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 14:15:00 | 2630.30 | 2600.76 | 2603.44 | SL hit (close>static) qty=1.00 sl=2629.10 alert=retest2 |

### Cycle 124 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 2631.50 | 2606.91 | 2605.99 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-12-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-22 15:15:00 | 2593.60 | 2607.85 | 2608.55 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 09:15:00 | 2617.50 | 2603.11 | 2603.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 11:15:00 | 2621.00 | 2608.95 | 2605.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 14:15:00 | 2602.10 | 2611.52 | 2608.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 14:15:00 | 2602.10 | 2611.52 | 2608.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 2602.10 | 2611.52 | 2608.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:45:00 | 2613.30 | 2611.52 | 2608.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 2597.00 | 2608.62 | 2607.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 2597.30 | 2608.62 | 2607.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 2595.20 | 2604.80 | 2605.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-29 11:15:00 | 2583.00 | 2600.44 | 2603.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 13:15:00 | 2609.90 | 2598.63 | 2601.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 13:15:00 | 2609.90 | 2598.63 | 2601.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 2609.90 | 2598.63 | 2601.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 2609.90 | 2598.63 | 2601.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 2605.00 | 2599.90 | 2602.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 2605.00 | 2599.90 | 2602.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 2600.30 | 2599.98 | 2602.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 2578.00 | 2599.98 | 2602.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 2570.10 | 2594.01 | 2599.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 12:15:00 | 2563.80 | 2585.36 | 2594.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 13:45:00 | 2564.00 | 2578.30 | 2589.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 13:15:00 | 2609.30 | 2590.73 | 2589.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 2609.30 | 2590.73 | 2589.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 2616.50 | 2595.89 | 2592.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 13:15:00 | 2609.80 | 2613.26 | 2604.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 13:30:00 | 2609.50 | 2613.26 | 2604.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 2608.00 | 2615.09 | 2607.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 09:45:00 | 2607.70 | 2615.09 | 2607.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 2593.40 | 2610.76 | 2606.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:45:00 | 2592.10 | 2610.76 | 2606.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 2591.30 | 2606.86 | 2604.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:00:00 | 2591.30 | 2606.86 | 2604.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 13:15:00 | 2590.00 | 2601.10 | 2602.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-05 09:15:00 | 2553.00 | 2592.10 | 2598.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 13:15:00 | 2553.30 | 2542.93 | 2558.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-06 14:00:00 | 2553.30 | 2542.93 | 2558.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 2545.40 | 2543.42 | 2557.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:45:00 | 2559.90 | 2543.42 | 2557.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 2558.20 | 2547.43 | 2556.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 2558.20 | 2547.43 | 2556.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 2553.20 | 2548.58 | 2556.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-07 11:15:00 | 2540.40 | 2548.58 | 2556.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 09:15:00 | 2413.38 | 2453.53 | 2487.42 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 2374.40 | 2362.87 | 2384.21 | SL hit (close>ema200) qty=0.50 sl=2362.87 alert=retest2 |

### Cycle 130 — BUY (started 2026-01-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 15:15:00 | 2380.00 | 2378.08 | 2378.03 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2026-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 09:15:00 | 2361.80 | 2374.82 | 2376.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 12:15:00 | 2342.40 | 2362.39 | 2369.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 2299.50 | 2291.64 | 2317.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:45:00 | 2301.70 | 2291.64 | 2317.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 14:15:00 | 2303.10 | 2284.58 | 2303.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 15:00:00 | 2303.10 | 2284.58 | 2303.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 2305.00 | 2288.66 | 2303.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:15:00 | 2275.90 | 2288.66 | 2303.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 15:15:00 | 2298.00 | 2243.87 | 2237.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 2298.00 | 2243.87 | 2237.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 13:15:00 | 2304.50 | 2277.42 | 2258.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 12:15:00 | 2311.90 | 2314.79 | 2288.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 12:30:00 | 2315.80 | 2314.79 | 2288.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 2336.40 | 2350.38 | 2329.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 2313.60 | 2350.38 | 2329.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 2306.80 | 2341.67 | 2327.44 | EMA400 retest candle locked (from upside) |

### Cycle 133 — SELL (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 12:15:00 | 2286.50 | 2313.81 | 2316.69 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 2471.60 | 2343.87 | 2328.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 10:15:00 | 2515.20 | 2378.13 | 2345.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 2438.70 | 2530.31 | 2479.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 2438.70 | 2530.31 | 2479.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 2438.70 | 2530.31 | 2479.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 2438.70 | 2530.31 | 2479.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 2426.10 | 2509.47 | 2474.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:45:00 | 2434.10 | 2509.47 | 2474.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — SELL (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 13:15:00 | 2375.10 | 2449.40 | 2452.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 2292.60 | 2397.57 | 2426.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 09:15:00 | 2372.00 | 2337.08 | 2372.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 09:15:00 | 2372.00 | 2337.08 | 2372.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 2372.00 | 2337.08 | 2372.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:00:00 | 2372.00 | 2337.08 | 2372.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 2359.80 | 2341.63 | 2371.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 11:15:00 | 2348.10 | 2341.63 | 2371.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 09:15:00 | 2382.00 | 2362.16 | 2370.04 | SL hit (close>static) qty=1.00 sl=2378.40 alert=retest2 |

### Cycle 136 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 10:15:00 | 2438.50 | 2377.43 | 2376.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 10:15:00 | 2469.50 | 2430.11 | 2407.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 12:15:00 | 2431.40 | 2433.15 | 2413.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 13:00:00 | 2431.40 | 2433.15 | 2413.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 2440.80 | 2439.52 | 2423.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 09:30:00 | 2422.10 | 2439.52 | 2423.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 2475.00 | 2478.34 | 2457.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:45:00 | 2491.10 | 2485.53 | 2462.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 09:15:00 | 2515.10 | 2490.12 | 2472.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 13:15:00 | 2458.40 | 2480.42 | 2482.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 13:15:00 | 2458.40 | 2480.42 | 2482.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-18 09:15:00 | 2439.30 | 2468.01 | 2475.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 13:15:00 | 2462.70 | 2455.07 | 2465.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 14:00:00 | 2462.70 | 2455.07 | 2465.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 2473.20 | 2458.70 | 2466.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 2473.20 | 2458.70 | 2466.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 2469.90 | 2460.94 | 2466.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 2500.70 | 2460.94 | 2466.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 2502.00 | 2469.15 | 2469.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:45:00 | 2502.10 | 2469.15 | 2469.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-19 10:15:00 | 2514.70 | 2478.26 | 2473.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 09:15:00 | 2517.10 | 2489.75 | 2482.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 11:15:00 | 2543.80 | 2551.81 | 2527.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-23 11:45:00 | 2532.90 | 2551.81 | 2527.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 2533.80 | 2548.21 | 2528.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 13:00:00 | 2533.80 | 2548.21 | 2528.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 2541.20 | 2544.56 | 2530.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 2556.90 | 2543.41 | 2530.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-24 14:30:00 | 2552.30 | 2546.69 | 2538.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-03-02 14:15:00 | 2812.59 | 2772.87 | 2728.15 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 13:15:00 | 2734.00 | 2755.07 | 2755.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 2627.10 | 2720.58 | 2738.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 10:15:00 | 2630.90 | 2626.61 | 2667.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-10 10:45:00 | 2623.50 | 2626.61 | 2667.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 15:15:00 | 2652.50 | 2630.88 | 2653.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:15:00 | 2666.20 | 2630.88 | 2653.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 2656.60 | 2636.03 | 2653.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:15:00 | 2640.70 | 2636.03 | 2653.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-12 09:15:00 | 2508.66 | 2574.26 | 2611.34 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-12 13:15:00 | 2547.40 | 2547.04 | 2584.56 | SL hit (close>ema200) qty=0.50 sl=2547.04 alert=retest2 |

### Cycle 140 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 2525.10 | 2463.15 | 2462.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 11:15:00 | 2533.60 | 2477.24 | 2469.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 2505.50 | 2523.19 | 2499.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 2505.50 | 2523.19 | 2499.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 2505.50 | 2523.19 | 2499.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:30:00 | 2522.90 | 2523.89 | 2502.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 2449.20 | 2524.72 | 2523.23 | SL hit (close<static) qty=1.00 sl=2476.40 alert=retest2 |

### Cycle 141 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 2439.30 | 2507.63 | 2515.60 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2026-03-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 10:15:00 | 2567.00 | 2516.94 | 2512.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 11:15:00 | 2587.50 | 2531.05 | 2519.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 2613.70 | 2642.94 | 2604.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:00:00 | 2613.70 | 2642.94 | 2604.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 2583.50 | 2631.05 | 2602.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 2583.50 | 2631.05 | 2602.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 2587.50 | 2622.34 | 2601.51 | EMA400 retest candle locked (from upside) |

### Cycle 143 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 2533.90 | 2582.88 | 2588.51 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 2611.60 | 2574.81 | 2573.57 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 2490.90 | 2560.86 | 2568.83 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 2573.10 | 2561.25 | 2561.24 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2026-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 10:15:00 | 2541.20 | 2560.45 | 2562.10 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2026-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 12:15:00 | 2577.80 | 2565.30 | 2564.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 14:15:00 | 2588.20 | 2571.72 | 2567.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 13:15:00 | 2740.10 | 2742.04 | 2713.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-10 14:00:00 | 2740.10 | 2742.04 | 2713.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 2716.50 | 2736.17 | 2717.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 2726.80 | 2732.39 | 2717.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 12:30:00 | 2727.90 | 2729.58 | 2718.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:00:00 | 2726.30 | 2729.58 | 2718.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 2751.00 | 2723.75 | 2718.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 12:15:00 | 2724.90 | 2728.25 | 2723.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 13:15:00 | 2725.00 | 2728.25 | 2723.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 13:15:00 | 2718.50 | 2726.30 | 2722.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:00:00 | 2718.50 | 2726.30 | 2722.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 2760.00 | 2733.04 | 2726.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-15 14:30:00 | 2723.50 | 2733.04 | 2726.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 2733.50 | 2751.82 | 2740.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 13:00:00 | 2733.50 | 2751.82 | 2740.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 2754.20 | 2752.29 | 2741.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 13:45:00 | 2718.80 | 2752.29 | 2741.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 2754.20 | 2752.68 | 2742.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 2779.00 | 2753.90 | 2743.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 10:00:00 | 2770.00 | 2757.12 | 2746.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 09:45:00 | 2769.70 | 2782.67 | 2767.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 12:15:00 | 2999.48 | 2943.96 | 2892.88 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 149 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 2905.40 | 2979.25 | 2981.78 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 2993.00 | 2981.97 | 2981.55 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 09:15:00 | 2961.80 | 2982.32 | 2982.37 | EMA200 below EMA400 |

### Cycle 152 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 3019.00 | 2989.66 | 2985.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 3045.80 | 3010.37 | 2998.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 11:15:00 | 3015.00 | 3016.11 | 3003.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 12:00:00 | 3015.00 | 3016.11 | 3003.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 2997.70 | 3012.43 | 3002.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 13:00:00 | 2997.70 | 3012.43 | 3002.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 2969.80 | 3003.91 | 2999.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:00:00 | 2969.80 | 3003.91 | 2999.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 2969.70 | 2997.06 | 2997.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:45:00 | 2965.60 | 2997.06 | 2997.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 153 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 2974.80 | 2992.61 | 2995.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 2913.10 | 2976.71 | 2987.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 15:15:00 | 2963.40 | 2946.95 | 2963.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 09:15:00 | 2963.30 | 2946.95 | 2963.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 2930.30 | 2943.62 | 2960.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:15:00 | 2913.60 | 2940.81 | 2956.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 09:15:00 | 2914.50 | 2923.35 | 2941.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 12:45:00 | 2919.40 | 2920.58 | 2933.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 2914.70 | 2927.79 | 2934.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 2905.10 | 2923.25 | 2931.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 11:45:00 | 2882.70 | 2914.17 | 2925.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 12:15:00 | 2879.70 | 2914.17 | 2925.74 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-06 13:00:00 | 2886.30 | 2908.59 | 2922.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-07 09:15:00 | 2980.30 | 2924.85 | 2925.21 | SL hit (close>static) qty=1.00 sl=2978.60 alert=retest2 |

### Cycle 154 — BUY (started 2026-05-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 10:15:00 | 2973.30 | 2934.54 | 2929.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 13:15:00 | 3019.10 | 2966.58 | 2946.71 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-16 11:00:00 | 3745.00 | 2024-05-23 11:15:00 | 3787.95 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-05-16 12:30:00 | 3745.05 | 2024-05-23 11:15:00 | 3787.95 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-05-16 15:15:00 | 3751.00 | 2024-05-23 11:15:00 | 3787.95 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-05-21 09:15:00 | 3734.00 | 2024-05-23 11:15:00 | 3787.95 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-05-28 09:15:00 | 3831.40 | 2024-05-28 11:15:00 | 3779.25 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-05-29 09:30:00 | 3772.55 | 2024-05-31 09:15:00 | 3583.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-29 09:30:00 | 3772.55 | 2024-05-31 09:15:00 | 3710.50 | STOP_HIT | 0.50 | 1.64% |
| BUY | retest1 | 2024-06-11 09:15:00 | 3989.90 | 2024-06-12 09:15:00 | 4189.40 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-06-11 10:30:00 | 3999.50 | 2024-06-12 09:15:00 | 4199.48 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2024-06-11 09:15:00 | 3989.90 | 2024-06-12 13:15:00 | 4099.15 | STOP_HIT | 0.50 | 2.74% |
| BUY | retest1 | 2024-06-11 10:30:00 | 3999.50 | 2024-06-12 13:15:00 | 4099.15 | STOP_HIT | 0.50 | 2.49% |
| BUY | retest2 | 2024-07-01 12:15:00 | 4256.60 | 2024-07-02 10:15:00 | 4203.00 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-07-04 09:15:00 | 4184.35 | 2024-07-04 14:15:00 | 4338.55 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2024-07-18 10:15:00 | 4064.40 | 2024-07-22 14:15:00 | 4119.85 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2024-07-18 12:45:00 | 4060.90 | 2024-07-22 14:15:00 | 4119.85 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2024-07-18 14:45:00 | 4053.30 | 2024-07-22 14:15:00 | 4119.85 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-07-19 09:15:00 | 4061.45 | 2024-07-22 14:15:00 | 4119.85 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-07-19 10:45:00 | 3994.90 | 2024-07-23 09:15:00 | 4093.45 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2024-07-19 11:30:00 | 4004.95 | 2024-07-23 09:15:00 | 4093.45 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2024-07-19 12:45:00 | 3996.15 | 2024-07-23 09:15:00 | 4093.45 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2024-07-19 14:15:00 | 4001.35 | 2024-07-23 09:15:00 | 4093.45 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2024-07-30 09:15:00 | 4224.00 | 2024-08-01 10:15:00 | 4124.00 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2024-07-30 14:15:00 | 4161.30 | 2024-08-01 10:15:00 | 4124.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-08-01 09:15:00 | 4156.95 | 2024-08-01 10:15:00 | 4124.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-08-06 14:00:00 | 3979.00 | 2024-08-07 13:15:00 | 4062.25 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2024-08-22 09:15:00 | 4090.50 | 2024-08-29 10:15:00 | 4055.70 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-08-26 09:30:00 | 4076.95 | 2024-08-29 10:15:00 | 4055.70 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2024-09-04 12:30:00 | 4108.45 | 2024-09-05 15:15:00 | 4065.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-09-04 15:00:00 | 4124.20 | 2024-09-05 15:15:00 | 4065.00 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2024-09-05 14:30:00 | 4113.90 | 2024-09-05 15:15:00 | 4065.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-09-10 15:00:00 | 3921.30 | 2024-09-13 09:15:00 | 4025.10 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2024-09-11 13:30:00 | 3928.00 | 2024-09-13 09:15:00 | 4025.10 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2024-09-12 10:15:00 | 3931.55 | 2024-09-13 09:15:00 | 4025.10 | STOP_HIT | 1.00 | -2.38% |
| BUY | retest2 | 2024-09-17 09:15:00 | 4112.00 | 2024-09-18 13:15:00 | 4022.50 | STOP_HIT | 1.00 | -2.18% |
| BUY | retest2 | 2024-09-18 10:30:00 | 4068.95 | 2024-09-18 13:15:00 | 4022.50 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-10-23 13:15:00 | 4473.25 | 2024-10-29 10:15:00 | 4510.85 | STOP_HIT | 1.00 | 0.84% |
| SELL | retest2 | 2024-11-04 10:15:00 | 4409.85 | 2024-11-05 09:15:00 | 4189.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-04 14:15:00 | 4399.95 | 2024-11-05 09:15:00 | 4179.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-04 14:45:00 | 4395.95 | 2024-11-05 09:15:00 | 4176.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-04 10:15:00 | 4409.85 | 2024-11-08 09:15:00 | 3968.87 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-04 14:15:00 | 4399.95 | 2024-11-08 09:15:00 | 3959.95 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-04 14:45:00 | 4395.95 | 2024-11-08 09:15:00 | 3956.36 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-11-27 09:15:00 | 3376.55 | 2024-11-28 09:15:00 | 3550.30 | STOP_HIT | 1.00 | -5.15% |
| BUY | retest1 | 2024-12-04 09:45:00 | 3631.90 | 2024-12-10 12:15:00 | 3685.50 | STOP_HIT | 1.00 | 1.48% |
| BUY | retest1 | 2024-12-04 10:30:00 | 3631.70 | 2024-12-10 12:15:00 | 3685.50 | STOP_HIT | 1.00 | 1.48% |
| BUY | retest1 | 2024-12-04 11:00:00 | 3632.90 | 2024-12-10 12:15:00 | 3685.50 | STOP_HIT | 1.00 | 1.45% |
| BUY | retest1 | 2024-12-04 12:30:00 | 3636.45 | 2024-12-10 12:15:00 | 3685.50 | STOP_HIT | 1.00 | 1.35% |
| BUY | retest2 | 2024-12-09 09:30:00 | 3712.00 | 2024-12-11 12:15:00 | 3689.40 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2024-12-09 12:30:00 | 3723.40 | 2024-12-11 12:15:00 | 3689.40 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-12-10 15:00:00 | 3703.85 | 2024-12-11 12:15:00 | 3689.40 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2024-12-11 09:15:00 | 3723.00 | 2024-12-11 12:15:00 | 3689.40 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-12-19 10:30:00 | 3726.55 | 2024-12-20 12:15:00 | 3627.45 | STOP_HIT | 1.00 | -2.66% |
| BUY | retest2 | 2024-12-19 11:30:00 | 3732.65 | 2024-12-20 12:15:00 | 3627.45 | STOP_HIT | 1.00 | -2.82% |
| SELL | retest2 | 2024-12-27 11:30:00 | 3597.05 | 2025-01-02 09:15:00 | 3622.35 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-12-27 14:45:00 | 3594.90 | 2025-01-02 09:15:00 | 3622.35 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-12-30 09:30:00 | 3570.10 | 2025-01-02 09:15:00 | 3622.35 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-12-31 14:15:00 | 3590.50 | 2025-01-02 09:15:00 | 3622.35 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2024-12-31 15:15:00 | 3553.00 | 2025-01-02 09:15:00 | 3622.35 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-01-01 15:15:00 | 3561.00 | 2025-01-02 09:15:00 | 3622.35 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-01-15 09:15:00 | 3302.15 | 2025-01-16 09:15:00 | 3380.55 | STOP_HIT | 1.00 | -2.37% |
| SELL | retest2 | 2025-01-15 12:15:00 | 3303.75 | 2025-01-16 09:15:00 | 3380.55 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-01-15 12:45:00 | 3303.80 | 2025-01-16 09:15:00 | 3380.55 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-02-04 11:45:00 | 3073.80 | 2025-02-07 09:15:00 | 2920.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-05 09:15:00 | 3058.85 | 2025-02-07 13:15:00 | 2905.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-04 11:45:00 | 3073.80 | 2025-02-11 12:15:00 | 2766.42 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-05 09:15:00 | 3058.85 | 2025-02-11 12:15:00 | 2752.97 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-21 12:45:00 | 2709.50 | 2025-02-21 15:15:00 | 2681.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-03-06 14:00:00 | 2655.15 | 2025-03-11 10:15:00 | 2920.67 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-06 14:30:00 | 2657.90 | 2025-03-11 10:15:00 | 2915.61 | TARGET_HIT | 1.00 | 9.70% |
| BUY | retest2 | 2025-03-06 15:00:00 | 2650.55 | 2025-03-11 11:15:00 | 2923.69 | TARGET_HIT | 1.00 | 10.31% |
| BUY | retest2 | 2025-03-07 09:15:00 | 2657.30 | 2025-03-11 11:15:00 | 2923.03 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-11 10:30:00 | 2868.35 | 2025-03-17 13:15:00 | 2839.00 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2025-03-11 13:15:00 | 2878.00 | 2025-03-17 13:15:00 | 2839.00 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-03-12 10:30:00 | 2865.40 | 2025-03-17 13:15:00 | 2839.00 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-03-12 12:15:00 | 2865.10 | 2025-03-17 13:15:00 | 2839.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-03-24 09:15:00 | 2892.65 | 2025-03-24 13:15:00 | 2877.15 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-03-24 13:15:00 | 2876.00 | 2025-03-24 13:15:00 | 2877.15 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-04-02 12:15:00 | 2700.20 | 2025-04-07 09:15:00 | 2565.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 10:00:00 | 2697.00 | 2025-04-07 09:15:00 | 2562.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 12:00:00 | 2692.50 | 2025-04-07 09:15:00 | 2557.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-02 12:15:00 | 2700.20 | 2025-04-08 10:15:00 | 2589.80 | STOP_HIT | 0.50 | 4.09% |
| SELL | retest2 | 2025-04-03 10:00:00 | 2697.00 | 2025-04-08 10:15:00 | 2589.80 | STOP_HIT | 0.50 | 3.97% |
| SELL | retest2 | 2025-04-03 12:00:00 | 2692.50 | 2025-04-08 10:15:00 | 2589.80 | STOP_HIT | 0.50 | 3.81% |
| BUY | retest2 | 2025-05-07 11:30:00 | 2967.80 | 2025-05-08 13:15:00 | 2907.90 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-05-07 12:00:00 | 2970.00 | 2025-05-08 13:15:00 | 2907.90 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-05-07 14:00:00 | 2966.20 | 2025-05-08 14:15:00 | 2889.00 | STOP_HIT | 1.00 | -2.60% |
| BUY | retest2 | 2025-05-07 14:30:00 | 2964.70 | 2025-05-08 14:15:00 | 2889.00 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-05-08 10:45:00 | 2995.00 | 2025-05-08 14:15:00 | 2889.00 | STOP_HIT | 1.00 | -3.54% |
| BUY | retest2 | 2025-05-08 11:45:00 | 2963.80 | 2025-05-08 14:15:00 | 2889.00 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-05-22 11:15:00 | 2990.00 | 2025-05-23 13:15:00 | 3046.00 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-05-22 12:45:00 | 2990.00 | 2025-05-23 13:15:00 | 3046.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-06-03 09:15:00 | 3077.40 | 2025-06-03 11:15:00 | 3040.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2025-06-17 10:30:00 | 2944.90 | 2025-06-20 14:15:00 | 2922.50 | STOP_HIT | 1.00 | 0.76% |
| SELL | retest2 | 2025-06-25 12:15:00 | 2836.30 | 2025-06-26 09:15:00 | 2917.00 | STOP_HIT | 1.00 | -2.85% |
| SELL | retest2 | 2025-06-25 13:45:00 | 2835.50 | 2025-06-26 09:15:00 | 2917.00 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2025-07-08 10:30:00 | 2904.40 | 2025-07-15 12:15:00 | 2920.10 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-07-08 12:45:00 | 2906.80 | 2025-07-15 12:15:00 | 2920.10 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2025-07-09 09:15:00 | 2904.60 | 2025-07-15 12:15:00 | 2920.10 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-07-09 10:45:00 | 2904.90 | 2025-07-15 12:15:00 | 2920.10 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-07-10 11:15:00 | 2900.90 | 2025-07-15 12:15:00 | 2920.10 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-07-23 10:30:00 | 2975.60 | 2025-07-24 09:15:00 | 2919.90 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-08-01 09:15:00 | 2820.70 | 2025-08-04 09:15:00 | 2913.20 | STOP_HIT | 1.00 | -3.28% |
| BUY | retest2 | 2025-08-19 09:15:00 | 3092.70 | 2025-08-21 15:15:00 | 3099.50 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2025-09-03 09:15:00 | 3064.00 | 2025-09-03 10:15:00 | 3022.80 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-09-08 11:15:00 | 2968.50 | 2025-09-08 11:15:00 | 3017.90 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest1 | 2025-09-12 09:15:00 | 3221.00 | 2025-09-16 09:15:00 | 3382.05 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-09-12 10:15:00 | 3218.00 | 2025-09-16 09:15:00 | 3378.90 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-09-12 15:00:00 | 3224.80 | 2025-09-16 09:15:00 | 3386.04 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-09-12 09:15:00 | 3221.00 | 2025-09-16 15:15:00 | 3376.50 | STOP_HIT | 0.50 | 4.83% |
| BUY | retest1 | 2025-09-12 10:15:00 | 3218.00 | 2025-09-16 15:15:00 | 3376.50 | STOP_HIT | 0.50 | 4.93% |
| BUY | retest1 | 2025-09-12 15:00:00 | 3224.80 | 2025-09-16 15:15:00 | 3376.50 | STOP_HIT | 0.50 | 4.70% |
| BUY | retest2 | 2025-09-18 14:15:00 | 3393.00 | 2025-09-19 11:15:00 | 3341.30 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-09-22 13:15:00 | 3345.50 | 2025-09-26 09:15:00 | 3178.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-22 15:15:00 | 3340.00 | 2025-09-26 09:15:00 | 3173.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 09:45:00 | 3350.50 | 2025-09-26 09:15:00 | 3182.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 10:45:00 | 3347.30 | 2025-09-26 09:15:00 | 3179.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 14:15:00 | 3316.90 | 2025-09-26 10:15:00 | 3151.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 11:45:00 | 3310.80 | 2025-09-26 10:15:00 | 3151.43 | PARTIAL | 0.50 | 4.81% |
| SELL | retest2 | 2025-09-24 13:00:00 | 3317.30 | 2025-09-26 11:15:00 | 3145.26 | PARTIAL | 0.50 | 5.19% |
| SELL | retest2 | 2025-09-22 13:15:00 | 3345.50 | 2025-09-30 15:15:00 | 3110.80 | STOP_HIT | 0.50 | 7.02% |
| SELL | retest2 | 2025-09-22 15:15:00 | 3340.00 | 2025-09-30 15:15:00 | 3110.80 | STOP_HIT | 0.50 | 6.86% |
| SELL | retest2 | 2025-09-23 09:45:00 | 3350.50 | 2025-09-30 15:15:00 | 3110.80 | STOP_HIT | 0.50 | 7.15% |
| SELL | retest2 | 2025-09-23 10:45:00 | 3347.30 | 2025-09-30 15:15:00 | 3110.80 | STOP_HIT | 0.50 | 7.07% |
| SELL | retest2 | 2025-09-23 14:15:00 | 3316.90 | 2025-09-30 15:15:00 | 3110.80 | STOP_HIT | 0.50 | 6.21% |
| SELL | retest2 | 2025-09-24 11:45:00 | 3310.80 | 2025-09-30 15:15:00 | 3110.80 | STOP_HIT | 0.50 | 6.04% |
| SELL | retest2 | 2025-09-24 13:00:00 | 3317.30 | 2025-09-30 15:15:00 | 3110.80 | STOP_HIT | 0.50 | 6.22% |
| SELL | retest2 | 2025-10-28 11:15:00 | 3141.80 | 2025-11-04 14:15:00 | 2984.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 13:00:00 | 3145.10 | 2025-11-04 14:15:00 | 2987.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 09:15:00 | 3140.90 | 2025-11-04 15:15:00 | 2983.86 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-28 11:15:00 | 3141.80 | 2025-11-06 09:15:00 | 3041.30 | STOP_HIT | 0.50 | 3.20% |
| SELL | retest2 | 2025-10-28 13:00:00 | 3145.10 | 2025-11-06 09:15:00 | 3041.30 | STOP_HIT | 0.50 | 3.30% |
| SELL | retest2 | 2025-10-30 09:15:00 | 3140.90 | 2025-11-06 09:15:00 | 3041.30 | STOP_HIT | 0.50 | 3.17% |
| BUY | retest2 | 2025-11-14 09:15:00 | 3067.40 | 2025-11-18 13:15:00 | 3047.10 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-11-14 11:00:00 | 3052.80 | 2025-11-18 13:15:00 | 3047.10 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-11-14 12:45:00 | 3052.30 | 2025-11-18 13:15:00 | 3047.10 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-11-14 13:15:00 | 3057.00 | 2025-11-18 13:15:00 | 3047.10 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-11-20 11:30:00 | 3031.80 | 2025-11-21 15:15:00 | 2880.21 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-20 11:30:00 | 3031.80 | 2025-11-24 14:15:00 | 2913.10 | STOP_HIT | 0.50 | 3.92% |
| SELL | retest2 | 2025-12-18 13:30:00 | 2582.00 | 2025-12-19 14:15:00 | 2630.30 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-12-19 10:00:00 | 2583.20 | 2025-12-19 14:15:00 | 2630.30 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2025-12-19 10:45:00 | 2578.30 | 2025-12-19 14:15:00 | 2630.30 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-12-30 12:15:00 | 2563.80 | 2025-12-31 13:15:00 | 2609.30 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-12-30 13:45:00 | 2564.00 | 2025-12-31 13:15:00 | 2609.30 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2026-01-07 11:15:00 | 2540.40 | 2026-01-12 09:15:00 | 2413.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-07 11:15:00 | 2540.40 | 2026-01-16 09:15:00 | 2374.40 | STOP_HIT | 0.50 | 6.53% |
| SELL | retest2 | 2026-01-23 09:15:00 | 2275.90 | 2026-01-28 15:15:00 | 2298.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2026-02-09 11:15:00 | 2348.10 | 2026-02-10 09:15:00 | 2382.00 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2026-02-13 11:45:00 | 2491.10 | 2026-02-17 13:15:00 | 2458.40 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2026-02-16 09:15:00 | 2515.10 | 2026-02-17 13:15:00 | 2458.40 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2026-02-24 09:15:00 | 2556.90 | 2026-03-02 14:15:00 | 2812.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-24 14:30:00 | 2552.30 | 2026-03-02 14:15:00 | 2807.53 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-11 10:15:00 | 2640.70 | 2026-03-12 09:15:00 | 2508.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 10:15:00 | 2640.70 | 2026-03-12 13:15:00 | 2547.40 | STOP_HIT | 0.50 | 3.53% |
| BUY | retest2 | 2026-03-19 10:30:00 | 2522.90 | 2026-03-23 09:15:00 | 2449.20 | STOP_HIT | 1.00 | -2.92% |
| BUY | retest2 | 2026-04-13 10:45:00 | 2726.80 | 2026-04-22 12:15:00 | 2999.48 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 12:30:00 | 2727.90 | 2026-04-22 12:15:00 | 3000.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 13:00:00 | 2726.30 | 2026-04-22 12:15:00 | 2998.93 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-15 09:15:00 | 2751.00 | 2026-04-22 12:15:00 | 3026.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-17 09:15:00 | 2779.00 | 2026-04-22 12:15:00 | 3047.00 | TARGET_HIT | 1.00 | 9.64% |
| BUY | retest2 | 2026-04-17 10:00:00 | 2770.00 | 2026-04-22 12:15:00 | 3046.67 | TARGET_HIT | 1.00 | 9.99% |
| BUY | retest2 | 2026-04-20 09:45:00 | 2769.70 | 2026-04-23 09:15:00 | 3056.90 | TARGET_HIT | 1.00 | 10.37% |
| SELL | retest2 | 2026-05-04 12:15:00 | 2913.60 | 2026-05-07 09:15:00 | 2980.30 | STOP_HIT | 1.00 | -2.29% |
| SELL | retest2 | 2026-05-05 09:15:00 | 2914.50 | 2026-05-07 09:15:00 | 2980.30 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2026-05-05 12:45:00 | 2919.40 | 2026-05-07 09:15:00 | 2980.30 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2026-05-06 09:15:00 | 2914.70 | 2026-05-07 09:15:00 | 2980.30 | STOP_HIT | 1.00 | -2.25% |
| SELL | retest2 | 2026-05-06 11:45:00 | 2882.70 | 2026-05-07 09:15:00 | 2980.30 | STOP_HIT | 1.00 | -3.39% |
| SELL | retest2 | 2026-05-06 12:15:00 | 2879.70 | 2026-05-07 09:15:00 | 2980.30 | STOP_HIT | 1.00 | -3.49% |
| SELL | retest2 | 2026-05-06 13:00:00 | 2886.30 | 2026-05-07 09:15:00 | 2980.30 | STOP_HIT | 1.00 | -3.26% |
