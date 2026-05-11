# Escorts Kubota Ltd. (ESCORTS)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 3148.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 133 |
| ALERT1 | 101 |
| ALERT2 | 99 |
| ALERT2_SKIP | 53 |
| ALERT3 | 259 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 116 |
| PARTIAL | 16 |
| TARGET_HIT | 9 |
| STOP_HIT | 111 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 136 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 61 / 75
- **Target hits / Stop hits / Partials:** 9 / 111 / 16
- **Avg / median % per leg:** 0.79% / -0.40%
- **Sum % (uncompounded):** 107.35%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 49 | 20 | 40.8% | 1 | 47 | 1 | 0.49% | 23.9% |
| BUY @ 2nd Alert (retest1) | 4 | 2 | 50.0% | 0 | 3 | 1 | 1.88% | 7.5% |
| BUY @ 3rd Alert (retest2) | 45 | 18 | 40.0% | 1 | 44 | 0 | 0.37% | 16.4% |
| SELL (all) | 87 | 41 | 47.1% | 8 | 64 | 15 | 0.96% | 83.4% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 0.58% | 0.6% |
| SELL @ 3rd Alert (retest2) | 86 | 40 | 46.5% | 8 | 63 | 15 | 0.96% | 82.8% |
| retest1 (combined) | 5 | 3 | 60.0% | 0 | 4 | 1 | 1.62% | 8.1% |
| retest2 (combined) | 131 | 58 | 44.3% | 9 | 107 | 15 | 0.76% | 99.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 13:15:00 | 3817.85 | 3824.54 | 3825.22 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 10:15:00 | 3827.00 | 3820.13 | 3819.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-27 13:15:00 | 3851.30 | 3829.60 | 3824.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-28 12:15:00 | 3840.00 | 3852.87 | 3840.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-28 12:15:00 | 3840.00 | 3852.87 | 3840.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 12:15:00 | 3840.00 | 3852.87 | 3840.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 12:30:00 | 3824.50 | 3852.87 | 3840.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 13:15:00 | 3825.90 | 3847.48 | 3839.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-28 14:00:00 | 3825.90 | 3847.48 | 3839.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-28 14:15:00 | 3850.90 | 3848.16 | 3840.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 11:00:00 | 3873.90 | 3852.62 | 3844.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 12:15:00 | 3868.45 | 3854.77 | 3845.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 09:15:00 | 3797.00 | 3843.95 | 3844.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 3797.00 | 3843.95 | 3844.88 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 10:15:00 | 3850.20 | 3840.90 | 3840.38 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 14:15:00 | 3819.25 | 3840.23 | 3840.65 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 3877.40 | 3842.50 | 3841.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 11:15:00 | 3918.90 | 3864.85 | 3852.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 3790.55 | 3869.60 | 3863.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 3790.55 | 3869.60 | 3863.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 3790.55 | 3869.60 | 3863.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 3790.55 | 3869.60 | 3863.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 3664.20 | 3828.52 | 3845.06 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 3980.20 | 3866.61 | 3853.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-10 10:15:00 | 4024.25 | 4001.03 | 3971.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 09:15:00 | 4101.85 | 4110.90 | 4071.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-12 10:00:00 | 4101.85 | 4110.90 | 4071.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 10:15:00 | 4099.35 | 4108.59 | 4074.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 10:30:00 | 4083.65 | 4108.59 | 4074.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 11:15:00 | 4066.35 | 4100.14 | 4073.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 12:00:00 | 4066.35 | 4100.14 | 4073.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 12:15:00 | 4095.30 | 4099.17 | 4075.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 13:15:00 | 4100.35 | 4099.17 | 4075.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 15:15:00 | 4229.75 | 4251.41 | 4253.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 15:15:00 | 4229.75 | 4251.41 | 4253.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 12:15:00 | 4192.60 | 4230.39 | 4242.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 4220.85 | 4213.11 | 4228.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-21 10:00:00 | 4220.85 | 4213.11 | 4228.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 10:15:00 | 4245.15 | 4219.52 | 4230.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 10:45:00 | 4254.40 | 4219.52 | 4230.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 11:15:00 | 4285.35 | 4232.69 | 4235.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-21 11:30:00 | 4286.00 | 4232.69 | 4235.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2024-06-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 12:15:00 | 4288.05 | 4243.76 | 4240.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 13:15:00 | 4383.80 | 4271.77 | 4253.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-24 11:15:00 | 4294.65 | 4297.49 | 4275.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-24 12:15:00 | 4270.50 | 4297.49 | 4275.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 4268.05 | 4291.60 | 4274.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 12:45:00 | 4252.00 | 4291.60 | 4274.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 13:15:00 | 4251.40 | 4283.56 | 4272.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 14:00:00 | 4251.40 | 4283.56 | 4272.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 09:15:00 | 4222.75 | 4261.50 | 4264.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 10:15:00 | 4211.45 | 4251.49 | 4259.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 09:15:00 | 4240.70 | 4220.32 | 4236.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 09:15:00 | 4240.70 | 4220.32 | 4236.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 4240.70 | 4220.32 | 4236.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:00:00 | 4240.70 | 4220.32 | 4236.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 10:15:00 | 4252.25 | 4226.71 | 4237.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 10:30:00 | 4252.95 | 4226.71 | 4237.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 11:15:00 | 4258.05 | 4232.98 | 4239.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 11:45:00 | 4259.95 | 4232.98 | 4239.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 13:15:00 | 4255.60 | 4240.23 | 4241.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 13:45:00 | 4257.30 | 4240.23 | 4241.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2024-06-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-26 14:15:00 | 4257.45 | 4243.67 | 4243.37 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-06-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-27 09:15:00 | 4223.15 | 4241.84 | 4242.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-27 10:15:00 | 4199.45 | 4233.36 | 4238.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-28 10:15:00 | 4223.50 | 4194.69 | 4211.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-28 10:15:00 | 4223.50 | 4194.69 | 4211.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 4223.50 | 4194.69 | 4211.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 11:00:00 | 4223.50 | 4194.69 | 4211.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 4226.30 | 4201.01 | 4212.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 11:30:00 | 4238.90 | 4201.01 | 4212.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 10:15:00 | 4184.65 | 4176.85 | 4193.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 10:30:00 | 4184.10 | 4176.85 | 4193.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 4153.15 | 4134.31 | 4149.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 09:30:00 | 4150.70 | 4134.31 | 4149.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 10:15:00 | 4110.00 | 4129.45 | 4146.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 12:00:00 | 4102.40 | 4123.76 | 4134.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 13:00:00 | 4097.15 | 4118.44 | 4130.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-04 14:45:00 | 4103.90 | 4113.77 | 4126.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-05 12:15:00 | 4209.50 | 4137.71 | 4132.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2024-07-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 12:15:00 | 4209.50 | 4137.71 | 4132.86 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 09:15:00 | 4073.75 | 4126.04 | 4129.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 10:15:00 | 4041.50 | 4109.13 | 4121.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-08 12:15:00 | 4105.00 | 4104.73 | 4116.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-08 13:00:00 | 4105.00 | 4104.73 | 4116.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 14:15:00 | 4113.65 | 4106.56 | 4115.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-08 14:30:00 | 4109.35 | 4106.56 | 4115.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 15:15:00 | 4094.05 | 4104.06 | 4113.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:15:00 | 4125.45 | 4104.06 | 4113.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 4112.80 | 4105.81 | 4113.64 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2024-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 12:15:00 | 4153.85 | 4121.19 | 4119.07 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 4062.15 | 4118.26 | 4120.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 13:15:00 | 4032.00 | 4086.25 | 4103.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-15 14:15:00 | 3971.85 | 3928.45 | 3958.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-15 14:15:00 | 3971.85 | 3928.45 | 3958.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 3971.85 | 3928.45 | 3958.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 15:00:00 | 3971.85 | 3928.45 | 3958.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 3990.00 | 3940.76 | 3961.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:15:00 | 4055.85 | 3940.76 | 3961.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2024-07-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 10:15:00 | 4034.15 | 3974.19 | 3973.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-16 12:15:00 | 4036.15 | 3990.23 | 3981.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 09:15:00 | 3971.00 | 3998.39 | 3989.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 09:15:00 | 3971.00 | 3998.39 | 3989.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 3971.00 | 3998.39 | 3989.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:30:00 | 3977.80 | 3998.39 | 3989.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 3921.00 | 3982.91 | 3983.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 11:15:00 | 3899.00 | 3966.13 | 3975.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 13:15:00 | 3982.15 | 3965.18 | 3973.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 13:15:00 | 3982.15 | 3965.18 | 3973.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 3982.15 | 3965.18 | 3973.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:00:00 | 3982.15 | 3965.18 | 3973.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 3990.20 | 3970.18 | 3974.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 15:00:00 | 3990.20 | 3970.18 | 3974.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 15:15:00 | 3989.60 | 3974.06 | 3976.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 09:15:00 | 3971.85 | 3974.06 | 3976.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 3977.30 | 3969.87 | 3973.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:30:00 | 4012.50 | 3969.87 | 3973.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 11:15:00 | 3955.05 | 3966.91 | 3971.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 13:00:00 | 3937.25 | 3960.97 | 3968.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 14:30:00 | 3936.30 | 3950.93 | 3962.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-22 09:15:00 | 3982.90 | 3951.73 | 3960.69 | SL hit (close>static) qty=1.00 sl=3977.30 alert=retest2 |

### Cycle 20 — BUY (started 2024-07-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 11:15:00 | 4003.75 | 3969.28 | 3967.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 11:15:00 | 4050.90 | 4008.82 | 3991.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-24 12:15:00 | 4086.50 | 4090.27 | 4051.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-24 12:45:00 | 4077.00 | 4090.27 | 4051.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 4083.10 | 4112.51 | 4083.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 12:00:00 | 4083.10 | 4112.51 | 4083.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 4077.60 | 4105.53 | 4082.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 12:45:00 | 4078.80 | 4105.53 | 4082.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 4058.20 | 4096.06 | 4080.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 14:00:00 | 4058.20 | 4096.06 | 4080.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 14:15:00 | 4058.95 | 4088.64 | 4078.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 14:45:00 | 4050.00 | 4088.64 | 4078.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 4075.30 | 4080.60 | 4076.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-26 14:15:00 | 4095.95 | 4076.29 | 4075.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 15:15:00 | 4130.00 | 4154.98 | 4156.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2024-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 15:15:00 | 4130.00 | 4154.98 | 4156.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 4045.05 | 4133.00 | 4146.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 3832.75 | 3811.34 | 3905.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 3832.75 | 3811.34 | 3905.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 3832.75 | 3811.34 | 3905.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 12:15:00 | 3754.05 | 3806.43 | 3886.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-13 09:15:00 | 3763.95 | 3719.76 | 3718.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 09:15:00 | 3763.95 | 3719.76 | 3718.55 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2024-08-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 14:15:00 | 3704.25 | 3720.00 | 3720.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 15:15:00 | 3694.90 | 3714.98 | 3717.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 3694.50 | 3681.59 | 3694.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 3694.50 | 3681.59 | 3694.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 3694.50 | 3681.59 | 3694.69 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 13:15:00 | 3728.00 | 3703.90 | 3702.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 14:15:00 | 3736.45 | 3710.41 | 3705.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 15:15:00 | 3725.60 | 3727.38 | 3719.18 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-08-20 09:15:00 | 3751.25 | 3727.38 | 3719.18 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 3728.30 | 3727.56 | 3720.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 10:00:00 | 3728.30 | 3727.56 | 3720.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 3712.70 | 3724.59 | 3719.34 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-08-20 10:15:00 | 3712.70 | 3724.59 | 3719.34 | SL hit (close<ema400) qty=1.00 sl=3719.34 alert=retest1 |

### Cycle 25 — SELL (started 2024-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 15:15:00 | 3830.85 | 3858.91 | 3859.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 10:15:00 | 3815.20 | 3848.42 | 3854.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 3846.40 | 3827.46 | 3838.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 3846.40 | 3827.46 | 3838.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 3846.40 | 3827.46 | 3838.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:45:00 | 3847.15 | 3827.46 | 3838.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 3853.95 | 3832.75 | 3839.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:30:00 | 3849.95 | 3832.75 | 3839.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 13:15:00 | 3850.60 | 3840.41 | 3841.88 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 14:15:00 | 3856.90 | 3843.71 | 3843.24 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2024-09-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-02 09:15:00 | 3801.65 | 3836.30 | 3840.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-02 10:15:00 | 3795.00 | 3828.04 | 3835.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-03 09:15:00 | 3810.00 | 3799.28 | 3814.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 09:15:00 | 3810.00 | 3799.28 | 3814.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 09:15:00 | 3810.00 | 3799.28 | 3814.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 09:45:00 | 3811.85 | 3799.28 | 3814.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 10:15:00 | 3822.25 | 3803.87 | 3815.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-03 11:00:00 | 3822.25 | 3803.87 | 3815.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 11:15:00 | 3787.45 | 3800.59 | 3812.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 11:00:00 | 3768.65 | 3783.36 | 3793.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 14:15:00 | 3745.00 | 3717.76 | 3715.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 14:15:00 | 3745.00 | 3717.76 | 3715.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 09:15:00 | 3778.65 | 3734.38 | 3724.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 3836.65 | 3839.32 | 3818.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 10:00:00 | 3836.65 | 3839.32 | 3818.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 3846.00 | 3870.57 | 3852.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:00:00 | 3846.00 | 3870.57 | 3852.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 3799.50 | 3856.36 | 3847.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:30:00 | 3809.90 | 3856.36 | 3847.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 15:15:00 | 3788.95 | 3833.97 | 3838.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 11:15:00 | 3774.70 | 3813.69 | 3827.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 3820.90 | 3811.07 | 3822.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 14:15:00 | 3820.90 | 3811.07 | 3822.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 3820.90 | 3811.07 | 3822.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 15:00:00 | 3820.90 | 3811.07 | 3822.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 3813.00 | 3811.46 | 3821.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:15:00 | 3872.65 | 3811.46 | 3821.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2024-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 09:15:00 | 3919.25 | 3833.01 | 3830.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 10:15:00 | 4007.55 | 3867.92 | 3846.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 09:15:00 | 4335.00 | 4343.55 | 4277.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-26 09:30:00 | 4336.40 | 4343.55 | 4277.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 4391.65 | 4344.78 | 4303.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 09:45:00 | 4394.25 | 4357.36 | 4313.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 10:15:00 | 4242.90 | 4308.73 | 4309.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2024-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 10:15:00 | 4242.90 | 4308.73 | 4309.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 09:15:00 | 4194.35 | 4259.40 | 4281.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-09 09:15:00 | 3993.50 | 3898.70 | 3946.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-09 09:15:00 | 3993.50 | 3898.70 | 3946.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 3993.50 | 3898.70 | 3946.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:30:00 | 3983.60 | 3898.70 | 3946.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 4045.45 | 3928.05 | 3955.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 11:00:00 | 4045.45 | 3928.05 | 3955.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 13:15:00 | 4047.70 | 3980.06 | 3974.53 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2024-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-11 09:15:00 | 3952.25 | 3982.18 | 3985.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 12:15:00 | 3912.50 | 3957.38 | 3972.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 15:15:00 | 3950.00 | 3947.10 | 3963.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-14 09:15:00 | 3914.55 | 3947.10 | 3963.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 3903.55 | 3938.39 | 3957.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-15 13:00:00 | 3893.80 | 3935.37 | 3947.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-16 11:30:00 | 3883.85 | 3918.24 | 3933.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:15:00 | 3699.11 | 3756.84 | 3787.36 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 14:15:00 | 3689.66 | 3721.39 | 3759.59 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-23 12:15:00 | 3713.55 | 3707.01 | 3736.10 | SL hit (close>ema200) qty=0.50 sl=3707.01 alert=retest2 |

### Cycle 34 — BUY (started 2024-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 12:15:00 | 3655.75 | 3570.14 | 3558.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 11:15:00 | 3686.50 | 3618.53 | 3588.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 14:15:00 | 3626.05 | 3631.62 | 3603.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 15:00:00 | 3626.05 | 3631.62 | 3603.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 3646.80 | 3636.80 | 3610.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 10:30:00 | 3706.40 | 3656.43 | 3621.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 12:45:00 | 3681.05 | 3701.82 | 3696.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 13:15:00 | 3683.70 | 3701.82 | 3696.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-07 10:15:00 | 3625.10 | 3697.00 | 3703.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 10:15:00 | 3625.10 | 3697.00 | 3703.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 10:15:00 | 3587.30 | 3641.08 | 3667.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-08 12:15:00 | 3655.45 | 3634.61 | 3659.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-08 12:45:00 | 3646.00 | 3634.61 | 3659.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 13:15:00 | 3630.85 | 3633.86 | 3656.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 14:15:00 | 3626.75 | 3633.86 | 3656.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-08 15:15:00 | 3625.00 | 3635.47 | 3655.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-11 15:00:00 | 3619.80 | 3635.83 | 3646.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-12 10:15:00 | 3667.65 | 3640.80 | 3646.03 | SL hit (close>static) qty=1.00 sl=3658.25 alert=retest2 |

### Cycle 36 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 3523.50 | 3488.98 | 3485.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 3632.00 | 3524.83 | 3503.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 15:15:00 | 3602.30 | 3611.66 | 3585.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 09:15:00 | 3588.70 | 3611.66 | 3585.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 3616.05 | 3612.53 | 3588.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 09:30:00 | 3605.40 | 3612.53 | 3588.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 3595.60 | 3611.20 | 3596.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 14:00:00 | 3595.60 | 3611.20 | 3596.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 14:15:00 | 3599.95 | 3608.95 | 3596.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 15:15:00 | 3590.00 | 3608.95 | 3596.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 15:15:00 | 3590.00 | 3605.16 | 3595.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 09:15:00 | 3609.85 | 3605.16 | 3595.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 10:15:00 | 3606.85 | 3604.83 | 3596.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 10:15:00 | 3570.50 | 3597.96 | 3594.13 | SL hit (close<static) qty=1.00 sl=3590.00 alert=retest2 |

### Cycle 37 — SELL (started 2024-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 11:15:00 | 3556.00 | 3589.57 | 3590.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 12:15:00 | 3512.75 | 3574.21 | 3583.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 10:15:00 | 3538.70 | 3538.24 | 3559.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-29 11:00:00 | 3538.70 | 3538.24 | 3559.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 3550.00 | 3540.59 | 3558.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:45:00 | 3554.65 | 3540.59 | 3558.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 3559.65 | 3545.92 | 3557.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 14:00:00 | 3559.65 | 3545.92 | 3557.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 14:15:00 | 3556.55 | 3548.04 | 3557.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 14:30:00 | 3559.25 | 3548.04 | 3557.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 3555.00 | 3549.43 | 3557.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 09:15:00 | 3518.85 | 3549.43 | 3557.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 09:45:00 | 3539.05 | 3544.82 | 3554.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 12:15:00 | 3517.20 | 3475.79 | 3475.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2024-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 12:15:00 | 3517.20 | 3475.79 | 3475.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 10:15:00 | 3579.70 | 3519.68 | 3499.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 12:15:00 | 3532.35 | 3541.51 | 3525.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 12:15:00 | 3532.35 | 3541.51 | 3525.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 12:15:00 | 3532.35 | 3541.51 | 3525.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 12:30:00 | 3530.00 | 3541.51 | 3525.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 13:15:00 | 3512.15 | 3535.64 | 3524.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 14:00:00 | 3512.15 | 3535.64 | 3524.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 3510.65 | 3530.64 | 3523.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-10 15:15:00 | 3526.50 | 3530.64 | 3523.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 13:15:00 | 3488.75 | 3519.23 | 3521.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2024-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-11 13:15:00 | 3488.75 | 3519.23 | 3521.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 09:15:00 | 3461.40 | 3500.61 | 3511.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 14:15:00 | 3443.15 | 3432.27 | 3455.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 15:00:00 | 3443.15 | 3432.27 | 3455.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 3431.45 | 3434.43 | 3452.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 10:15:00 | 3426.00 | 3434.43 | 3452.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:30:00 | 3415.45 | 3425.58 | 3438.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 10:15:00 | 3254.70 | 3286.69 | 3321.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-20 11:15:00 | 3244.68 | 3278.30 | 3314.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-23 10:15:00 | 3217.45 | 3215.98 | 3260.67 | SL hit (close>ema200) qty=0.50 sl=3215.98 alert=retest2 |

### Cycle 40 — BUY (started 2024-12-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 10:15:00 | 3271.15 | 3193.72 | 3185.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 10:15:00 | 3306.95 | 3270.21 | 3236.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 14:15:00 | 3257.30 | 3277.70 | 3251.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-30 15:00:00 | 3257.30 | 3277.70 | 3251.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 3263.10 | 3274.78 | 3252.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:15:00 | 3274.55 | 3274.78 | 3252.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 3294.45 | 3278.71 | 3256.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 10:15:00 | 3311.50 | 3278.71 | 3256.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 09:15:00 | 3226.50 | 3295.53 | 3280.13 | SL hit (close<static) qty=1.00 sl=3250.00 alert=retest2 |

### Cycle 41 — SELL (started 2025-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 15:15:00 | 3317.05 | 3337.71 | 3338.31 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2025-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 09:15:00 | 3354.15 | 3341.00 | 3339.75 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2025-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 15:15:00 | 3331.00 | 3338.49 | 3339.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 09:15:00 | 3318.05 | 3334.40 | 3337.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 13:15:00 | 3339.10 | 3315.38 | 3325.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-08 13:15:00 | 3339.10 | 3315.38 | 3325.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 13:15:00 | 3339.10 | 3315.38 | 3325.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 14:00:00 | 3339.10 | 3315.38 | 3325.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 3345.05 | 3321.32 | 3327.33 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2025-01-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 09:15:00 | 3346.00 | 3331.15 | 3331.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-09 12:15:00 | 3405.00 | 3352.07 | 3341.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 15:15:00 | 3398.05 | 3398.66 | 3380.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-13 09:15:00 | 3373.45 | 3398.66 | 3380.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 3356.60 | 3390.24 | 3378.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 10:00:00 | 3356.60 | 3390.24 | 3378.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 10:15:00 | 3340.00 | 3380.20 | 3374.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 11:00:00 | 3340.00 | 3380.20 | 3374.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2025-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 11:15:00 | 3323.55 | 3368.87 | 3369.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 12:15:00 | 3313.20 | 3357.73 | 3364.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 09:15:00 | 3373.55 | 3337.49 | 3350.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-14 09:15:00 | 3373.55 | 3337.49 | 3350.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 3373.55 | 3337.49 | 3350.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 10:00:00 | 3373.55 | 3337.49 | 3350.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 3427.45 | 3355.48 | 3357.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-14 11:00:00 | 3427.45 | 3355.48 | 3357.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2025-01-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-14 11:15:00 | 3426.05 | 3369.60 | 3363.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-14 13:15:00 | 3450.85 | 3395.17 | 3376.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 09:15:00 | 3563.65 | 3575.14 | 3539.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-20 10:00:00 | 3563.65 | 3575.14 | 3539.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 3583.60 | 3576.83 | 3543.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:30:00 | 3543.55 | 3576.83 | 3543.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 3542.00 | 3565.95 | 3554.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 11:00:00 | 3542.00 | 3565.95 | 3554.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 11:15:00 | 3602.50 | 3573.26 | 3558.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 10:15:00 | 3623.05 | 3576.98 | 3571.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 10:15:00 | 3549.05 | 3574.00 | 3575.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 10:15:00 | 3549.05 | 3574.00 | 3575.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 12:15:00 | 3522.45 | 3560.51 | 3568.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 3473.95 | 3413.86 | 3452.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 11:15:00 | 3473.95 | 3413.86 | 3452.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 11:15:00 | 3473.95 | 3413.86 | 3452.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 12:00:00 | 3473.95 | 3413.86 | 3452.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 3484.95 | 3428.08 | 3455.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 3484.95 | 3428.08 | 3455.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 09:15:00 | 3519.15 | 3468.03 | 3467.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 10:15:00 | 3530.00 | 3480.42 | 3473.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 09:15:00 | 3520.55 | 3586.85 | 3563.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 09:15:00 | 3520.55 | 3586.85 | 3563.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 09:15:00 | 3520.55 | 3586.85 | 3563.66 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 3482.00 | 3541.55 | 3546.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-01 13:15:00 | 3474.35 | 3528.11 | 3540.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 09:15:00 | 3370.00 | 3328.58 | 3370.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-05 09:15:00 | 3370.00 | 3328.58 | 3370.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 3370.00 | 3328.58 | 3370.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:00:00 | 3370.00 | 3328.58 | 3370.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 3355.95 | 3334.06 | 3369.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 10:30:00 | 3372.35 | 3334.06 | 3369.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 3316.20 | 3329.83 | 3352.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 11:00:00 | 3313.20 | 3326.51 | 3348.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 11:45:00 | 3311.75 | 3288.16 | 3311.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 10:00:00 | 3302.20 | 3281.33 | 3298.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 14:15:00 | 3315.10 | 3305.88 | 3306.14 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 3300.00 | 3304.08 | 3305.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 09:15:00 | 3218.40 | 3304.08 | 3305.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 3147.54 | 3284.45 | 3296.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 3146.16 | 3284.45 | 3296.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 3137.09 | 3284.45 | 3296.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 3149.34 | 3284.45 | 3296.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-12 09:15:00 | 3057.48 | 3138.61 | 3205.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-13 09:15:00 | 2981.88 | 3057.83 | 3124.90 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 50 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 3016.85 | 2995.74 | 2993.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 3051.95 | 3021.99 | 3009.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 3077.70 | 3084.33 | 3053.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-21 10:00:00 | 3077.70 | 3084.33 | 3053.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 14:15:00 | 3062.05 | 3075.74 | 3060.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 14:30:00 | 3064.35 | 3075.74 | 3060.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 15:15:00 | 3056.60 | 3071.91 | 3060.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:15:00 | 3020.15 | 3071.91 | 3060.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 09:15:00 | 3021.70 | 3061.87 | 3056.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-24 09:30:00 | 2998.75 | 3061.87 | 3056.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 2976.35 | 3044.77 | 3049.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 11:15:00 | 2958.75 | 3027.56 | 3041.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-27 15:15:00 | 2928.60 | 2928.31 | 2953.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-28 09:15:00 | 2910.00 | 2928.31 | 2953.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 09:15:00 | 2874.45 | 2917.54 | 2946.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 10:45:00 | 2862.20 | 2904.63 | 2937.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-03 15:15:00 | 2951.50 | 2921.13 | 2917.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2025-03-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 15:15:00 | 2951.50 | 2921.13 | 2917.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 10:15:00 | 2955.95 | 2930.94 | 2922.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-04 12:15:00 | 2931.95 | 2932.80 | 2925.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-04 12:15:00 | 2931.95 | 2932.80 | 2925.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 12:15:00 | 2931.95 | 2932.80 | 2925.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-04 12:45:00 | 2932.00 | 2932.80 | 2925.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 13:15:00 | 2952.15 | 2936.67 | 2927.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-04 13:30:00 | 2928.65 | 2936.67 | 2927.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 11:15:00 | 3024.50 | 3055.68 | 3030.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:00:00 | 3024.50 | 3055.68 | 3030.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 3030.30 | 3050.61 | 3030.17 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2025-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 10:15:00 | 2965.00 | 3016.02 | 3019.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 11:15:00 | 2943.25 | 3001.46 | 3012.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 09:15:00 | 2963.95 | 2922.05 | 2946.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 09:15:00 | 2963.95 | 2922.05 | 2946.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 2963.95 | 2922.05 | 2946.35 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2025-03-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 13:15:00 | 2974.60 | 2940.27 | 2935.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 3001.00 | 2958.04 | 2945.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 09:15:00 | 3236.65 | 3250.17 | 3215.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:00:00 | 3236.65 | 3250.17 | 3215.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 3217.30 | 3245.89 | 3220.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:45:00 | 3225.80 | 3245.89 | 3220.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 3210.15 | 3238.74 | 3219.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 13:45:00 | 3219.95 | 3238.87 | 3220.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 15:15:00 | 3233.20 | 3232.85 | 3219.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:45:00 | 3222.00 | 3230.86 | 3221.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 11:30:00 | 3221.65 | 3233.84 | 3223.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 14:15:00 | 3236.45 | 3245.98 | 3233.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-26 15:00:00 | 3236.45 | 3245.98 | 3233.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 15:15:00 | 3231.00 | 3242.98 | 3232.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:15:00 | 3244.30 | 3242.98 | 3232.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 3254.70 | 3245.33 | 3234.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 10:30:00 | 3265.80 | 3248.02 | 3237.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 12:00:00 | 3261.20 | 3250.66 | 3239.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 09:15:00 | 3288.50 | 3252.38 | 3243.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 14:15:00 | 3265.85 | 3261.47 | 3253.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 3255.55 | 3260.28 | 3253.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 15:15:00 | 3247.00 | 3260.28 | 3253.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 3247.00 | 3257.63 | 3252.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:15:00 | 3267.30 | 3257.63 | 3252.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 3217.75 | 3249.65 | 3249.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:00:00 | 3217.75 | 3249.65 | 3249.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-01 10:15:00 | 3183.15 | 3236.35 | 3243.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 10:15:00 | 3183.15 | 3236.35 | 3243.58 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 11:15:00 | 3247.40 | 3234.72 | 3233.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 13:15:00 | 3261.30 | 3240.84 | 3236.81 | Break + close above crossover candle high |

### Cycle 57 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 3159.25 | 3233.96 | 3235.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 11:15:00 | 3157.95 | 3207.62 | 3222.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 3066.10 | 3060.78 | 3107.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 3066.10 | 3060.78 | 3107.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 3101.75 | 3076.19 | 3106.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:30:00 | 3097.80 | 3076.19 | 3106.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 3070.55 | 3075.06 | 3103.53 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2025-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 12:15:00 | 3167.00 | 3111.59 | 3109.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 3175.65 | 3132.55 | 3121.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 12:15:00 | 3222.60 | 3245.09 | 3216.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 13:00:00 | 3222.60 | 3245.09 | 3216.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 13:15:00 | 3223.40 | 3240.75 | 3216.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 13:30:00 | 3221.00 | 3240.75 | 3216.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 14:15:00 | 3229.10 | 3238.42 | 3217.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 14:30:00 | 3220.30 | 3238.42 | 3217.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 3223.00 | 3235.59 | 3220.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:15:00 | 3257.50 | 3239.14 | 3224.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 10:15:00 | 3321.90 | 3371.07 | 3374.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 3321.90 | 3371.07 | 3374.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 11:15:00 | 3284.40 | 3326.93 | 3341.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 3308.20 | 3288.83 | 3313.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-02 10:00:00 | 3308.20 | 3288.83 | 3313.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 3304.30 | 3291.92 | 3312.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 10:30:00 | 3317.30 | 3291.92 | 3312.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 3191.70 | 3230.97 | 3252.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 15:15:00 | 3180.10 | 3216.67 | 3237.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 09:30:00 | 3173.50 | 3200.27 | 3225.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 10:45:00 | 3181.90 | 3196.79 | 3221.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 12:15:00 | 3178.20 | 3195.83 | 3219.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 3186.50 | 3197.59 | 3211.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 12:00:00 | 3137.20 | 3182.55 | 3202.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 09:45:00 | 3131.40 | 3166.49 | 3184.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 10:30:00 | 3145.10 | 3158.99 | 3179.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 09:15:00 | 3358.80 | 3182.59 | 3178.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 3358.80 | 3182.59 | 3178.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 3388.30 | 3290.61 | 3236.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 12:15:00 | 3550.00 | 3551.36 | 3516.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-16 13:00:00 | 3550.00 | 3551.36 | 3516.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 3530.90 | 3546.01 | 3525.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 3515.70 | 3546.01 | 3525.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 3535.90 | 3543.99 | 3526.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 11:30:00 | 3541.20 | 3542.79 | 3527.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 10:15:00 | 3515.10 | 3529.58 | 3526.77 | SL hit (close<static) qty=1.00 sl=3521.40 alert=retest2 |

### Cycle 61 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 3516.70 | 3524.51 | 3524.80 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 3569.60 | 3528.04 | 3525.50 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 14:15:00 | 3518.60 | 3524.94 | 3525.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 15:15:00 | 3501.40 | 3520.23 | 3523.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 14:15:00 | 3495.50 | 3488.76 | 3502.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 14:15:00 | 3495.50 | 3488.76 | 3502.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 3495.50 | 3488.76 | 3502.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 15:00:00 | 3495.50 | 3488.76 | 3502.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 3523.10 | 3495.63 | 3504.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:15:00 | 3544.80 | 3495.63 | 3504.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 3536.00 | 3503.70 | 3507.01 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2025-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 10:15:00 | 3546.00 | 3512.16 | 3510.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 11:15:00 | 3579.60 | 3525.65 | 3516.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 14:15:00 | 3514.00 | 3527.88 | 3520.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 14:15:00 | 3514.00 | 3527.88 | 3520.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 14:15:00 | 3514.00 | 3527.88 | 3520.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 15:00:00 | 3514.00 | 3527.88 | 3520.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 3497.00 | 3521.71 | 3518.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:15:00 | 3506.10 | 3521.71 | 3518.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 10:15:00 | 3458.40 | 3505.16 | 3511.13 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2025-05-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 14:15:00 | 3537.50 | 3510.33 | 3507.45 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 09:15:00 | 3449.70 | 3502.55 | 3504.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 12:15:00 | 3415.00 | 3469.93 | 3487.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 13:15:00 | 3343.90 | 3334.27 | 3369.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-30 13:30:00 | 3337.80 | 3334.27 | 3369.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 3356.60 | 3334.74 | 3360.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 10:00:00 | 3356.60 | 3334.74 | 3360.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 3383.00 | 3344.39 | 3362.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:00:00 | 3383.00 | 3344.39 | 3362.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 3388.50 | 3353.21 | 3364.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 12:15:00 | 3410.00 | 3353.21 | 3364.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2025-06-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 14:15:00 | 3387.00 | 3374.67 | 3373.11 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 09:15:00 | 3342.70 | 3368.81 | 3370.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 10:15:00 | 3330.00 | 3361.05 | 3367.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 09:15:00 | 3283.00 | 3269.25 | 3296.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 09:15:00 | 3283.00 | 3269.25 | 3296.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 3283.00 | 3269.25 | 3296.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 13:15:00 | 3259.80 | 3283.11 | 3289.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 09:30:00 | 3258.10 | 3268.98 | 3279.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 10:15:00 | 3246.20 | 3268.98 | 3279.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-13 09:15:00 | 2933.82 | 3169.67 | 3206.07 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2025-06-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 09:15:00 | 3200.30 | 3158.16 | 3157.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 10:15:00 | 3217.80 | 3170.09 | 3163.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 15:15:00 | 3238.00 | 3247.75 | 3225.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 12:15:00 | 3242.10 | 3246.83 | 3232.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 3242.10 | 3246.83 | 3232.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:30:00 | 3246.10 | 3246.83 | 3232.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 3224.20 | 3241.75 | 3232.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 3224.20 | 3241.75 | 3232.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 3254.90 | 3244.38 | 3234.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 3256.00 | 3244.38 | 3234.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 12:15:00 | 3327.60 | 3341.35 | 3341.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 12:15:00 | 3327.60 | 3341.35 | 3341.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 15:15:00 | 3318.20 | 3330.80 | 3336.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 09:15:00 | 3366.00 | 3337.84 | 3339.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 09:15:00 | 3366.00 | 3337.84 | 3339.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 3366.00 | 3337.84 | 3339.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:30:00 | 3370.50 | 3337.84 | 3339.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 10:15:00 | 3366.30 | 3343.53 | 3341.58 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 12:15:00 | 3332.90 | 3353.47 | 3354.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 09:15:00 | 3316.20 | 3336.07 | 3344.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 14:15:00 | 3338.60 | 3319.33 | 3331.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 14:15:00 | 3338.60 | 3319.33 | 3331.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 3338.60 | 3319.33 | 3331.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 15:00:00 | 3338.60 | 3319.33 | 3331.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 3343.60 | 3324.18 | 3332.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 3366.50 | 3324.18 | 3332.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 3352.50 | 3329.69 | 3333.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:45:00 | 3355.40 | 3329.69 | 3333.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2025-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-08 11:15:00 | 3383.90 | 3340.53 | 3337.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 14:15:00 | 3390.80 | 3361.99 | 3349.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-09 15:15:00 | 3372.00 | 3376.39 | 3365.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 09:15:00 | 3368.00 | 3376.39 | 3365.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 09:15:00 | 3346.00 | 3370.31 | 3363.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 10:00:00 | 3346.00 | 3370.31 | 3363.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 10:15:00 | 3342.40 | 3364.73 | 3361.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:00:00 | 3342.40 | 3364.73 | 3361.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — SELL (started 2025-07-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 12:15:00 | 3347.10 | 3357.53 | 3358.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 14:15:00 | 3324.50 | 3348.31 | 3354.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 3329.80 | 3288.73 | 3302.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 3329.80 | 3288.73 | 3302.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 3329.80 | 3288.73 | 3302.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:15:00 | 3353.00 | 3288.73 | 3302.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 3355.70 | 3302.12 | 3307.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:30:00 | 3346.60 | 3302.12 | 3307.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2025-07-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 12:15:00 | 3350.80 | 3319.21 | 3314.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 3364.00 | 3333.56 | 3322.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 15:15:00 | 3409.90 | 3412.67 | 3394.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-21 09:15:00 | 3414.00 | 3412.67 | 3394.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 3424.60 | 3415.06 | 3396.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 15:15:00 | 3434.00 | 3418.20 | 3405.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 10:15:00 | 3448.90 | 3419.75 | 3408.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 13:15:00 | 3437.70 | 3429.00 | 3416.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 12:00:00 | 3450.00 | 3456.75 | 3446.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 12:15:00 | 3451.80 | 3455.76 | 3447.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 12:30:00 | 3449.60 | 3455.76 | 3447.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 13:15:00 | 3444.00 | 3453.41 | 3446.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 14:00:00 | 3444.00 | 3453.41 | 3446.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 14:15:00 | 3442.30 | 3451.19 | 3446.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 15:00:00 | 3442.30 | 3451.19 | 3446.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 3440.00 | 3448.95 | 3445.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 3436.60 | 3448.95 | 3445.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 3432.90 | 3445.74 | 3444.64 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 3425.00 | 3441.59 | 3442.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 3425.00 | 3441.59 | 3442.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 11:15:00 | 3418.40 | 3436.95 | 3440.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 3444.00 | 3427.19 | 3433.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 3444.00 | 3427.19 | 3433.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 3444.00 | 3427.19 | 3433.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 14:30:00 | 3386.00 | 3425.03 | 3430.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 09:15:00 | 3485.00 | 3435.07 | 3433.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2025-07-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 09:15:00 | 3485.00 | 3435.07 | 3433.83 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 14:15:00 | 3432.50 | 3449.58 | 3450.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 3385.30 | 3434.71 | 3443.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 10:15:00 | 3327.20 | 3322.96 | 3358.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 10:30:00 | 3319.00 | 3322.96 | 3358.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 3417.40 | 3339.69 | 3357.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 3417.40 | 3339.69 | 3357.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 3391.00 | 3349.95 | 3360.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 3415.00 | 3349.95 | 3360.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 3409.20 | 3369.42 | 3367.81 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 3334.00 | 3379.39 | 3382.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 3316.00 | 3353.30 | 3368.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 3350.10 | 3347.81 | 3363.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 3350.10 | 3347.81 | 3363.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 3350.10 | 3347.81 | 3363.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 3350.10 | 3347.81 | 3363.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 3337.50 | 3347.40 | 3360.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:30:00 | 3324.90 | 3348.26 | 3359.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 09:30:00 | 3322.00 | 3349.64 | 3356.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 15:15:00 | 3369.00 | 3357.52 | 3357.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — BUY (started 2025-08-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 15:15:00 | 3369.00 | 3357.52 | 3357.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 14:15:00 | 3383.00 | 3367.11 | 3362.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 11:15:00 | 3368.60 | 3369.71 | 3365.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 11:15:00 | 3368.60 | 3369.71 | 3365.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 3368.60 | 3369.71 | 3365.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:45:00 | 3364.90 | 3369.71 | 3365.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 3372.20 | 3370.21 | 3366.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 12:30:00 | 3368.30 | 3370.21 | 3366.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 3369.60 | 3370.08 | 3366.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:00:00 | 3369.60 | 3370.08 | 3366.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 3383.00 | 3374.12 | 3369.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 10:15:00 | 3411.50 | 3374.12 | 3369.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 3479.40 | 3393.24 | 3383.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 10:15:00 | 3548.00 | 3577.94 | 3579.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 3548.00 | 3577.94 | 3579.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 11:15:00 | 3536.30 | 3569.61 | 3576.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 14:15:00 | 3568.80 | 3565.73 | 3572.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 14:15:00 | 3568.80 | 3565.73 | 3572.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 14:15:00 | 3568.80 | 3565.73 | 3572.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 15:00:00 | 3568.80 | 3565.73 | 3572.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 3574.00 | 3567.39 | 3572.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:15:00 | 3564.00 | 3567.39 | 3572.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 3567.20 | 3567.35 | 3571.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 09:15:00 | 3531.70 | 3564.88 | 3569.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 12:30:00 | 3547.00 | 3555.44 | 3562.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 3530.40 | 3554.89 | 3560.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 10:15:00 | 3547.30 | 3554.29 | 3559.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 11:15:00 | 3552.50 | 3513.48 | 3529.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 12:00:00 | 3552.50 | 3513.48 | 3529.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 12:15:00 | 3579.60 | 3526.71 | 3533.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-29 13:00:00 | 3579.60 | 3526.71 | 3533.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-29 14:15:00 | 3578.60 | 3541.37 | 3539.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 14:15:00 | 3578.60 | 3541.37 | 3539.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 09:15:00 | 3606.00 | 3559.66 | 3548.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 14:15:00 | 3651.80 | 3655.41 | 3623.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 15:00:00 | 3651.80 | 3655.41 | 3623.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 3612.30 | 3728.99 | 3701.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 3612.30 | 3728.99 | 3701.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 3630.00 | 3709.19 | 3694.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-05 09:15:00 | 3675.40 | 3709.19 | 3694.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 10:15:00 | 3645.00 | 3683.65 | 3685.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2025-09-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 10:15:00 | 3645.00 | 3683.65 | 3685.03 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2025-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 15:15:00 | 3693.00 | 3685.82 | 3685.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 3750.50 | 3698.76 | 3691.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 09:15:00 | 3727.80 | 3753.51 | 3730.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 3727.80 | 3753.51 | 3730.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 3727.80 | 3753.51 | 3730.40 | EMA400 retest candle locked (from upside) |

### Cycle 87 — SELL (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 10:15:00 | 3675.30 | 3729.03 | 3730.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-10 11:15:00 | 3646.50 | 3712.53 | 3723.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 10:15:00 | 3688.80 | 3686.71 | 3702.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 11:15:00 | 3700.30 | 3689.43 | 3702.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 3700.30 | 3689.43 | 3702.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:00:00 | 3700.30 | 3689.43 | 3702.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 3710.10 | 3693.56 | 3702.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 12:45:00 | 3713.40 | 3693.56 | 3702.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 3711.00 | 3697.05 | 3703.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:45:00 | 3710.20 | 3697.05 | 3703.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 3685.00 | 3694.81 | 3701.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:15:00 | 3690.00 | 3694.81 | 3701.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 3694.10 | 3694.67 | 3700.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 12:00:00 | 3660.80 | 3678.44 | 3688.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 13:30:00 | 3655.90 | 3671.07 | 3683.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:15:00 | 3650.80 | 3662.97 | 3674.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 13:00:00 | 3661.10 | 3663.88 | 3667.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 3665.20 | 3662.24 | 3665.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 15:00:00 | 3665.20 | 3662.24 | 3665.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 3664.00 | 3662.59 | 3665.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 3745.00 | 3662.59 | 3665.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-18 09:15:00 | 3787.00 | 3687.47 | 3676.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — BUY (started 2025-09-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 09:15:00 | 3787.00 | 3687.47 | 3676.74 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 3677.50 | 3719.23 | 3720.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 10:15:00 | 3666.00 | 3708.59 | 3715.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 10:15:00 | 3671.00 | 3666.15 | 3686.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 11:00:00 | 3671.00 | 3666.15 | 3686.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 3713.20 | 3675.56 | 3688.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 12:00:00 | 3713.20 | 3675.56 | 3688.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 3707.10 | 3681.87 | 3690.27 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-23 14:15:00 | 3729.90 | 3697.78 | 3696.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-23 15:15:00 | 3749.50 | 3708.13 | 3701.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-24 09:15:00 | 3705.10 | 3707.52 | 3701.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 09:15:00 | 3705.10 | 3707.52 | 3701.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 09:15:00 | 3705.10 | 3707.52 | 3701.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 10:15:00 | 3692.00 | 3707.52 | 3701.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 3662.20 | 3698.46 | 3698.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 3662.20 | 3698.46 | 3698.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 3659.40 | 3690.65 | 3694.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 09:15:00 | 3640.00 | 3678.42 | 3686.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 15:15:00 | 3454.90 | 3454.88 | 3499.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-01 09:15:00 | 3514.00 | 3454.88 | 3499.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 3489.00 | 3461.70 | 3498.92 | EMA400 retest candle locked (from downside) |

### Cycle 92 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 3628.60 | 3521.86 | 3517.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 15:15:00 | 3649.00 | 3565.33 | 3538.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 10:15:00 | 3567.90 | 3568.22 | 3545.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 11:00:00 | 3567.90 | 3568.22 | 3545.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 3549.30 | 3566.13 | 3554.38 | EMA400 retest candle locked (from upside) |

### Cycle 93 — SELL (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 14:15:00 | 3537.00 | 3549.18 | 3549.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-06 15:15:00 | 3533.00 | 3545.94 | 3547.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-07 11:15:00 | 3530.30 | 3529.53 | 3538.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-07 12:00:00 | 3530.30 | 3529.53 | 3538.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 3543.10 | 3532.00 | 3538.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:45:00 | 3544.20 | 3532.00 | 3538.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 3522.30 | 3530.06 | 3536.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:45:00 | 3542.00 | 3530.06 | 3536.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — BUY (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-08 09:15:00 | 3672.00 | 3558.47 | 3548.48 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 3581.70 | 3612.08 | 3613.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 14:15:00 | 3576.10 | 3598.69 | 3606.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 09:15:00 | 3598.20 | 3594.96 | 3603.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 3598.20 | 3594.96 | 3603.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 3598.20 | 3594.96 | 3603.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:30:00 | 3600.00 | 3594.96 | 3603.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 3601.60 | 3596.29 | 3602.91 | EMA400 retest candle locked (from downside) |

### Cycle 96 — BUY (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 15:15:00 | 3628.60 | 3606.28 | 3605.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 13:15:00 | 3670.00 | 3634.73 | 3620.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 3745.90 | 3747.65 | 3704.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 10:00:00 | 3745.90 | 3747.65 | 3704.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 3700.00 | 3732.27 | 3707.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 3700.00 | 3732.27 | 3707.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 3691.10 | 3724.04 | 3706.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:00:00 | 3691.10 | 3724.04 | 3706.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 3700.00 | 3714.60 | 3704.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:30:00 | 3691.80 | 3710.18 | 3703.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 3698.60 | 3707.86 | 3703.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:30:00 | 3695.00 | 3707.86 | 3703.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 12:15:00 | 3701.50 | 3704.32 | 3702.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 12:45:00 | 3692.30 | 3704.32 | 3702.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 3691.90 | 3701.84 | 3701.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:00:00 | 3691.90 | 3701.84 | 3701.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2025-10-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 14:15:00 | 3693.60 | 3700.19 | 3700.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 10:15:00 | 3677.40 | 3694.76 | 3697.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 14:15:00 | 3660.50 | 3649.61 | 3665.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 14:15:00 | 3660.50 | 3649.61 | 3665.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 3660.50 | 3649.61 | 3665.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 15:00:00 | 3660.50 | 3649.61 | 3665.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 3663.50 | 3652.39 | 3665.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 13:15:00 | 3655.50 | 3662.33 | 3666.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 15:00:00 | 3659.10 | 3659.36 | 3664.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 3726.80 | 3678.04 | 3672.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 3726.80 | 3678.04 | 3672.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 11:15:00 | 3748.00 | 3715.41 | 3698.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 14:15:00 | 3717.30 | 3722.45 | 3706.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-29 15:00:00 | 3717.30 | 3722.45 | 3706.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 3710.40 | 3718.48 | 3707.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:45:00 | 3700.90 | 3718.48 | 3707.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 3698.00 | 3714.38 | 3706.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:45:00 | 3701.00 | 3714.38 | 3706.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 11:15:00 | 3700.10 | 3711.53 | 3705.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 11:45:00 | 3705.80 | 3711.53 | 3705.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 12:15:00 | 3700.00 | 3709.22 | 3705.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:00:00 | 3700.00 | 3709.22 | 3705.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 3688.20 | 3705.02 | 3703.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 13:45:00 | 3686.40 | 3705.02 | 3703.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 3710.00 | 3707.45 | 3705.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 3718.80 | 3707.45 | 3705.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 3712.00 | 3708.36 | 3705.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 11:00:00 | 3738.60 | 3714.41 | 3708.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 10:15:00 | 3711.00 | 3780.71 | 3785.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 10:15:00 | 3711.00 | 3780.71 | 3785.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 12:15:00 | 3692.10 | 3751.88 | 3770.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 09:15:00 | 3607.50 | 3582.67 | 3625.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-11 10:00:00 | 3607.50 | 3582.67 | 3625.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 3630.00 | 3592.14 | 3625.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 3630.00 | 3592.14 | 3625.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 3640.00 | 3601.71 | 3626.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:00:00 | 3640.00 | 3601.71 | 3626.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 3631.50 | 3607.67 | 3627.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 15:00:00 | 3616.60 | 3613.22 | 3626.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 11:00:00 | 3617.60 | 3598.73 | 3606.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 12:30:00 | 3619.50 | 3606.33 | 3608.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:15:00 | 3596.00 | 3581.12 | 3585.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 12:15:00 | 3605.00 | 3590.52 | 3589.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 12:15:00 | 3605.00 | 3590.52 | 3589.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 14:15:00 | 3629.00 | 3601.27 | 3594.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 3567.60 | 3597.59 | 3594.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 3567.60 | 3597.59 | 3594.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 3567.60 | 3597.59 | 3594.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:45:00 | 3568.00 | 3597.59 | 3594.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 3578.40 | 3593.76 | 3592.94 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2025-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 11:15:00 | 3577.70 | 3590.54 | 3591.55 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2025-11-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 14:15:00 | 3613.80 | 3592.46 | 3591.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 09:15:00 | 3639.00 | 3605.55 | 3598.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-19 13:15:00 | 3627.40 | 3627.98 | 3613.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 13:30:00 | 3630.00 | 3627.98 | 3613.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 3581.50 | 3618.68 | 3610.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 14:30:00 | 3582.00 | 3618.68 | 3610.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 3579.80 | 3610.91 | 3607.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 09:45:00 | 3594.50 | 3607.01 | 3605.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 10:15:00 | 3580.00 | 3601.60 | 3603.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 10:15:00 | 3580.00 | 3601.60 | 3603.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 12:15:00 | 3575.00 | 3593.45 | 3599.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 09:15:00 | 3581.00 | 3579.31 | 3589.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 3581.00 | 3579.31 | 3589.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 3581.00 | 3579.31 | 3589.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 3581.00 | 3579.31 | 3589.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 3612.00 | 3585.85 | 3591.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:30:00 | 3612.70 | 3585.85 | 3591.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 3624.00 | 3593.48 | 3594.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 12:15:00 | 3600.00 | 3593.48 | 3594.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 13:15:00 | 3622.00 | 3598.15 | 3596.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — BUY (started 2025-11-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 13:15:00 | 3622.00 | 3598.15 | 3596.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 09:15:00 | 3674.90 | 3616.19 | 3605.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-25 10:15:00 | 3643.80 | 3651.67 | 3634.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-25 11:00:00 | 3643.80 | 3651.67 | 3634.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 3616.30 | 3644.60 | 3632.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:45:00 | 3619.90 | 3644.60 | 3632.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 3608.50 | 3637.38 | 3630.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 13:30:00 | 3625.20 | 3635.90 | 3630.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 12:15:00 | 3755.80 | 3795.86 | 3795.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 3755.80 | 3795.86 | 3795.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 13:15:00 | 3751.50 | 3786.99 | 3791.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 11:15:00 | 3775.80 | 3765.82 | 3777.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 11:15:00 | 3775.80 | 3765.82 | 3777.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 3775.80 | 3765.82 | 3777.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:00:00 | 3775.80 | 3765.82 | 3777.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 3780.60 | 3768.78 | 3778.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:45:00 | 3784.00 | 3768.78 | 3778.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 3803.30 | 3775.68 | 3780.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 13:30:00 | 3798.00 | 3775.68 | 3780.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 3804.60 | 3781.46 | 3782.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 15:15:00 | 3780.00 | 3781.46 | 3782.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 3591.00 | 3673.92 | 3703.37 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 13:15:00 | 3674.80 | 3655.95 | 3683.26 | SL hit (close>ema200) qty=0.50 sl=3655.95 alert=retest2 |

### Cycle 106 — BUY (started 2025-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 09:15:00 | 3720.00 | 3678.57 | 3674.72 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 14:15:00 | 3662.70 | 3671.82 | 3672.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 15:15:00 | 3651.00 | 3667.65 | 3670.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-15 10:15:00 | 3676.40 | 3669.38 | 3670.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-15 10:15:00 | 3676.40 | 3669.38 | 3670.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 3676.40 | 3669.38 | 3670.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 11:00:00 | 3676.40 | 3669.38 | 3670.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 3658.00 | 3667.10 | 3669.72 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2025-12-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 14:15:00 | 3711.90 | 3678.38 | 3674.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 15:15:00 | 3725.50 | 3687.80 | 3678.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 10:15:00 | 3672.80 | 3688.74 | 3681.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 10:15:00 | 3672.80 | 3688.74 | 3681.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 3672.80 | 3688.74 | 3681.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 11:00:00 | 3672.80 | 3688.74 | 3681.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 3669.70 | 3684.93 | 3680.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 12:15:00 | 3670.00 | 3684.93 | 3680.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 3678.00 | 3678.52 | 3678.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 3681.80 | 3678.52 | 3678.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 3686.50 | 3680.11 | 3678.82 | EMA400 retest candle locked (from upside) |

### Cycle 109 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 3644.90 | 3672.25 | 3675.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 3642.10 | 3666.22 | 3672.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 13:15:00 | 3647.60 | 3629.77 | 3647.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 13:15:00 | 3647.60 | 3629.77 | 3647.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 3647.60 | 3629.77 | 3647.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:30:00 | 3655.40 | 3629.77 | 3647.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 3636.80 | 3631.18 | 3646.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 15:15:00 | 3629.40 | 3631.18 | 3646.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:00:00 | 3628.90 | 3631.12 | 3642.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 12:45:00 | 3627.70 | 3629.90 | 3640.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-23 09:15:00 | 3735.50 | 3633.91 | 3627.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — BUY (started 2025-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 09:15:00 | 3735.50 | 3633.91 | 3627.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-24 10:15:00 | 3750.90 | 3723.35 | 3688.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 12:15:00 | 3725.80 | 3725.86 | 3695.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 13:15:00 | 3731.30 | 3725.86 | 3695.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 11:15:00 | 3703.60 | 3721.80 | 3707.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:00:00 | 3703.60 | 3721.80 | 3707.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 3697.50 | 3716.94 | 3706.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 12:30:00 | 3696.90 | 3716.94 | 3706.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 13:15:00 | 3696.30 | 3712.81 | 3705.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 13:30:00 | 3697.00 | 3712.81 | 3705.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 15:15:00 | 3703.80 | 3707.04 | 3703.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:15:00 | 3683.70 | 3707.04 | 3703.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 3687.00 | 3703.03 | 3702.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:15:00 | 3677.30 | 3703.03 | 3702.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2025-12-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 10:15:00 | 3666.90 | 3695.81 | 3699.11 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 14:15:00 | 3722.50 | 3701.91 | 3700.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 12:15:00 | 3736.50 | 3716.64 | 3710.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-31 15:15:00 | 3715.20 | 3718.24 | 3713.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 15:15:00 | 3715.20 | 3718.24 | 3713.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 3715.20 | 3718.24 | 3713.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 3807.90 | 3718.24 | 3713.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 15:15:00 | 3884.00 | 3899.33 | 3900.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 15:15:00 | 3884.00 | 3899.33 | 3900.21 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 09:15:00 | 3932.70 | 3906.00 | 3903.17 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 3887.70 | 3901.40 | 3901.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 3857.10 | 3889.11 | 3895.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 15:15:00 | 3850.00 | 3848.54 | 3868.47 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-01-12 09:15:00 | 3754.00 | 3848.54 | 3868.47 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 3732.20 | 3704.09 | 3728.63 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 3732.20 | 3704.09 | 3728.63 | SL hit (close>ema400) qty=1.00 sl=3728.63 alert=retest1 |

### Cycle 116 — BUY (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 10:15:00 | 3617.20 | 3580.31 | 3575.28 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 3545.00 | 3571.28 | 3571.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 3499.20 | 3551.38 | 3562.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 3489.20 | 3476.32 | 3506.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 15:15:00 | 3489.20 | 3476.32 | 3506.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 3489.20 | 3476.32 | 3506.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 09:45:00 | 3447.00 | 3465.58 | 3498.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 3424.40 | 3461.65 | 3479.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-30 09:15:00 | 3274.65 | 3342.94 | 3397.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-30 13:15:00 | 3347.30 | 3337.68 | 3376.98 | SL hit (close>ema200) qty=0.50 sl=3337.68 alert=retest2 |

### Cycle 118 — BUY (started 2026-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 11:15:00 | 3450.40 | 3400.57 | 3395.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 13:15:00 | 3480.00 | 3424.38 | 3407.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 3442.00 | 3445.83 | 3423.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-02 09:45:00 | 3450.60 | 3445.83 | 3423.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 3412.80 | 3439.22 | 3422.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:00:00 | 3412.80 | 3439.22 | 3422.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 3410.20 | 3433.42 | 3421.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 12:00:00 | 3410.20 | 3433.42 | 3421.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 12:15:00 | 3443.00 | 3435.33 | 3423.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 13:30:00 | 3444.80 | 3443.85 | 3428.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-09 09:15:00 | 3789.28 | 3761.78 | 3704.27 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 3724.90 | 3754.52 | 3755.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 13:15:00 | 3697.60 | 3743.14 | 3750.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 09:15:00 | 3675.80 | 3660.37 | 3689.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 11:15:00 | 3689.90 | 3666.70 | 3687.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 3689.90 | 3666.70 | 3687.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:00:00 | 3689.90 | 3666.70 | 3687.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 3669.90 | 3667.34 | 3685.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 13:30:00 | 3630.10 | 3655.87 | 3678.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-17 09:15:00 | 3448.59 | 3534.65 | 3588.87 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 12:15:00 | 3509.80 | 3506.45 | 3560.17 | SL hit (close>ema200) qty=0.50 sl=3506.45 alert=retest2 |

### Cycle 120 — BUY (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 09:15:00 | 3536.00 | 3490.69 | 3489.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 13:15:00 | 3550.40 | 3520.47 | 3505.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 15:15:00 | 3584.90 | 3587.45 | 3558.26 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-26 09:15:00 | 3615.00 | 3587.45 | 3558.26 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 3572.40 | 3588.43 | 3566.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 11:30:00 | 3564.50 | 3588.43 | 3566.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 3549.00 | 3580.25 | 3570.99 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 3549.00 | 3580.25 | 3570.99 | SL hit (close<ema400) qty=1.00 sl=3570.99 alert=retest1 |

### Cycle 121 — SELL (started 2026-02-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 11:15:00 | 3525.50 | 3561.72 | 3563.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 12:15:00 | 3521.30 | 3553.64 | 3559.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 11:15:00 | 3290.40 | 3281.71 | 3347.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 12:00:00 | 3290.40 | 3281.71 | 3347.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 3326.50 | 3291.63 | 3335.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 3341.40 | 3291.63 | 3335.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 3339.10 | 3301.13 | 3336.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 3370.00 | 3301.13 | 3336.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 3358.30 | 3312.56 | 3338.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 3360.60 | 3312.56 | 3338.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 3348.50 | 3319.75 | 3339.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 12:15:00 | 3305.90 | 3321.36 | 3338.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:15:00 | 3316.30 | 3325.79 | 3337.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 09:30:00 | 3314.60 | 3232.20 | 3264.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-10 13:15:00 | 3328.00 | 3279.94 | 3279.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2026-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 13:15:00 | 3328.00 | 3279.94 | 3279.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 3332.20 | 3290.39 | 3284.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 3281.60 | 3354.80 | 3332.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 3281.60 | 3354.80 | 3332.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 3281.60 | 3354.80 | 3332.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 10:00:00 | 3281.60 | 3354.80 | 3332.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 3290.50 | 3341.94 | 3328.86 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 12:15:00 | 3279.90 | 3318.64 | 3319.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 13:15:00 | 3234.00 | 3301.72 | 3312.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 09:15:00 | 3081.10 | 3065.99 | 3099.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 3081.10 | 3065.99 | 3099.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 3081.10 | 3065.99 | 3099.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 3027.80 | 3087.52 | 3098.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-19 15:15:00 | 2876.41 | 2955.41 | 3015.22 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-20 09:15:00 | 3015.00 | 2967.33 | 3015.20 | SL hit (close>ema200) qty=0.50 sl=2967.33 alert=retest2 |

### Cycle 124 — BUY (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 15:15:00 | 3086.00 | 3042.29 | 3037.46 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 2933.00 | 3020.43 | 3027.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 10:15:00 | 2902.00 | 2996.75 | 3016.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 2903.10 | 2899.72 | 2937.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 2904.70 | 2899.72 | 2937.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 2909.20 | 2901.62 | 2935.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 14:30:00 | 2891.00 | 2901.11 | 2932.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 15:00:00 | 2899.10 | 2901.11 | 2932.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 11:15:00 | 2959.10 | 2926.38 | 2935.04 | SL hit (close>static) qty=1.00 sl=2949.80 alert=retest2 |

### Cycle 126 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 2972.90 | 2942.25 | 2941.15 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 2866.40 | 2933.71 | 2938.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 12:15:00 | 2863.50 | 2901.54 | 2920.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 2838.10 | 2789.58 | 2831.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 2838.10 | 2789.58 | 2831.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 2838.10 | 2789.58 | 2831.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 2840.00 | 2789.58 | 2831.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 2810.20 | 2793.70 | 2829.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 2752.40 | 2818.14 | 2829.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 14:15:00 | 2847.30 | 2802.34 | 2811.29 | SL hit (close>static) qty=1.00 sl=2839.20 alert=retest2 |

### Cycle 128 — BUY (started 2026-04-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 11:15:00 | 2853.60 | 2823.16 | 2819.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 12:15:00 | 2885.00 | 2835.53 | 2825.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 15:15:00 | 2895.10 | 2895.32 | 2872.08 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-08 09:15:00 | 2986.50 | 2895.32 | 2872.08 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-09 09:15:00 | 3135.83 | 3039.18 | 2973.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 3146.70 | 3170.98 | 3119.86 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 3146.70 | 3170.98 | 3119.86 | SL hit (close<ema200) qty=0.50 sl=3170.98 alert=retest1 |

### Cycle 129 — SELL (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 12:15:00 | 3308.20 | 3322.85 | 3323.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 09:15:00 | 3285.30 | 3309.49 | 3316.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 3273.20 | 3257.83 | 3280.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 3273.20 | 3257.83 | 3280.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 3273.20 | 3257.83 | 3280.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 3292.00 | 3257.83 | 3280.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 3247.40 | 3255.75 | 3277.75 | EMA400 retest candle locked (from downside) |

### Cycle 130 — BUY (started 2026-04-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 14:15:00 | 3310.50 | 3280.41 | 3278.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 09:15:00 | 3357.00 | 3300.78 | 3288.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 10:15:00 | 3296.30 | 3299.89 | 3289.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 10:15:00 | 3296.30 | 3299.89 | 3289.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 10:15:00 | 3296.30 | 3299.89 | 3289.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 11:00:00 | 3296.30 | 3299.89 | 3289.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 11:15:00 | 3293.00 | 3298.51 | 3289.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 12:00:00 | 3293.00 | 3298.51 | 3289.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 12:15:00 | 3290.10 | 3296.83 | 3289.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 13:15:00 | 3280.20 | 3296.83 | 3289.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 13:15:00 | 3280.00 | 3293.46 | 3288.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 13:45:00 | 3272.20 | 3293.46 | 3288.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 3301.40 | 3295.05 | 3289.87 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 3221.00 | 3275.44 | 3282.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 12:15:00 | 3220.40 | 3255.98 | 3271.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 15:15:00 | 3250.00 | 3249.16 | 3263.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 15:15:00 | 3250.00 | 3249.16 | 3263.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 3250.00 | 3249.16 | 3263.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 3307.00 | 3249.16 | 3263.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 3290.40 | 3257.41 | 3266.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:45:00 | 3270.60 | 3258.13 | 3265.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 10:15:00 | 3273.70 | 3239.02 | 3237.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 132 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 3273.70 | 3239.02 | 3237.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 3325.00 | 3273.70 | 3256.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 3215.10 | 3306.83 | 3291.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 3215.10 | 3306.83 | 3291.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 3215.10 | 3306.83 | 3291.67 | EMA400 retest candle locked (from upside) |

### Cycle 133 — SELL (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 11:15:00 | 3188.10 | 3263.61 | 3273.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 13:15:00 | 3158.40 | 3229.67 | 3255.52 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-14 13:15:00 | 3635.00 | 2024-05-23 13:15:00 | 3817.85 | STOP_HIT | 1.00 | 5.03% |
| BUY | retest2 | 2024-05-15 11:00:00 | 3625.00 | 2024-05-23 13:15:00 | 3817.85 | STOP_HIT | 1.00 | 5.32% |
| BUY | retest2 | 2024-05-29 11:00:00 | 3873.90 | 2024-05-30 09:15:00 | 3797.00 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2024-05-29 12:15:00 | 3868.45 | 2024-05-30 09:15:00 | 3797.00 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-06-12 13:15:00 | 4100.35 | 2024-06-19 15:15:00 | 4229.75 | STOP_HIT | 1.00 | 3.16% |
| SELL | retest2 | 2024-07-04 12:00:00 | 4102.40 | 2024-07-05 12:15:00 | 4209.50 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2024-07-04 13:00:00 | 4097.15 | 2024-07-05 12:15:00 | 4209.50 | STOP_HIT | 1.00 | -2.74% |
| SELL | retest2 | 2024-07-04 14:45:00 | 4103.90 | 2024-07-05 12:15:00 | 4209.50 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2024-07-19 13:00:00 | 3937.25 | 2024-07-22 09:15:00 | 3982.90 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-07-19 14:30:00 | 3936.30 | 2024-07-22 09:15:00 | 3982.90 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-07-26 14:15:00 | 4095.95 | 2024-08-01 15:15:00 | 4130.00 | STOP_HIT | 1.00 | 0.83% |
| SELL | retest2 | 2024-08-06 12:15:00 | 3754.05 | 2024-08-13 09:15:00 | 3763.95 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2024-08-20 09:15:00 | 3751.25 | 2024-08-20 10:15:00 | 3712.70 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2024-08-20 12:15:00 | 3730.00 | 2024-08-28 15:15:00 | 3830.85 | STOP_HIT | 1.00 | 2.70% |
| SELL | retest2 | 2024-09-05 11:00:00 | 3768.65 | 2024-09-10 14:15:00 | 3745.00 | STOP_HIT | 1.00 | 0.63% |
| BUY | retest2 | 2024-09-27 09:45:00 | 4394.25 | 2024-09-30 10:15:00 | 4242.90 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2024-10-15 13:00:00 | 3893.80 | 2024-10-22 10:15:00 | 3699.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-16 11:30:00 | 3883.85 | 2024-10-22 14:15:00 | 3689.66 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-15 13:00:00 | 3893.80 | 2024-10-23 12:15:00 | 3713.55 | STOP_HIT | 0.50 | 4.63% |
| SELL | retest2 | 2024-10-16 11:30:00 | 3883.85 | 2024-10-23 12:15:00 | 3713.55 | STOP_HIT | 0.50 | 4.38% |
| BUY | retest2 | 2024-10-31 10:30:00 | 3706.40 | 2024-11-07 10:15:00 | 3625.10 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2024-11-05 12:45:00 | 3681.05 | 2024-11-07 10:15:00 | 3625.10 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-11-05 13:15:00 | 3683.70 | 2024-11-07 10:15:00 | 3625.10 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-11-08 14:15:00 | 3626.75 | 2024-11-12 10:15:00 | 3667.65 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-11-08 15:15:00 | 3625.00 | 2024-11-12 10:15:00 | 3667.65 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-11-11 15:00:00 | 3619.80 | 2024-11-12 10:15:00 | 3667.65 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2024-11-12 12:15:00 | 3628.65 | 2024-11-18 14:15:00 | 3447.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 12:15:00 | 3628.65 | 2024-11-19 09:15:00 | 3504.00 | STOP_HIT | 0.50 | 3.44% |
| SELL | retest2 | 2024-11-12 13:15:00 | 3577.25 | 2024-11-22 13:15:00 | 3523.50 | STOP_HIT | 1.00 | 1.50% |
| BUY | retest2 | 2024-11-28 09:15:00 | 3609.85 | 2024-11-28 10:15:00 | 3570.50 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-11-28 10:15:00 | 3606.85 | 2024-11-28 10:15:00 | 3570.50 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-12-02 09:15:00 | 3518.85 | 2024-12-06 12:15:00 | 3517.20 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2024-12-02 09:45:00 | 3539.05 | 2024-12-06 12:15:00 | 3517.20 | STOP_HIT | 1.00 | 0.62% |
| BUY | retest2 | 2024-12-10 15:15:00 | 3526.50 | 2024-12-11 13:15:00 | 3488.75 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-12-16 10:15:00 | 3426.00 | 2024-12-20 10:15:00 | 3254.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-17 09:30:00 | 3415.45 | 2024-12-20 11:15:00 | 3244.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 10:15:00 | 3426.00 | 2024-12-23 10:15:00 | 3217.45 | STOP_HIT | 0.50 | 6.09% |
| SELL | retest2 | 2024-12-17 09:30:00 | 3415.45 | 2024-12-23 10:15:00 | 3217.45 | STOP_HIT | 0.50 | 5.80% |
| BUY | retest2 | 2024-12-31 10:15:00 | 3311.50 | 2025-01-01 09:15:00 | 3226.50 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-01-01 12:30:00 | 3300.85 | 2025-01-06 15:15:00 | 3317.05 | STOP_HIT | 1.00 | 0.49% |
| BUY | retest2 | 2025-01-01 13:00:00 | 3301.00 | 2025-01-06 15:15:00 | 3317.05 | STOP_HIT | 1.00 | 0.49% |
| BUY | retest2 | 2025-01-01 14:45:00 | 3299.70 | 2025-01-06 15:15:00 | 3317.05 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2025-01-23 10:15:00 | 3623.05 | 2025-01-24 10:15:00 | 3549.05 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-02-06 11:00:00 | 3313.20 | 2025-02-11 09:15:00 | 3147.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 11:45:00 | 3311.75 | 2025-02-11 09:15:00 | 3146.16 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 10:00:00 | 3302.20 | 2025-02-11 09:15:00 | 3137.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 14:15:00 | 3315.10 | 2025-02-11 09:15:00 | 3149.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 3218.40 | 2025-02-12 09:15:00 | 3057.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 11:00:00 | 3313.20 | 2025-02-13 09:15:00 | 2981.88 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-07 11:45:00 | 3311.75 | 2025-02-13 09:15:00 | 2980.58 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-10 10:00:00 | 3302.20 | 2025-02-13 09:15:00 | 2971.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-10 14:15:00 | 3315.10 | 2025-02-13 09:15:00 | 2983.59 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-11 09:15:00 | 3218.40 | 2025-02-17 09:15:00 | 2896.56 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-28 10:45:00 | 2862.20 | 2025-03-03 15:15:00 | 2951.50 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2025-03-25 13:45:00 | 3219.95 | 2025-04-01 10:15:00 | 3183.15 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-03-25 15:15:00 | 3233.20 | 2025-04-01 10:15:00 | 3183.15 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-03-26 09:45:00 | 3222.00 | 2025-04-01 10:15:00 | 3183.15 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-03-26 11:30:00 | 3221.65 | 2025-04-01 10:15:00 | 3183.15 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-03-27 10:30:00 | 3265.80 | 2025-04-01 10:15:00 | 3183.15 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2025-03-27 12:00:00 | 3261.20 | 2025-04-01 10:15:00 | 3183.15 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2025-03-28 09:15:00 | 3288.50 | 2025-04-01 10:15:00 | 3183.15 | STOP_HIT | 1.00 | -3.20% |
| BUY | retest2 | 2025-03-28 14:15:00 | 3265.85 | 2025-04-01 10:15:00 | 3183.15 | STOP_HIT | 1.00 | -2.53% |
| BUY | retest2 | 2025-04-17 12:15:00 | 3257.50 | 2025-04-25 10:15:00 | 3321.90 | STOP_HIT | 1.00 | 1.98% |
| SELL | retest2 | 2025-05-06 15:15:00 | 3180.10 | 2025-05-12 09:15:00 | 3358.80 | STOP_HIT | 1.00 | -5.62% |
| SELL | retest2 | 2025-05-07 09:30:00 | 3173.50 | 2025-05-12 09:15:00 | 3358.80 | STOP_HIT | 1.00 | -5.84% |
| SELL | retest2 | 2025-05-07 10:45:00 | 3181.90 | 2025-05-12 09:15:00 | 3358.80 | STOP_HIT | 1.00 | -5.56% |
| SELL | retest2 | 2025-05-07 12:15:00 | 3178.20 | 2025-05-12 09:15:00 | 3358.80 | STOP_HIT | 1.00 | -5.68% |
| SELL | retest2 | 2025-05-08 12:00:00 | 3137.20 | 2025-05-12 09:15:00 | 3358.80 | STOP_HIT | 1.00 | -7.06% |
| SELL | retest2 | 2025-05-09 09:45:00 | 3131.40 | 2025-05-12 09:15:00 | 3358.80 | STOP_HIT | 1.00 | -7.26% |
| SELL | retest2 | 2025-05-09 10:30:00 | 3145.10 | 2025-05-12 09:15:00 | 3358.80 | STOP_HIT | 1.00 | -6.79% |
| BUY | retest2 | 2025-05-19 11:30:00 | 3541.20 | 2025-05-20 10:15:00 | 3515.10 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-06-10 13:15:00 | 3259.80 | 2025-06-13 09:15:00 | 2933.82 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-11 09:30:00 | 3258.10 | 2025-06-13 09:15:00 | 2932.29 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-11 10:15:00 | 3246.20 | 2025-06-13 09:15:00 | 2921.58 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-23 09:15:00 | 3256.00 | 2025-07-01 12:15:00 | 3327.60 | STOP_HIT | 1.00 | 2.20% |
| BUY | retest2 | 2025-07-21 15:15:00 | 3434.00 | 2025-07-25 10:15:00 | 3425.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-07-22 10:15:00 | 3448.90 | 2025-07-25 10:15:00 | 3425.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-07-22 13:15:00 | 3437.70 | 2025-07-25 10:15:00 | 3425.00 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2025-07-24 12:00:00 | 3450.00 | 2025-07-25 10:15:00 | 3425.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-07-28 14:30:00 | 3386.00 | 2025-07-29 09:15:00 | 3485.00 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-08-08 10:30:00 | 3324.90 | 2025-08-11 15:15:00 | 3369.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-08-11 09:30:00 | 3322.00 | 2025-08-11 15:15:00 | 3369.00 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-08-14 10:15:00 | 3411.50 | 2025-08-22 10:15:00 | 3548.00 | STOP_HIT | 1.00 | 4.00% |
| BUY | retest2 | 2025-08-18 09:15:00 | 3479.40 | 2025-08-22 10:15:00 | 3548.00 | STOP_HIT | 1.00 | 1.97% |
| SELL | retest2 | 2025-08-26 09:15:00 | 3531.70 | 2025-08-29 14:15:00 | 3578.60 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2025-08-26 12:30:00 | 3547.00 | 2025-08-29 14:15:00 | 3578.60 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-08-28 09:15:00 | 3530.40 | 2025-08-29 14:15:00 | 3578.60 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-08-28 10:15:00 | 3547.30 | 2025-08-29 14:15:00 | 3578.60 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-09-05 09:15:00 | 3675.40 | 2025-09-05 10:15:00 | 3645.00 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-09-15 12:00:00 | 3660.80 | 2025-09-18 09:15:00 | 3787.00 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2025-09-15 13:30:00 | 3655.90 | 2025-09-18 09:15:00 | 3787.00 | STOP_HIT | 1.00 | -3.59% |
| SELL | retest2 | 2025-09-16 11:15:00 | 3650.80 | 2025-09-18 09:15:00 | 3787.00 | STOP_HIT | 1.00 | -3.73% |
| SELL | retest2 | 2025-09-17 13:00:00 | 3661.10 | 2025-09-18 09:15:00 | 3787.00 | STOP_HIT | 1.00 | -3.44% |
| SELL | retest2 | 2025-10-27 13:15:00 | 3655.50 | 2025-10-28 10:15:00 | 3726.80 | STOP_HIT | 1.00 | -1.95% |
| SELL | retest2 | 2025-10-27 15:00:00 | 3659.10 | 2025-10-28 10:15:00 | 3726.80 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-10-31 11:00:00 | 3738.60 | 2025-11-06 10:15:00 | 3711.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-11-11 15:00:00 | 3616.60 | 2025-11-17 12:15:00 | 3605.00 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2025-11-13 11:00:00 | 3617.60 | 2025-11-17 12:15:00 | 3605.00 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2025-11-13 12:30:00 | 3619.50 | 2025-11-17 12:15:00 | 3605.00 | STOP_HIT | 1.00 | 0.40% |
| SELL | retest2 | 2025-11-17 11:15:00 | 3596.00 | 2025-11-17 12:15:00 | 3605.00 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-11-20 09:45:00 | 3594.50 | 2025-11-20 10:15:00 | 3580.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2025-11-21 12:15:00 | 3600.00 | 2025-11-21 13:15:00 | 3622.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-11-25 13:30:00 | 3625.20 | 2025-12-02 12:15:00 | 3755.80 | STOP_HIT | 1.00 | 3.60% |
| SELL | retest2 | 2025-12-03 15:15:00 | 3780.00 | 2025-12-09 09:15:00 | 3591.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-03 15:15:00 | 3780.00 | 2025-12-09 13:15:00 | 3674.80 | STOP_HIT | 0.50 | 2.78% |
| SELL | retest2 | 2025-12-18 15:15:00 | 3629.40 | 2025-12-23 09:15:00 | 3735.50 | STOP_HIT | 1.00 | -2.92% |
| SELL | retest2 | 2025-12-19 11:00:00 | 3628.90 | 2025-12-23 09:15:00 | 3735.50 | STOP_HIT | 1.00 | -2.94% |
| SELL | retest2 | 2025-12-19 12:45:00 | 3627.70 | 2025-12-23 09:15:00 | 3735.50 | STOP_HIT | 1.00 | -2.97% |
| BUY | retest2 | 2026-01-01 09:15:00 | 3807.90 | 2026-01-07 15:15:00 | 3884.00 | STOP_HIT | 1.00 | 2.00% |
| SELL | retest1 | 2026-01-12 09:15:00 | 3754.00 | 2026-01-16 09:15:00 | 3732.20 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2026-01-16 12:15:00 | 3714.00 | 2026-01-20 09:15:00 | 3528.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-16 12:15:00 | 3714.00 | 2026-01-21 11:15:00 | 3543.60 | STOP_HIT | 0.50 | 4.59% |
| SELL | retest2 | 2026-01-28 09:45:00 | 3447.00 | 2026-01-30 09:15:00 | 3274.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-28 09:45:00 | 3447.00 | 2026-01-30 13:15:00 | 3347.30 | STOP_HIT | 0.50 | 2.89% |
| SELL | retest2 | 2026-01-29 09:15:00 | 3424.40 | 2026-02-01 11:15:00 | 3450.40 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2026-02-02 13:30:00 | 3444.80 | 2026-02-09 09:15:00 | 3789.28 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-02-13 13:30:00 | 3630.10 | 2026-02-17 09:15:00 | 3448.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-13 13:30:00 | 3630.10 | 2026-02-17 12:15:00 | 3509.80 | STOP_HIT | 0.50 | 3.31% |
| BUY | retest1 | 2026-02-26 09:15:00 | 3615.00 | 2026-02-27 09:15:00 | 3549.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-03-06 12:15:00 | 3305.90 | 2026-03-10 13:15:00 | 3328.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2026-03-06 14:15:00 | 3316.30 | 2026-03-10 13:15:00 | 3328.00 | STOP_HIT | 1.00 | -0.35% |
| SELL | retest2 | 2026-03-10 09:30:00 | 3314.60 | 2026-03-10 13:15:00 | 3328.00 | STOP_HIT | 1.00 | -0.40% |
| SELL | retest2 | 2026-03-19 09:15:00 | 3027.80 | 2026-03-19 15:15:00 | 2876.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-19 09:15:00 | 3027.80 | 2026-03-20 09:15:00 | 3015.00 | STOP_HIT | 0.50 | 0.42% |
| SELL | retest2 | 2026-03-20 11:30:00 | 3040.60 | 2026-03-20 15:15:00 | 3086.00 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2026-03-24 14:30:00 | 2891.00 | 2026-03-25 11:15:00 | 2959.10 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2026-03-24 15:00:00 | 2899.10 | 2026-03-25 11:15:00 | 2959.10 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2026-04-02 09:15:00 | 2752.40 | 2026-04-02 14:15:00 | 2847.30 | STOP_HIT | 1.00 | -3.45% |
| SELL | retest2 | 2026-04-06 09:15:00 | 2797.90 | 2026-04-06 11:15:00 | 2853.60 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2026-04-06 09:45:00 | 2803.00 | 2026-04-06 11:15:00 | 2853.60 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest1 | 2026-04-08 09:15:00 | 2986.50 | 2026-04-09 09:15:00 | 3135.83 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2026-04-08 09:15:00 | 2986.50 | 2026-04-13 09:15:00 | 3146.70 | STOP_HIT | 0.50 | 5.36% |
| BUY | retest2 | 2026-04-13 10:45:00 | 3167.00 | 2026-04-23 12:15:00 | 3308.20 | STOP_HIT | 1.00 | 4.46% |
| BUY | retest2 | 2026-04-13 12:00:00 | 3156.60 | 2026-04-23 12:15:00 | 3308.20 | STOP_HIT | 1.00 | 4.80% |
| BUY | retest2 | 2026-04-15 09:15:00 | 3188.90 | 2026-04-23 12:15:00 | 3308.20 | STOP_HIT | 1.00 | 3.74% |
| SELL | retest2 | 2026-05-04 10:45:00 | 3270.60 | 2026-05-06 10:15:00 | 3273.70 | STOP_HIT | 1.00 | -0.09% |
