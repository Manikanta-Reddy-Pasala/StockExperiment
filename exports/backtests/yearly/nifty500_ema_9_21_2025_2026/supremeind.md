# Supreme Industries Ltd. (SUPREMEIND)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 3654.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 67 |
| ALERT1 | 46 |
| ALERT2 | 46 |
| ALERT2_SKIP | 27 |
| ALERT3 | 119 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 52 |
| PARTIAL | 1 |
| TARGET_HIT | 2 |
| STOP_HIT | 50 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 53 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 42
- **Target hits / Stop hits / Partials:** 2 / 50 / 1
- **Avg / median % per leg:** -0.47% / -1.19%
- **Sum % (uncompounded):** -24.69%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 3 | 15.8% | 2 | 17 | 0 | -0.07% | -1.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 19 | 3 | 15.8% | 2 | 17 | 0 | -0.07% | -1.3% |
| SELL (all) | 34 | 8 | 23.5% | 0 | 33 | 1 | -0.69% | -23.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 34 | 8 | 23.5% | 0 | 33 | 1 | -0.69% | -23.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 53 | 11 | 20.8% | 2 | 50 | 1 | -0.47% | -24.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 4311.40 | 4354.38 | 4359.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 4291.00 | 4341.70 | 4353.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 10:15:00 | 4310.60 | 4296.40 | 4320.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 10:15:00 | 4310.60 | 4296.40 | 4320.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 4310.60 | 4296.40 | 4320.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:45:00 | 4343.80 | 4296.40 | 4320.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 4272.40 | 4291.60 | 4315.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 12:45:00 | 4265.50 | 4287.08 | 4311.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 14:15:00 | 4340.00 | 4302.93 | 4314.93 | SL hit (close>static) qty=1.00 sl=4320.70 alert=retest2 |

### Cycle 2 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 4376.40 | 4326.30 | 4322.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 12:15:00 | 4400.00 | 4341.04 | 4329.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 13:15:00 | 4622.90 | 4643.88 | 4574.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 13:45:00 | 4624.00 | 4643.88 | 4574.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 4490.50 | 4600.37 | 4570.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:00:00 | 4490.50 | 4600.37 | 4570.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 4489.00 | 4578.10 | 4563.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:15:00 | 4473.00 | 4578.10 | 4563.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 4455.00 | 4537.24 | 4546.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 4422.00 | 4487.08 | 4518.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 14:15:00 | 4454.90 | 4439.53 | 4475.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 14:15:00 | 4454.90 | 4439.53 | 4475.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 4454.90 | 4439.53 | 4475.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:45:00 | 4497.70 | 4439.53 | 4475.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 4452.10 | 4442.04 | 4473.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 4332.50 | 4442.04 | 4473.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 11:15:00 | 4536.00 | 4482.49 | 4477.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 4536.00 | 4482.49 | 4477.01 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 09:15:00 | 4449.10 | 4470.43 | 4473.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-25 10:15:00 | 4425.50 | 4461.44 | 4468.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 09:15:00 | 4385.50 | 4379.04 | 4403.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 09:15:00 | 4385.50 | 4379.04 | 4403.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 4385.50 | 4379.04 | 4403.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:30:00 | 4391.80 | 4379.04 | 4403.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 4336.80 | 4353.14 | 4377.27 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 12:15:00 | 4382.80 | 4376.81 | 4376.06 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 4344.00 | 4373.09 | 4375.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 10:15:00 | 4314.60 | 4361.39 | 4369.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 4293.30 | 4287.20 | 4310.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 4293.30 | 4287.20 | 4310.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 4293.30 | 4287.20 | 4310.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 4295.50 | 4287.20 | 4310.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 4274.60 | 4271.56 | 4288.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 4274.60 | 4271.56 | 4288.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 4256.30 | 4268.51 | 4285.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:45:00 | 4246.60 | 4264.99 | 4282.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 14:30:00 | 4245.70 | 4263.38 | 4277.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:45:00 | 4253.00 | 4260.29 | 4273.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 11:15:00 | 4204.70 | 4144.70 | 4141.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 4204.70 | 4144.70 | 4141.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 4212.90 | 4158.34 | 4147.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 4186.00 | 4194.76 | 4171.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 10:00:00 | 4186.00 | 4194.76 | 4171.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 4216.10 | 4242.64 | 4223.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 4216.10 | 4242.64 | 4223.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 4231.00 | 4240.31 | 4224.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 13:45:00 | 4241.10 | 4234.37 | 4225.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 09:15:00 | 4268.50 | 4229.61 | 4224.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 09:15:00 | 4123.40 | 4228.40 | 4236.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 09:15:00 | 4123.40 | 4228.40 | 4236.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 12:15:00 | 4072.30 | 4162.72 | 4201.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 4135.00 | 4112.40 | 4143.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 14:15:00 | 4135.00 | 4112.40 | 4143.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 4135.00 | 4112.40 | 4143.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:45:00 | 4146.20 | 4112.40 | 4143.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 4134.70 | 4116.86 | 4142.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 4083.00 | 4116.86 | 4142.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:45:00 | 4111.10 | 4115.95 | 4139.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 13:15:00 | 4236.10 | 4124.00 | 4134.07 | SL hit (close>static) qty=1.00 sl=4144.50 alert=retest2 |

### Cycle 10 — BUY (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 14:15:00 | 4250.20 | 4149.24 | 4144.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-25 09:15:00 | 4292.80 | 4192.15 | 4165.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 14:15:00 | 4295.70 | 4307.28 | 4266.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-28 15:00:00 | 4295.70 | 4307.28 | 4266.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 4264.00 | 4312.67 | 4297.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:00:00 | 4264.00 | 4312.67 | 4297.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 4278.60 | 4305.85 | 4295.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:30:00 | 4249.60 | 4305.85 | 4295.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 4274.00 | 4297.17 | 4293.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 12:30:00 | 4276.00 | 4297.17 | 4293.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 4276.70 | 4293.08 | 4291.76 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 14:15:00 | 4276.10 | 4289.68 | 4290.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 15:15:00 | 4270.00 | 4285.74 | 4288.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 11:15:00 | 4283.50 | 4277.26 | 4283.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 11:15:00 | 4283.50 | 4277.26 | 4283.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 4283.50 | 4277.26 | 4283.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 12:00:00 | 4283.50 | 4277.26 | 4283.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 4294.30 | 4280.67 | 4284.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:00:00 | 4294.30 | 4280.67 | 4284.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 13:15:00 | 4312.80 | 4287.09 | 4286.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 09:15:00 | 4319.00 | 4299.37 | 4293.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 11:15:00 | 4293.70 | 4300.99 | 4295.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 11:15:00 | 4293.70 | 4300.99 | 4295.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 4293.70 | 4300.99 | 4295.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:45:00 | 4298.20 | 4300.99 | 4295.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 4293.10 | 4299.42 | 4294.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:00:00 | 4293.10 | 4299.42 | 4294.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 4275.50 | 4294.63 | 4293.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:00:00 | 4275.50 | 4294.63 | 4293.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 4247.00 | 4285.11 | 4288.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 4239.00 | 4270.24 | 4281.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 4284.10 | 4266.52 | 4277.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 11:15:00 | 4284.10 | 4266.52 | 4277.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 4284.10 | 4266.52 | 4277.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:00:00 | 4284.10 | 4266.52 | 4277.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 4288.70 | 4270.96 | 4278.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:30:00 | 4287.30 | 4270.96 | 4278.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 4276.50 | 4272.07 | 4278.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 14:00:00 | 4248.30 | 4269.58 | 4275.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 4194.40 | 4270.05 | 4274.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 10:15:00 | 4300.80 | 4191.34 | 4184.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 4300.80 | 4191.34 | 4184.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 4327.50 | 4251.90 | 4216.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 10:15:00 | 4273.00 | 4285.35 | 4247.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 11:00:00 | 4273.00 | 4285.35 | 4247.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 4242.40 | 4275.83 | 4252.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:00:00 | 4242.40 | 4275.83 | 4252.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 4215.00 | 4263.66 | 4248.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 4215.00 | 4263.66 | 4248.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 4289.20 | 4265.49 | 4252.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:45:00 | 4294.70 | 4265.49 | 4252.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 4301.80 | 4304.08 | 4288.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:30:00 | 4299.40 | 4304.08 | 4288.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 4307.00 | 4304.66 | 4290.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 4389.60 | 4304.66 | 4290.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 09:15:00 | 4570.10 | 4600.44 | 4601.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 09:15:00 | 4570.10 | 4600.44 | 4601.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 4501.90 | 4557.08 | 4578.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 4523.90 | 4500.28 | 4527.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 4523.90 | 4500.28 | 4527.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 4523.90 | 4500.28 | 4527.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 4517.20 | 4500.28 | 4527.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 4534.40 | 4507.11 | 4527.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 4534.40 | 4507.11 | 4527.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 4560.00 | 4517.68 | 4530.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:45:00 | 4556.40 | 4517.68 | 4530.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 4573.80 | 4545.65 | 4541.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 4599.00 | 4559.48 | 4548.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 4586.20 | 4610.20 | 4596.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 4586.20 | 4610.20 | 4596.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 4586.20 | 4610.20 | 4596.05 | EMA400 retest candle locked (from upside) |

### Cycle 17 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 4581.40 | 4589.65 | 4589.70 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 09:15:00 | 4599.40 | 4590.55 | 4589.99 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 4555.10 | 4583.36 | 4586.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 09:15:00 | 4530.20 | 4566.28 | 4576.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 4328.00 | 4315.02 | 4373.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 09:45:00 | 4331.20 | 4315.02 | 4373.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 4373.00 | 4334.37 | 4372.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:30:00 | 4362.00 | 4334.37 | 4372.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 4352.60 | 4338.02 | 4371.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 15:15:00 | 4348.90 | 4346.07 | 4369.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 4447.90 | 4366.89 | 4374.79 | SL hit (close>static) qty=1.00 sl=4379.90 alert=retest2 |

### Cycle 20 — BUY (started 2025-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 11:15:00 | 4449.40 | 4390.67 | 4384.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 12:15:00 | 4485.30 | 4409.60 | 4393.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 10:15:00 | 4470.40 | 4471.99 | 4436.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 11:00:00 | 4470.40 | 4471.99 | 4436.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 4510.40 | 4561.51 | 4535.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 4510.40 | 4561.51 | 4535.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 4507.80 | 4550.77 | 4533.12 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 11:15:00 | 4491.80 | 4521.52 | 4523.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 4472.20 | 4511.66 | 4518.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 12:15:00 | 4375.00 | 4346.99 | 4372.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 12:15:00 | 4375.00 | 4346.99 | 4372.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 4375.00 | 4346.99 | 4372.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:00:00 | 4375.00 | 4346.99 | 4372.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 4378.40 | 4353.27 | 4372.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:45:00 | 4347.70 | 4353.12 | 4370.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 10:30:00 | 4340.00 | 4344.12 | 4361.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 10:15:00 | 4220.50 | 4195.12 | 4192.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 10:15:00 | 4220.50 | 4195.12 | 4192.07 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 4162.40 | 4193.57 | 4194.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 4149.30 | 4176.41 | 4185.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 4173.60 | 4172.37 | 4180.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 11:00:00 | 4173.60 | 4172.37 | 4180.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 4182.80 | 4174.46 | 4181.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 4182.80 | 4174.46 | 4181.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 4180.30 | 4175.63 | 4181.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 13:30:00 | 4170.00 | 4176.04 | 4180.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 14:30:00 | 4173.10 | 4176.17 | 4180.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 15:15:00 | 4173.00 | 4176.17 | 4180.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 10:15:00 | 4187.30 | 4178.56 | 4180.41 | SL hit (close>static) qty=1.00 sl=4186.00 alert=retest2 |

### Cycle 24 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 4235.50 | 4191.65 | 4186.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 13:15:00 | 4256.50 | 4204.62 | 4192.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 09:15:00 | 4245.40 | 4247.63 | 4227.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 10:00:00 | 4245.40 | 4247.63 | 4227.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 4220.10 | 4242.12 | 4227.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 4220.10 | 4242.12 | 4227.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 4198.10 | 4233.32 | 4224.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:00:00 | 4198.10 | 4233.32 | 4224.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 4173.40 | 4221.33 | 4219.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:00:00 | 4173.40 | 4221.33 | 4219.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 13:15:00 | 4184.00 | 4213.87 | 4216.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 14:15:00 | 4163.70 | 4203.83 | 4211.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 4210.80 | 4198.68 | 4206.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 10:15:00 | 4210.80 | 4198.68 | 4206.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 4210.80 | 4198.68 | 4206.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 4210.80 | 4198.68 | 4206.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 4193.60 | 4197.67 | 4205.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 12:30:00 | 4189.90 | 4197.35 | 4204.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 14:00:00 | 4189.50 | 4195.78 | 4203.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 4223.70 | 4204.65 | 4205.67 | SL hit (close>static) qty=1.00 sl=4219.90 alert=retest2 |

### Cycle 26 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 4225.70 | 4208.86 | 4207.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 15:15:00 | 4247.90 | 4229.46 | 4219.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 10:15:00 | 4213.00 | 4228.75 | 4220.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 10:15:00 | 4213.00 | 4228.75 | 4220.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 4213.00 | 4228.75 | 4220.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:00:00 | 4213.00 | 4228.75 | 4220.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 4213.10 | 4225.62 | 4220.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:45:00 | 4215.70 | 4225.62 | 4220.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 4195.60 | 4219.62 | 4218.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:45:00 | 4209.90 | 4219.62 | 4218.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 4198.40 | 4215.37 | 4216.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 09:15:00 | 4164.30 | 4192.74 | 4203.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 14:15:00 | 3847.60 | 3842.20 | 3899.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 15:00:00 | 3847.60 | 3842.20 | 3899.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 3796.20 | 3812.31 | 3836.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 14:45:00 | 3831.00 | 3812.31 | 3836.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 3876.90 | 3822.95 | 3834.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:00:00 | 3876.90 | 3822.95 | 3834.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 11:15:00 | 3930.30 | 3844.42 | 3843.46 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 3844.40 | 3871.60 | 3872.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 12:15:00 | 3841.00 | 3865.48 | 3869.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 14:15:00 | 3825.30 | 3817.26 | 3836.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 14:15:00 | 3825.30 | 3817.26 | 3836.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 3825.30 | 3817.26 | 3836.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 3825.30 | 3817.26 | 3836.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 3831.20 | 3820.05 | 3835.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 3832.00 | 3820.05 | 3835.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 3842.00 | 3824.44 | 3836.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:45:00 | 3853.70 | 3824.44 | 3836.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 3860.00 | 3831.55 | 3838.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 3860.00 | 3831.55 | 3838.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 3875.80 | 3844.15 | 3843.19 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 3829.40 | 3844.69 | 3844.84 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 11:15:00 | 3855.80 | 3846.91 | 3845.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 12:15:00 | 3859.00 | 3849.33 | 3847.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 13:15:00 | 3857.80 | 3873.93 | 3864.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 13:15:00 | 3857.80 | 3873.93 | 3864.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 3857.80 | 3873.93 | 3864.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:00:00 | 3857.80 | 3873.93 | 3864.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 3922.00 | 3883.54 | 3869.28 | EMA400 retest candle locked (from upside) |

### Cycle 33 — SELL (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 13:15:00 | 3833.00 | 3860.28 | 3863.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 14:15:00 | 3828.00 | 3853.83 | 3860.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 3442.80 | 3437.31 | 3465.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 3442.80 | 3437.31 | 3465.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 3442.80 | 3437.31 | 3465.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 3390.70 | 3425.36 | 3440.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 13:15:00 | 3370.60 | 3334.70 | 3331.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 3370.60 | 3334.70 | 3331.40 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 12:15:00 | 3332.30 | 3335.22 | 3335.47 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 13:15:00 | 3342.00 | 3336.58 | 3336.06 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 14:15:00 | 3328.30 | 3334.92 | 3335.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 10:15:00 | 3312.70 | 3330.48 | 3333.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 3250.60 | 3244.41 | 3270.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-12 09:30:00 | 3260.70 | 3244.41 | 3270.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 3257.20 | 3230.88 | 3245.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:00:00 | 3257.20 | 3230.88 | 3245.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 3256.20 | 3235.95 | 3246.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 13:15:00 | 3266.90 | 3235.95 | 3246.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 15:15:00 | 3284.00 | 3256.35 | 3254.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 09:15:00 | 3369.60 | 3279.00 | 3264.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 09:15:00 | 3362.90 | 3370.00 | 3329.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-17 10:00:00 | 3362.90 | 3370.00 | 3329.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 3329.10 | 3357.50 | 3330.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:30:00 | 3334.30 | 3357.50 | 3330.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 3341.70 | 3354.34 | 3331.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 13:15:00 | 3358.70 | 3354.34 | 3331.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 14:45:00 | 3349.70 | 3349.48 | 3332.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 3348.80 | 3346.58 | 3333.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 12:00:00 | 3349.80 | 3342.83 | 3334.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 3346.00 | 3345.19 | 3338.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 3378.00 | 3345.19 | 3338.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 3343.50 | 3344.85 | 3338.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 3393.30 | 3361.86 | 3350.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 15:15:00 | 3336.00 | 3354.46 | 3353.62 | SL hit (close<static) qty=1.00 sl=3336.10 alert=retest2 |

### Cycle 39 — SELL (started 2025-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 09:15:00 | 3346.30 | 3352.83 | 3352.96 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2025-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 13:15:00 | 3370.70 | 3353.57 | 3352.79 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 09:15:00 | 3350.40 | 3352.08 | 3352.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 13:15:00 | 3329.80 | 3344.97 | 3348.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 3300.00 | 3282.78 | 3291.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 15:15:00 | 3300.00 | 3282.78 | 3291.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 3300.00 | 3282.78 | 3291.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 3364.00 | 3282.78 | 3291.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 3341.50 | 3294.52 | 3295.65 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 3339.40 | 3303.50 | 3299.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 13:15:00 | 3358.00 | 3326.26 | 3311.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 11:15:00 | 3614.70 | 3622.35 | 3588.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 11:45:00 | 3613.20 | 3622.35 | 3588.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 3604.00 | 3615.32 | 3591.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:30:00 | 3613.10 | 3617.76 | 3594.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:30:00 | 3614.60 | 3617.39 | 3598.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 11:15:00 | 3585.80 | 3607.90 | 3597.23 | SL hit (close<static) qty=1.00 sl=3586.00 alert=retest2 |

### Cycle 43 — SELL (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 14:15:00 | 3558.40 | 3586.32 | 3589.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 11:15:00 | 3528.60 | 3559.52 | 3574.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 3473.80 | 3472.52 | 3497.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 15:00:00 | 3473.80 | 3472.52 | 3497.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 3504.70 | 3479.35 | 3496.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:00:00 | 3504.70 | 3479.35 | 3496.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 3492.20 | 3481.92 | 3495.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:00:00 | 3481.70 | 3481.88 | 3494.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:00:00 | 3486.10 | 3482.72 | 3493.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:15:00 | 3484.70 | 3484.60 | 3493.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:00:00 | 3480.90 | 3483.86 | 3492.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 3499.90 | 3487.07 | 3493.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:15:00 | 3514.20 | 3487.07 | 3493.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 3543.00 | 3498.25 | 3497.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 3543.00 | 3498.25 | 3497.65 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 3486.00 | 3499.34 | 3500.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 12:15:00 | 3443.80 | 3483.80 | 3493.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 3374.00 | 3363.89 | 3408.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 13:00:00 | 3374.00 | 3363.89 | 3408.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 3447.80 | 3370.79 | 3396.01 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 12:15:00 | 3461.20 | 3413.75 | 3411.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 13:15:00 | 3492.20 | 3429.44 | 3418.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-27 09:15:00 | 3441.90 | 3475.27 | 3459.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 3441.90 | 3475.27 | 3459.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 3441.90 | 3475.27 | 3459.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:00:00 | 3441.90 | 3475.27 | 3459.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 3463.60 | 3472.94 | 3459.42 | EMA400 retest candle locked (from upside) |

### Cycle 47 — SELL (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 14:15:00 | 3441.40 | 3451.06 | 3451.97 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 3461.00 | 3453.05 | 3452.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 09:15:00 | 3526.00 | 3467.64 | 3459.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 3489.40 | 3496.90 | 3482.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 3489.40 | 3496.90 | 3482.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 3489.40 | 3496.90 | 3482.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:30:00 | 3488.20 | 3496.90 | 3482.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 3484.20 | 3494.36 | 3482.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 3476.90 | 3494.36 | 3482.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 3495.50 | 3494.59 | 3483.42 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 13:15:00 | 3474.50 | 3483.77 | 3484.51 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 14:15:00 | 3509.00 | 3488.82 | 3486.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 3530.20 | 3498.21 | 3491.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 13:15:00 | 3503.30 | 3508.35 | 3499.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 13:15:00 | 3503.30 | 3508.35 | 3499.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 3503.30 | 3508.35 | 3499.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 3500.30 | 3508.35 | 3499.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 3492.40 | 3505.16 | 3498.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 3492.40 | 3505.16 | 3498.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 3491.50 | 3502.43 | 3498.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 3478.80 | 3502.43 | 3498.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 3505.40 | 3502.54 | 3499.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:15:00 | 3493.90 | 3502.54 | 3499.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 3523.50 | 3506.73 | 3501.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 12:15:00 | 3528.30 | 3506.73 | 3501.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-16 14:15:00 | 3881.13 | 3833.73 | 3811.55 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 51 — SELL (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 14:15:00 | 3975.50 | 3996.75 | 3997.32 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 10:15:00 | 4037.00 | 4002.79 | 3999.65 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 3961.50 | 3995.29 | 3996.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 13:15:00 | 3941.00 | 3984.43 | 3991.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 12:15:00 | 3935.10 | 3905.66 | 3929.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 12:15:00 | 3935.10 | 3905.66 | 3929.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 3935.10 | 3905.66 | 3929.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:00:00 | 3935.10 | 3905.66 | 3929.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 3922.10 | 3908.95 | 3928.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:30:00 | 3926.70 | 3908.95 | 3928.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 3945.50 | 3916.26 | 3930.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 3945.50 | 3916.26 | 3930.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 3940.00 | 3921.01 | 3931.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 4025.00 | 3921.01 | 3931.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 4029.50 | 3942.71 | 3940.26 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 3835.20 | 3935.87 | 3942.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 11:15:00 | 3787.50 | 3888.15 | 3918.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 3907.70 | 3855.31 | 3885.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 3907.70 | 3855.31 | 3885.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 3907.70 | 3855.31 | 3885.89 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 3947.90 | 3907.90 | 3903.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 3962.10 | 3918.74 | 3908.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 3945.10 | 3983.81 | 3954.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 3945.10 | 3983.81 | 3954.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 3945.10 | 3983.81 | 3954.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 3935.10 | 3983.81 | 3954.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 4001.20 | 3987.29 | 3959.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:30:00 | 4028.10 | 3995.89 | 3965.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 10:15:00 | 3920.40 | 3986.18 | 3977.42 | SL hit (close<static) qty=1.00 sl=3934.50 alert=retest2 |

### Cycle 57 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 3894.70 | 3957.89 | 3965.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 15:15:00 | 3877.30 | 3924.34 | 3946.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 3929.60 | 3911.54 | 3928.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 3929.60 | 3911.54 | 3928.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 3929.60 | 3911.54 | 3928.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 15:00:00 | 3929.60 | 3911.54 | 3928.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 3890.20 | 3907.27 | 3925.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 3914.00 | 3907.27 | 3925.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 3921.40 | 3910.10 | 3924.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:45:00 | 3922.00 | 3910.10 | 3924.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 3907.70 | 3909.62 | 3923.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 3877.00 | 3909.62 | 3923.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 15:15:00 | 3960.10 | 3917.34 | 3920.60 | SL hit (close>static) qty=1.00 sl=3941.80 alert=retest2 |

### Cycle 58 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 4003.20 | 3934.51 | 3928.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 4006.30 | 3971.12 | 3949.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 3914.60 | 3967.43 | 3953.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 3914.60 | 3967.43 | 3953.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 3914.60 | 3967.43 | 3953.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 3925.40 | 3967.43 | 3953.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 3896.80 | 3953.30 | 3948.75 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 3903.20 | 3943.28 | 3944.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 3867.30 | 3917.62 | 3932.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 3720.00 | 3718.72 | 3770.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:45:00 | 3724.90 | 3718.72 | 3770.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 3758.80 | 3726.74 | 3769.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 3760.10 | 3726.74 | 3769.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 3768.50 | 3743.45 | 3767.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 3901.40 | 3743.45 | 3767.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 3897.40 | 3774.24 | 3779.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 3897.40 | 3774.24 | 3779.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 3900.00 | 3799.39 | 3790.04 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 3797.90 | 3805.28 | 3805.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 3745.90 | 3793.40 | 3800.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 15:15:00 | 3740.00 | 3735.70 | 3759.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 15:15:00 | 3740.00 | 3735.70 | 3759.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 3740.00 | 3735.70 | 3759.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 3837.50 | 3735.70 | 3759.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 3792.70 | 3747.10 | 3762.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 3756.10 | 3747.10 | 3762.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 3568.29 | 3650.12 | 3704.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 3612.30 | 3599.82 | 3657.96 | SL hit (close>ema200) qty=0.50 sl=3599.82 alert=retest2 |

### Cycle 62 — BUY (started 2026-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 09:15:00 | 3694.10 | 3665.57 | 3662.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 10:15:00 | 3711.10 | 3674.68 | 3666.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 13:15:00 | 3764.80 | 3778.66 | 3742.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-08 14:00:00 | 3764.80 | 3778.66 | 3742.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 14:15:00 | 3794.60 | 3781.84 | 3746.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 12:15:00 | 3800.20 | 3783.02 | 3758.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:45:00 | 3802.20 | 3779.86 | 3765.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 15:15:00 | 3801.00 | 3782.88 | 3772.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 09:45:00 | 3824.10 | 3793.04 | 3779.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 12:15:00 | 3796.30 | 3797.32 | 3784.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 12:30:00 | 3802.00 | 3797.32 | 3784.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 3787.30 | 3796.22 | 3787.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 3833.40 | 3796.22 | 3787.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 3723.30 | 3801.85 | 3800.76 | SL hit (close<static) qty=1.00 sl=3745.60 alert=retest2 |

### Cycle 63 — SELL (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 10:15:00 | 3743.00 | 3790.08 | 3795.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 11:15:00 | 3686.10 | 3769.28 | 3785.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 09:15:00 | 3718.10 | 3703.92 | 3740.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-17 09:45:00 | 3705.50 | 3703.92 | 3740.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 15:15:00 | 3731.00 | 3707.03 | 3725.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:15:00 | 3729.10 | 3707.03 | 3725.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 3706.00 | 3706.82 | 3723.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 13:15:00 | 3672.70 | 3705.27 | 3715.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:30:00 | 3671.00 | 3691.99 | 3698.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 11:30:00 | 3655.00 | 3685.25 | 3695.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 15:00:00 | 3673.60 | 3685.87 | 3693.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 3696.90 | 3684.87 | 3691.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 10:00:00 | 3696.90 | 3684.87 | 3691.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 3631.40 | 3674.17 | 3685.84 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-27 10:15:00 | 3720.00 | 3683.98 | 3682.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 3720.00 | 3683.98 | 3682.65 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 09:15:00 | 3657.50 | 3684.42 | 3685.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 12:15:00 | 3648.80 | 3672.49 | 3679.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 3674.60 | 3658.64 | 3669.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 3674.60 | 3658.64 | 3669.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 3674.60 | 3658.64 | 3669.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 3607.10 | 3667.41 | 3670.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 15:00:00 | 3618.40 | 3618.50 | 3638.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 3635.10 | 3638.29 | 3642.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 11:45:00 | 3631.60 | 3637.75 | 3638.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 3642.30 | 3638.66 | 3638.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:00:00 | 3642.30 | 3638.66 | 3638.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 3633.60 | 3637.65 | 3638.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:30:00 | 3648.80 | 3637.65 | 3638.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 3637.10 | 3637.54 | 3638.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 15:15:00 | 3625.20 | 3637.54 | 3638.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 3650.00 | 3638.06 | 3638.15 | SL hit (close>static) qty=1.00 sl=3644.30 alert=retest2 |

### Cycle 66 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 3661.30 | 3642.71 | 3640.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 11:15:00 | 3689.60 | 3652.08 | 3644.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 10:15:00 | 3687.10 | 3688.47 | 3669.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 11:00:00 | 3687.10 | 3688.47 | 3669.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 3676.50 | 3686.08 | 3669.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:45:00 | 3673.00 | 3686.08 | 3669.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 3665.10 | 3681.88 | 3669.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 13:00:00 | 3665.10 | 3681.88 | 3669.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 3665.00 | 3678.51 | 3669.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 13:45:00 | 3658.20 | 3678.51 | 3669.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 3663.10 | 3675.43 | 3668.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 14:30:00 | 3659.90 | 3675.43 | 3668.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 3669.60 | 3674.26 | 3668.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 3688.20 | 3674.26 | 3668.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 12:15:00 | 3650.20 | 3666.88 | 3666.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 3650.20 | 3666.88 | 3666.89 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 09:15:00 | 3642.00 | 2025-05-26 13:15:00 | 4006.20 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-13 12:45:00 | 4265.50 | 2025-06-13 14:15:00 | 4340.00 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-06-23 09:15:00 | 4332.50 | 2025-06-24 11:15:00 | 4536.00 | STOP_HIT | 1.00 | -4.70% |
| SELL | retest2 | 2025-07-07 11:45:00 | 4246.60 | 2025-07-15 11:15:00 | 4204.70 | STOP_HIT | 1.00 | 0.99% |
| SELL | retest2 | 2025-07-07 14:30:00 | 4245.70 | 2025-07-15 11:15:00 | 4204.70 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2025-07-08 09:45:00 | 4253.00 | 2025-07-15 11:15:00 | 4204.70 | STOP_HIT | 1.00 | 1.14% |
| BUY | retest2 | 2025-07-18 13:45:00 | 4241.10 | 2025-07-22 09:15:00 | 4123.40 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-07-21 09:15:00 | 4268.50 | 2025-07-22 09:15:00 | 4123.40 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2025-07-24 09:15:00 | 4083.00 | 2025-07-24 13:15:00 | 4236.10 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2025-07-24 09:45:00 | 4111.10 | 2025-07-24 13:15:00 | 4236.10 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2025-08-05 14:00:00 | 4248.30 | 2025-08-11 10:15:00 | 4300.80 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-08-06 09:15:00 | 4194.40 | 2025-08-11 10:15:00 | 4300.80 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-08-18 09:15:00 | 4389.60 | 2025-08-28 09:15:00 | 4570.10 | STOP_HIT | 1.00 | 4.11% |
| SELL | retest2 | 2025-09-11 15:15:00 | 4348.90 | 2025-09-12 09:15:00 | 4447.90 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-09-24 14:45:00 | 4347.70 | 2025-10-07 10:15:00 | 4220.50 | STOP_HIT | 1.00 | 2.93% |
| SELL | retest2 | 2025-09-25 10:30:00 | 4340.00 | 2025-10-07 10:15:00 | 4220.50 | STOP_HIT | 1.00 | 2.75% |
| SELL | retest2 | 2025-10-09 13:30:00 | 4170.00 | 2025-10-10 10:15:00 | 4187.30 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-10-09 14:30:00 | 4173.10 | 2025-10-10 10:15:00 | 4187.30 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-10-09 15:15:00 | 4173.00 | 2025-10-10 10:15:00 | 4187.30 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-10-15 12:30:00 | 4189.90 | 2025-10-16 09:15:00 | 4223.70 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-10-15 14:00:00 | 4189.50 | 2025-10-16 09:15:00 | 4223.70 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-11-28 09:15:00 | 3390.70 | 2025-12-05 13:15:00 | 3370.60 | STOP_HIT | 1.00 | 0.59% |
| BUY | retest2 | 2025-12-17 13:15:00 | 3358.70 | 2025-12-22 15:15:00 | 3336.00 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-12-17 14:45:00 | 3349.70 | 2025-12-23 09:15:00 | 3346.30 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2025-12-18 09:15:00 | 3348.80 | 2025-12-23 09:15:00 | 3346.30 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-12-18 12:00:00 | 3349.80 | 2025-12-23 09:15:00 | 3346.30 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2025-12-22 09:15:00 | 3393.30 | 2025-12-23 09:15:00 | 3346.30 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2026-01-07 14:30:00 | 3613.10 | 2026-01-08 11:15:00 | 3585.80 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2026-01-08 09:30:00 | 3614.60 | 2026-01-08 11:15:00 | 3585.80 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-01-14 12:00:00 | 3481.70 | 2026-01-16 09:15:00 | 3543.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2026-01-14 13:00:00 | 3486.10 | 2026-01-16 09:15:00 | 3543.00 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2026-01-14 14:15:00 | 3484.70 | 2026-01-16 09:15:00 | 3543.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2026-01-14 15:00:00 | 3480.90 | 2026-01-16 09:15:00 | 3543.00 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2026-02-02 12:15:00 | 3528.30 | 2026-02-16 14:15:00 | 3881.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-12 11:30:00 | 4028.10 | 2026-03-13 10:15:00 | 3920.40 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2026-03-17 11:15:00 | 3877.00 | 2026-03-17 15:15:00 | 3960.10 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2026-04-01 10:15:00 | 3756.10 | 2026-04-02 09:15:00 | 3568.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:15:00 | 3756.10 | 2026-04-02 13:15:00 | 3612.30 | STOP_HIT | 0.50 | 3.83% |
| BUY | retest2 | 2026-04-09 12:15:00 | 3800.20 | 2026-04-16 09:15:00 | 3723.30 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2026-04-10 09:45:00 | 3802.20 | 2026-04-16 09:15:00 | 3723.30 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2026-04-10 15:15:00 | 3801.00 | 2026-04-16 09:15:00 | 3723.30 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2026-04-13 09:45:00 | 3824.10 | 2026-04-16 09:15:00 | 3723.30 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2026-04-15 09:15:00 | 3833.40 | 2026-04-16 09:15:00 | 3723.30 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2026-04-21 13:15:00 | 3672.70 | 2026-04-27 10:15:00 | 3720.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2026-04-23 10:30:00 | 3671.00 | 2026-04-27 10:15:00 | 3720.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2026-04-23 11:30:00 | 3655.00 | 2026-04-27 10:15:00 | 3720.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-04-23 15:00:00 | 3673.60 | 2026-04-27 10:15:00 | 3720.00 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-04-30 09:15:00 | 3607.10 | 2026-05-06 09:15:00 | 3650.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2026-04-30 15:00:00 | 3618.40 | 2026-05-06 10:15:00 | 3661.30 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2026-05-04 12:00:00 | 3635.10 | 2026-05-06 10:15:00 | 3661.30 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2026-05-05 11:45:00 | 3631.60 | 2026-05-06 10:15:00 | 3661.30 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-05-05 15:15:00 | 3625.20 | 2026-05-06 10:15:00 | 3661.30 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2026-05-08 09:15:00 | 3688.20 | 2026-05-08 12:15:00 | 3650.20 | STOP_HIT | 1.00 | -1.03% |
