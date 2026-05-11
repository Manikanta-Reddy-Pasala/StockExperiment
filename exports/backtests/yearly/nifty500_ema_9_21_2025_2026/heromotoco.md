# Hero MotoCorp Ltd. (HEROMOTOCO)

## Backtest Summary

- **Window:** 2025-06-16 09:15:00 → 2026-05-08 15:15:00 (1549 bars)
- **Last close:** 5325.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 69 |
| ALERT1 | 50 |
| ALERT2 | 49 |
| ALERT2_SKIP | 24 |
| ALERT3 | 108 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 54 |
| PARTIAL | 2 |
| TARGET_HIT | 5 |
| STOP_HIT | 48 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 55 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 17 / 38
- **Target hits / Stop hits / Partials:** 5 / 48 / 2
- **Avg / median % per leg:** 0.49% / -0.76%
- **Sum % (uncompounded):** 27.11%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 20 | 7 | 35.0% | 5 | 15 | 0 | 2.18% | 43.5% |
| BUY @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.38% | -2.8% |
| BUY @ 3rd Alert (retest2) | 18 | 7 | 38.9% | 5 | 13 | 0 | 2.57% | 46.3% |
| SELL (all) | 35 | 10 | 28.6% | 0 | 33 | 2 | -0.47% | -16.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 35 | 10 | 28.6% | 0 | 33 | 2 | -0.47% | -16.4% |
| retest1 (combined) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.38% | -2.8% |
| retest2 (combined) | 53 | 17 | 32.1% | 5 | 46 | 2 | 0.56% | 29.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-06-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-19 11:15:00 | 4380.80 | 4378.85 | 4378.70 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 4370.10 | 4377.10 | 4377.92 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-06-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-19 14:15:00 | 4384.40 | 4378.94 | 4378.64 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 15:15:00 | 4374.40 | 4378.03 | 4378.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 09:15:00 | 4282.40 | 4358.91 | 4369.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 09:15:00 | 4299.00 | 4281.02 | 4306.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-24 10:00:00 | 4299.00 | 4281.02 | 4306.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 4311.30 | 4287.08 | 4307.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:00:00 | 4311.30 | 4287.08 | 4307.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 4319.00 | 4293.46 | 4308.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-26 09:30:00 | 4256.00 | 4296.16 | 4302.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-27 10:15:00 | 4320.50 | 4295.27 | 4295.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 10:15:00 | 4320.50 | 4295.27 | 4295.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 11:15:00 | 4344.00 | 4305.01 | 4299.54 | Break + close above crossover candle high |

### Cycle 6 — SELL (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 09:15:00 | 4231.90 | 4299.41 | 4300.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 10:15:00 | 4202.30 | 4246.23 | 4268.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 09:15:00 | 4240.00 | 4230.11 | 4248.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 09:15:00 | 4240.00 | 4230.11 | 4248.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 4240.00 | 4230.11 | 4248.11 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 4286.00 | 4251.96 | 4251.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 11:15:00 | 4307.20 | 4268.25 | 4259.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 10:15:00 | 4328.10 | 4334.92 | 4315.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 12:15:00 | 4304.00 | 4327.31 | 4315.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 4304.00 | 4327.31 | 4315.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 13:00:00 | 4304.00 | 4327.31 | 4315.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 4302.40 | 4322.33 | 4314.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 13:45:00 | 4303.00 | 4322.33 | 4314.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 4274.90 | 4307.69 | 4309.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 4248.00 | 4295.75 | 4303.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 11:15:00 | 4297.00 | 4296.00 | 4302.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 12:00:00 | 4297.00 | 4296.00 | 4302.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 4305.40 | 4297.88 | 4303.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:45:00 | 4302.00 | 4297.88 | 4303.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 4283.50 | 4295.00 | 4301.37 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 4320.00 | 4305.17 | 4304.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 11:15:00 | 4327.50 | 4309.64 | 4306.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 11:15:00 | 4324.60 | 4326.30 | 4318.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 11:15:00 | 4324.60 | 4326.30 | 4318.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 4324.60 | 4326.30 | 4318.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:45:00 | 4323.30 | 4326.30 | 4318.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 4329.90 | 4327.02 | 4319.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:00:00 | 4329.90 | 4327.02 | 4319.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 4322.60 | 4326.61 | 4320.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 15:00:00 | 4322.60 | 4326.61 | 4320.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 4315.80 | 4324.45 | 4320.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 4334.80 | 4324.45 | 4320.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 4310.10 | 4321.58 | 4319.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 4310.10 | 4321.58 | 4319.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 4275.30 | 4312.32 | 4315.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 12:15:00 | 4232.30 | 4289.99 | 4304.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 14:15:00 | 4253.90 | 4242.43 | 4262.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 15:00:00 | 4253.90 | 4242.43 | 4262.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 4326.00 | 4259.52 | 4266.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 4300.40 | 4259.52 | 4266.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 4399.70 | 4287.56 | 4278.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 4402.20 | 4310.49 | 4290.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 4393.20 | 4393.57 | 4345.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 09:45:00 | 4400.30 | 4393.57 | 4345.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 4413.10 | 4429.90 | 4409.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 4407.30 | 4429.90 | 4409.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 4403.50 | 4424.62 | 4409.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:30:00 | 4400.90 | 4424.62 | 4409.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 4406.10 | 4420.91 | 4409.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:45:00 | 4406.20 | 4420.91 | 4409.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 4406.90 | 4418.11 | 4408.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 11:45:00 | 4418.00 | 4408.23 | 4406.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 12:30:00 | 4418.50 | 4409.70 | 4407.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 09:45:00 | 4414.10 | 4410.92 | 4408.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 10:15:00 | 4383.50 | 4405.44 | 4406.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 10:15:00 | 4383.50 | 4405.44 | 4406.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 11:15:00 | 4360.60 | 4396.47 | 4402.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 4374.20 | 4366.18 | 4382.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 09:45:00 | 4373.20 | 4366.18 | 4382.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 4366.50 | 4358.18 | 4369.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 4296.10 | 4358.18 | 4369.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 10:15:00 | 4321.50 | 4290.25 | 4286.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 10:15:00 | 4321.50 | 4290.25 | 4286.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 12:15:00 | 4330.00 | 4303.41 | 4293.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 10:15:00 | 4257.80 | 4304.42 | 4299.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 10:15:00 | 4257.80 | 4304.42 | 4299.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 4257.80 | 4304.42 | 4299.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:45:00 | 4265.40 | 4304.42 | 4299.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 11:15:00 | 4227.50 | 4289.04 | 4292.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 4218.00 | 4253.70 | 4271.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 11:15:00 | 4264.80 | 4254.77 | 4269.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 11:15:00 | 4264.80 | 4254.77 | 4269.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 4264.80 | 4254.77 | 4269.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:45:00 | 4270.00 | 4254.77 | 4269.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 4282.40 | 4260.29 | 4270.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 12:45:00 | 4293.50 | 4260.29 | 4270.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 4293.60 | 4266.96 | 4272.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:30:00 | 4288.00 | 4266.96 | 4272.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 4259.10 | 4261.48 | 4268.20 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 11:15:00 | 4297.00 | 4273.18 | 4272.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 09:15:00 | 4426.40 | 4320.17 | 4296.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 4491.40 | 4522.10 | 4466.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-06 10:00:00 | 4491.40 | 4522.10 | 4466.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 4475.00 | 4496.41 | 4473.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:30:00 | 4470.20 | 4496.41 | 4473.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 4470.10 | 4491.15 | 4472.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 4556.60 | 4491.15 | 4472.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 4533.50 | 4499.62 | 4478.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 12:45:00 | 4607.60 | 4531.81 | 4499.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 4604.00 | 4573.12 | 4571.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-18 09:15:00 | 5068.36 | 4790.95 | 4735.72 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 5015.30 | 5058.65 | 5060.74 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 14:15:00 | 5058.10 | 5048.43 | 5047.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-26 09:15:00 | 5115.00 | 5064.72 | 5055.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 14:15:00 | 5060.70 | 5080.72 | 5069.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 14:15:00 | 5060.70 | 5080.72 | 5069.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 5060.70 | 5080.72 | 5069.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 5060.70 | 5080.72 | 5069.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 5074.90 | 5079.56 | 5069.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 5151.00 | 5079.56 | 5069.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 09:45:00 | 5102.90 | 5113.66 | 5099.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 5348.50 | 5402.02 | 5407.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 11:15:00 | 5348.50 | 5402.02 | 5407.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 09:15:00 | 5306.00 | 5355.80 | 5380.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 5347.00 | 5324.19 | 5347.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 5347.00 | 5324.19 | 5347.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 5347.00 | 5324.19 | 5347.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:30:00 | 5374.50 | 5324.19 | 5347.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 5352.00 | 5329.76 | 5348.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:00:00 | 5352.00 | 5329.76 | 5348.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 5315.00 | 5326.80 | 5345.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 13:30:00 | 5293.00 | 5318.04 | 5337.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 11:30:00 | 5291.00 | 5306.10 | 5324.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 09:30:00 | 5293.00 | 5303.33 | 5314.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 13:15:00 | 5333.00 | 5322.68 | 5321.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 13:15:00 | 5333.00 | 5322.68 | 5321.63 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 14:15:00 | 5313.00 | 5320.74 | 5320.84 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 15:15:00 | 5322.50 | 5321.09 | 5321.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 5353.00 | 5327.48 | 5323.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 14:15:00 | 5352.50 | 5352.57 | 5340.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 14:30:00 | 5354.00 | 5352.57 | 5340.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 5351.00 | 5358.47 | 5348.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:45:00 | 5352.00 | 5358.47 | 5348.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 5357.50 | 5358.27 | 5349.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 15:00:00 | 5375.50 | 5361.72 | 5351.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 5374.00 | 5404.25 | 5407.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 5374.00 | 5404.25 | 5407.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 15:15:00 | 5362.50 | 5395.90 | 5402.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 10:15:00 | 5308.50 | 5307.58 | 5340.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 11:15:00 | 5330.00 | 5307.58 | 5340.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 5338.00 | 5309.24 | 5333.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:00:00 | 5338.00 | 5309.24 | 5333.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 5345.00 | 5316.39 | 5334.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:45:00 | 5369.00 | 5316.39 | 5334.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 5356.50 | 5324.41 | 5336.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 5358.00 | 5324.41 | 5336.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-26 09:15:00 | 5422.50 | 5344.03 | 5344.01 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 10:15:00 | 5327.00 | 5347.51 | 5349.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 11:15:00 | 5282.00 | 5334.41 | 5343.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 5384.50 | 5341.80 | 5345.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 13:15:00 | 5384.50 | 5341.80 | 5345.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 5384.50 | 5341.80 | 5345.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 5384.50 | 5341.80 | 5345.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 5283.00 | 5330.04 | 5339.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:45:00 | 5424.00 | 5330.04 | 5339.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 5345.50 | 5333.13 | 5340.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 5370.00 | 5333.13 | 5340.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 5375.00 | 5341.51 | 5343.35 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 10:15:00 | 5400.00 | 5353.20 | 5348.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 11:15:00 | 5423.00 | 5367.16 | 5355.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 14:15:00 | 5427.50 | 5434.18 | 5410.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-01 15:00:00 | 5427.50 | 5434.18 | 5410.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 5392.50 | 5426.38 | 5410.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 11:00:00 | 5526.00 | 5446.30 | 5421.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 15:15:00 | 5520.00 | 5554.52 | 5558.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 15:15:00 | 5520.00 | 5554.52 | 5558.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 5487.50 | 5541.12 | 5552.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 12:15:00 | 5526.50 | 5515.81 | 5527.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 12:15:00 | 5526.50 | 5515.81 | 5527.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 5526.50 | 5515.81 | 5527.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:45:00 | 5523.50 | 5515.81 | 5527.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 5521.00 | 5516.85 | 5526.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 14:30:00 | 5498.00 | 5514.18 | 5524.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 5552.50 | 5521.09 | 5525.74 | SL hit (close>static) qty=1.00 sl=5532.50 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 11:15:00 | 5552.00 | 5529.98 | 5529.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 12:15:00 | 5577.50 | 5539.48 | 5533.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 15:15:00 | 5560.00 | 5561.61 | 5552.08 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 09:15:00 | 5594.50 | 5561.61 | 5552.08 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 5518.00 | 5563.95 | 5558.48 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-15 13:15:00 | 5518.00 | 5563.95 | 5558.48 | SL hit (close<ema400) qty=1.00 sl=5558.48 alert=retest1 |

### Cycle 28 — SELL (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 15:15:00 | 5535.00 | 5553.61 | 5554.44 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 5583.00 | 5559.49 | 5557.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 10:15:00 | 5613.00 | 5584.05 | 5573.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 15:15:00 | 5627.50 | 5635.43 | 5614.43 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-21 13:45:00 | 5650.00 | 5638.84 | 5617.89 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 5646.50 | 5639.12 | 5621.57 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-23 14:15:00 | 5571.50 | 5627.76 | 5623.56 | SL hit (close<ema400) qty=1.00 sl=5623.56 alert=retest1 |

### Cycle 30 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 5584.00 | 5619.01 | 5619.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 10:15:00 | 5560.00 | 5604.48 | 5613.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 5596.50 | 5563.61 | 5583.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 5596.50 | 5563.61 | 5583.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 5596.50 | 5563.61 | 5583.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 5596.50 | 5563.61 | 5583.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 5647.50 | 5580.39 | 5588.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:45:00 | 5652.00 | 5580.39 | 5588.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 12:15:00 | 5651.00 | 5601.49 | 5597.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 5655.00 | 5630.44 | 5614.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 11:15:00 | 5612.50 | 5627.66 | 5615.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 11:15:00 | 5612.50 | 5627.66 | 5615.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 5612.50 | 5627.66 | 5615.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 5612.50 | 5627.66 | 5615.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 5605.00 | 5623.13 | 5614.79 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 5566.50 | 5606.17 | 5608.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 11:15:00 | 5509.00 | 5548.97 | 5572.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 09:15:00 | 5616.00 | 5544.75 | 5559.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 5616.00 | 5544.75 | 5559.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 5616.00 | 5544.75 | 5559.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 5616.00 | 5544.75 | 5559.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 5613.00 | 5558.40 | 5564.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 5613.00 | 5558.40 | 5564.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 13:15:00 | 5572.00 | 5568.32 | 5568.14 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 5546.50 | 5563.95 | 5566.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 09:15:00 | 5534.00 | 5555.73 | 5561.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 5349.00 | 5342.02 | 5401.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 12:45:00 | 5357.00 | 5342.02 | 5401.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 5374.00 | 5317.65 | 5343.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 5362.00 | 5317.65 | 5343.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 5389.50 | 5332.02 | 5347.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:45:00 | 5386.50 | 5332.02 | 5347.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 5396.00 | 5360.95 | 5358.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 11:15:00 | 5411.00 | 5370.36 | 5363.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 5520.00 | 5523.96 | 5485.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 5520.00 | 5523.96 | 5485.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 5498.00 | 5518.77 | 5486.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 5549.00 | 5518.77 | 5486.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:30:00 | 5603.00 | 5534.79 | 5499.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:00:00 | 5537.50 | 5552.47 | 5521.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-26 09:15:00 | 6103.90 | 6065.83 | 6029.25 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 13:15:00 | 6209.50 | 6231.65 | 6232.99 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 6315.50 | 6246.12 | 6238.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 09:15:00 | 6376.00 | 6323.76 | 6288.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 13:15:00 | 6339.00 | 6339.94 | 6308.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-05 13:45:00 | 6332.50 | 6339.94 | 6308.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 6231.00 | 6322.91 | 6309.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 6231.00 | 6322.91 | 6309.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 6217.50 | 6301.83 | 6300.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 6217.50 | 6301.83 | 6300.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 6199.50 | 6281.36 | 6291.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 6173.00 | 6245.71 | 6272.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 6004.00 | 5977.17 | 6043.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 10:00:00 | 6004.00 | 5977.17 | 6043.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 5999.50 | 5979.29 | 6011.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:45:00 | 6033.00 | 5979.29 | 6011.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 5994.50 | 5982.33 | 6009.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 11:45:00 | 5975.50 | 5981.27 | 6006.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 12:45:00 | 5974.50 | 5979.01 | 6003.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:15:00 | 5676.72 | 5811.90 | 5873.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:15:00 | 5675.77 | 5811.90 | 5873.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 10:15:00 | 5768.50 | 5753.63 | 5800.83 | SL hit (close>ema200) qty=0.50 sl=5753.63 alert=retest2 |

### Cycle 39 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 5721.50 | 5662.89 | 5655.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 10:15:00 | 5746.00 | 5694.76 | 5672.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 15:15:00 | 5990.00 | 5992.03 | 5956.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 09:15:00 | 5981.00 | 5992.03 | 5956.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 5950.00 | 5986.68 | 5967.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:30:00 | 5951.00 | 5986.68 | 5967.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 5989.00 | 5987.15 | 5969.85 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 5901.50 | 5961.88 | 5962.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 5863.00 | 5942.10 | 5953.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 5749.00 | 5717.53 | 5752.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 14:15:00 | 5749.00 | 5717.53 | 5752.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 5749.00 | 5717.53 | 5752.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 5749.00 | 5717.53 | 5752.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 5705.00 | 5715.02 | 5748.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:45:00 | 5698.00 | 5708.82 | 5742.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:00:00 | 5700.00 | 5708.28 | 5736.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:45:00 | 5701.00 | 5695.20 | 5712.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:15:00 | 5700.00 | 5695.20 | 5712.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 5668.00 | 5689.76 | 5708.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:15:00 | 5650.50 | 5689.76 | 5708.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 12:15:00 | 5729.00 | 5694.94 | 5699.05 | SL hit (close>static) qty=1.00 sl=5720.00 alert=retest2 |

### Cycle 41 — BUY (started 2026-01-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 13:15:00 | 5744.50 | 5704.85 | 5703.19 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2026-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 13:15:00 | 5620.00 | 5699.71 | 5705.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 5583.50 | 5676.47 | 5694.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 5615.00 | 5578.25 | 5618.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 5615.00 | 5578.25 | 5618.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 5615.00 | 5578.25 | 5618.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 5637.50 | 5578.25 | 5618.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 5583.50 | 5579.30 | 5614.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:45:00 | 5566.50 | 5579.54 | 5611.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:30:00 | 5569.00 | 5571.53 | 5605.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 15:15:00 | 5510.00 | 5425.90 | 5425.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 5510.00 | 5425.90 | 5425.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 09:15:00 | 5550.50 | 5450.82 | 5436.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 11:15:00 | 5534.00 | 5539.21 | 5503.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 12:00:00 | 5534.00 | 5539.21 | 5503.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 5481.50 | 5527.67 | 5501.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 5481.50 | 5527.67 | 5501.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 5515.50 | 5525.23 | 5502.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 14:45:00 | 5536.00 | 5525.99 | 5505.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 12:30:00 | 5542.00 | 5563.35 | 5535.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 5531.00 | 5543.52 | 5532.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 10:15:00 | 5430.50 | 5510.11 | 5518.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 5430.50 | 5510.11 | 5518.43 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2026-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 13:15:00 | 5566.50 | 5526.81 | 5524.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 14:15:00 | 5617.00 | 5544.85 | 5532.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 5801.00 | 5814.30 | 5743.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:45:00 | 5794.50 | 5814.30 | 5743.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 5764.50 | 5791.60 | 5757.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:45:00 | 5759.00 | 5791.60 | 5757.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 5800.00 | 5791.51 | 5763.60 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 09:15:00 | 5725.50 | 5760.97 | 5761.27 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 10:15:00 | 5785.50 | 5759.49 | 5757.17 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 5668.00 | 5745.55 | 5753.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 10:15:00 | 5527.50 | 5600.11 | 5650.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 10:15:00 | 5536.00 | 5525.07 | 5577.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-17 10:30:00 | 5528.00 | 5525.07 | 5577.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 5554.00 | 5535.67 | 5569.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:00:00 | 5554.00 | 5535.67 | 5569.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 5577.50 | 5544.03 | 5570.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 15:00:00 | 5577.50 | 5544.03 | 5570.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 5565.00 | 5548.23 | 5569.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 5609.50 | 5548.23 | 5569.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 5575.50 | 5553.68 | 5570.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:30:00 | 5624.50 | 5553.68 | 5570.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 5583.00 | 5559.55 | 5571.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:45:00 | 5599.50 | 5559.55 | 5571.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 5563.00 | 5560.24 | 5570.84 | EMA400 retest candle locked (from downside) |

### Cycle 49 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 5594.50 | 5575.04 | 5574.99 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 5550.00 | 5570.03 | 5572.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 10:15:00 | 5526.50 | 5561.33 | 5568.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 5436.50 | 5435.23 | 5480.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 14:00:00 | 5436.50 | 5435.23 | 5480.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 5475.50 | 5451.06 | 5477.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 5504.50 | 5451.06 | 5477.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 5498.50 | 5460.55 | 5479.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:30:00 | 5465.50 | 5459.74 | 5476.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 10:15:00 | 5505.50 | 5483.05 | 5481.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 10:15:00 | 5505.50 | 5483.05 | 5481.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 15:15:00 | 5520.00 | 5498.49 | 5490.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 5747.00 | 5749.79 | 5691.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 10:15:00 | 5738.50 | 5749.79 | 5691.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 5743.00 | 5739.83 | 5704.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:30:00 | 5703.50 | 5739.83 | 5704.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 5705.50 | 5732.96 | 5704.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 5705.50 | 5732.96 | 5704.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 5715.00 | 5729.37 | 5705.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 5679.00 | 5729.37 | 5705.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 5681.50 | 5719.80 | 5703.67 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 5561.00 | 5669.99 | 5682.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 5530.50 | 5642.09 | 5668.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 5525.50 | 5514.43 | 5563.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 5525.50 | 5514.43 | 5563.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 5525.50 | 5514.43 | 5563.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 11:15:00 | 5497.00 | 5514.04 | 5558.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 5571.50 | 5531.01 | 5552.80 | SL hit (close>static) qty=1.00 sl=5570.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 5685.00 | 5525.70 | 5517.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 5704.00 | 5609.22 | 5563.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 5615.00 | 5632.73 | 5588.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 10:45:00 | 5620.00 | 5632.73 | 5588.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 5618.50 | 5629.88 | 5590.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:45:00 | 5599.50 | 5629.88 | 5590.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 5578.50 | 5619.61 | 5589.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 5578.50 | 5619.61 | 5589.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 5583.50 | 5612.38 | 5589.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:45:00 | 5569.00 | 5612.38 | 5589.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 5561.00 | 5602.11 | 5586.56 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 5419.50 | 5560.45 | 5570.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 15:15:00 | 5390.50 | 5459.72 | 5508.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 12:15:00 | 5247.50 | 5236.20 | 5315.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 13:00:00 | 5247.50 | 5236.20 | 5315.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 5282.00 | 5246.85 | 5306.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:30:00 | 5303.50 | 5246.85 | 5306.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 5360.50 | 5271.92 | 5307.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 5360.50 | 5271.92 | 5307.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 5322.00 | 5281.94 | 5309.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 5296.00 | 5281.94 | 5309.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 15:15:00 | 5353.00 | 5326.38 | 5322.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 5353.00 | 5326.38 | 5322.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 5420.50 | 5345.21 | 5331.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 5297.50 | 5392.29 | 5373.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 5297.50 | 5392.29 | 5373.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 5297.50 | 5392.29 | 5373.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 5297.50 | 5392.29 | 5373.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 5280.50 | 5353.32 | 5357.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 5190.00 | 5306.29 | 5334.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 5283.00 | 5266.83 | 5306.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 09:45:00 | 5295.50 | 5266.83 | 5306.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 5350.50 | 5283.56 | 5310.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 5337.50 | 5283.56 | 5310.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 5293.00 | 5285.45 | 5308.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 5272.50 | 5285.45 | 5308.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:30:00 | 5267.50 | 5280.15 | 5302.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 13:45:00 | 5286.50 | 5192.72 | 5201.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 14:30:00 | 5282.00 | 5203.28 | 5205.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 15:15:00 | 5258.00 | 5214.22 | 5210.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 5258.00 | 5214.22 | 5210.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 5332.50 | 5237.88 | 5221.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 5289.50 | 5293.02 | 5259.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 14:45:00 | 5299.00 | 5293.02 | 5259.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 5158.00 | 5263.61 | 5251.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:45:00 | 5167.00 | 5263.61 | 5251.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 5154.50 | 5241.79 | 5243.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 5140.00 | 5221.43 | 5233.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 5169.50 | 5116.69 | 5151.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 5169.50 | 5116.69 | 5151.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 5169.50 | 5116.69 | 5151.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:00:00 | 5136.50 | 5120.65 | 5150.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 5125.00 | 5137.62 | 5151.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 5335.50 | 5097.26 | 5075.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 5335.50 | 5097.26 | 5075.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 5365.50 | 5297.68 | 5243.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 5343.50 | 5395.65 | 5331.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 5343.50 | 5395.65 | 5331.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 5343.50 | 5395.65 | 5331.58 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 14:15:00 | 5253.00 | 5296.37 | 5300.68 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 11:15:00 | 5318.50 | 5302.93 | 5301.82 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 13:15:00 | 5290.00 | 5300.04 | 5300.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-15 14:15:00 | 5289.50 | 5297.93 | 5299.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 09:15:00 | 5233.50 | 5198.86 | 5233.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-17 09:15:00 | 5233.50 | 5198.86 | 5233.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 5233.50 | 5198.86 | 5233.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 09:30:00 | 5199.00 | 5198.86 | 5233.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 5227.00 | 5204.49 | 5232.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 12:15:00 | 5216.50 | 5208.29 | 5232.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 13:00:00 | 5216.00 | 5209.83 | 5230.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 14:00:00 | 5215.50 | 5210.97 | 5229.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 09:15:00 | 5276.50 | 5230.32 | 5234.10 | SL hit (close>static) qty=1.00 sl=5250.00 alert=retest2 |

### Cycle 63 — BUY (started 2026-04-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 10:15:00 | 5281.50 | 5240.56 | 5238.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 11:15:00 | 5295.00 | 5251.45 | 5243.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 15:15:00 | 5254.50 | 5265.64 | 5254.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 15:15:00 | 5254.50 | 5265.64 | 5254.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 5254.50 | 5265.64 | 5254.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:00:00 | 5295.00 | 5271.51 | 5257.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 11:00:00 | 5295.00 | 5276.21 | 5261.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 11:45:00 | 5297.00 | 5281.37 | 5264.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 12:15:00 | 5221.00 | 5260.96 | 5262.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 12:15:00 | 5221.00 | 5260.96 | 5262.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 13:15:00 | 5215.00 | 5251.77 | 5257.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 5071.00 | 5015.24 | 5068.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 5071.00 | 5015.24 | 5068.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 5071.00 | 5015.24 | 5068.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 5071.00 | 5015.24 | 5068.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 5062.00 | 5024.59 | 5068.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:30:00 | 5052.00 | 5046.29 | 5068.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:00:00 | 5054.50 | 5046.29 | 5068.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:45:00 | 5035.50 | 5052.18 | 5063.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 13:30:00 | 5055.50 | 5050.85 | 5060.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 5057.50 | 5052.18 | 5060.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 5057.50 | 5052.18 | 5060.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 5083.00 | 5058.35 | 5062.55 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-28 15:15:00 | 5083.00 | 5058.35 | 5062.55 | SL hit (close>static) qty=1.00 sl=5082.50 alert=retest2 |

### Cycle 65 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 5170.50 | 5080.78 | 5072.37 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 5032.00 | 5086.34 | 5087.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 11:15:00 | 5029.50 | 5074.97 | 5082.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 13:15:00 | 5073.50 | 5069.72 | 5078.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 13:15:00 | 5073.50 | 5069.72 | 5078.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 5073.50 | 5069.72 | 5078.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:00:00 | 5073.50 | 5069.72 | 5078.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 5105.50 | 5076.88 | 5080.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:45:00 | 5112.00 | 5076.88 | 5080.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 15:15:00 | 5110.00 | 5083.50 | 5083.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 5148.50 | 5096.50 | 5089.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 11:15:00 | 5086.50 | 5098.50 | 5091.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 11:15:00 | 5086.50 | 5098.50 | 5091.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 5086.50 | 5098.50 | 5091.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 12:00:00 | 5086.50 | 5098.50 | 5091.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 12:15:00 | 5024.00 | 5083.60 | 5085.63 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 15:15:00 | 5127.00 | 5087.80 | 5082.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 5255.50 | 5121.34 | 5098.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 10:15:00 | 4985.00 | 5094.07 | 5088.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 10:15:00 | 4985.00 | 5094.07 | 5088.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 4985.00 | 5094.07 | 5088.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:00:00 | 4985.00 | 5094.07 | 5088.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 5080.00 | 5091.26 | 5087.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 12:15:00 | 5091.00 | 5091.26 | 5087.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 13:00:00 | 5089.50 | 5090.91 | 5087.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 14:00:00 | 5092.00 | 5091.13 | 5087.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-06-26 09:30:00 | 4256.00 | 2025-06-27 10:15:00 | 4320.50 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-07-21 11:45:00 | 4418.00 | 2025-07-22 10:15:00 | 4383.50 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-07-21 12:30:00 | 4418.50 | 2025-07-22 10:15:00 | 4383.50 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-07-22 09:45:00 | 4414.10 | 2025-07-22 10:15:00 | 4383.50 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-07-24 09:15:00 | 4296.10 | 2025-07-29 10:15:00 | 4321.50 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-08-07 12:45:00 | 4607.60 | 2025-08-18 09:15:00 | 5068.36 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-12 09:15:00 | 4604.00 | 2025-08-18 09:15:00 | 5064.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-28 09:15:00 | 5151.00 | 2025-09-10 11:15:00 | 5348.50 | STOP_HIT | 1.00 | 3.83% |
| BUY | retest2 | 2025-08-29 09:45:00 | 5102.90 | 2025-09-10 11:15:00 | 5348.50 | STOP_HIT | 1.00 | 4.81% |
| SELL | retest2 | 2025-09-12 13:30:00 | 5293.00 | 2025-09-16 13:15:00 | 5333.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-09-15 11:30:00 | 5291.00 | 2025-09-16 13:15:00 | 5333.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-09-16 09:30:00 | 5293.00 | 2025-09-16 13:15:00 | 5333.00 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-09-18 15:00:00 | 5375.50 | 2025-09-23 14:15:00 | 5374.00 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-10-03 11:00:00 | 5526.00 | 2025-10-08 15:15:00 | 5520.00 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-10-10 14:30:00 | 5498.00 | 2025-10-13 09:15:00 | 5552.50 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest1 | 2025-10-15 09:15:00 | 5594.50 | 2025-10-15 13:15:00 | 5518.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest1 | 2025-10-21 13:45:00 | 5650.00 | 2025-10-23 14:15:00 | 5571.50 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-11-14 09:15:00 | 5549.00 | 2025-11-26 09:15:00 | 6103.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-14 10:30:00 | 5603.00 | 2025-11-26 09:15:00 | 6091.25 | TARGET_HIT | 1.00 | 8.71% |
| BUY | retest2 | 2025-11-14 15:00:00 | 5537.50 | 2025-11-27 09:15:00 | 6163.30 | TARGET_HIT | 1.00 | 11.30% |
| SELL | retest2 | 2025-12-12 11:45:00 | 5975.50 | 2025-12-18 09:15:00 | 5676.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 12:45:00 | 5974.50 | 2025-12-18 09:15:00 | 5675.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 11:45:00 | 5975.50 | 2025-12-19 10:15:00 | 5768.50 | STOP_HIT | 0.50 | 3.46% |
| SELL | retest2 | 2025-12-12 12:45:00 | 5974.50 | 2025-12-19 10:15:00 | 5768.50 | STOP_HIT | 0.50 | 3.45% |
| SELL | retest2 | 2026-01-14 09:45:00 | 5698.00 | 2026-01-19 12:15:00 | 5729.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-01-14 12:00:00 | 5700.00 | 2026-01-19 13:15:00 | 5744.50 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2026-01-16 11:45:00 | 5701.00 | 2026-01-19 13:15:00 | 5744.50 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-01-16 12:15:00 | 5700.00 | 2026-01-19 13:15:00 | 5744.50 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2026-01-16 13:15:00 | 5650.50 | 2026-01-19 13:15:00 | 5744.50 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-01-22 11:45:00 | 5566.50 | 2026-01-28 15:15:00 | 5510.00 | STOP_HIT | 1.00 | 1.02% |
| SELL | retest2 | 2026-01-22 12:30:00 | 5569.00 | 2026-01-28 15:15:00 | 5510.00 | STOP_HIT | 1.00 | 1.06% |
| BUY | retest2 | 2026-01-30 14:45:00 | 5536.00 | 2026-02-02 10:15:00 | 5430.50 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2026-02-01 12:30:00 | 5542.00 | 2026-02-02 10:15:00 | 5430.50 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2026-02-02 09:15:00 | 5531.00 | 2026-02-02 10:15:00 | 5430.50 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2026-02-23 11:30:00 | 5465.50 | 2026-02-24 10:15:00 | 5505.50 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-03-05 11:15:00 | 5497.00 | 2026-03-05 14:15:00 | 5571.50 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-03-06 14:45:00 | 5497.50 | 2026-03-10 10:15:00 | 5685.00 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2026-03-09 09:15:00 | 5351.50 | 2026-03-10 10:15:00 | 5685.00 | STOP_HIT | 1.00 | -6.23% |
| SELL | retest2 | 2026-03-17 11:15:00 | 5296.00 | 2026-03-17 15:15:00 | 5353.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-03-20 12:15:00 | 5272.50 | 2026-03-24 15:15:00 | 5258.00 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest2 | 2026-03-20 13:30:00 | 5267.50 | 2026-03-24 15:15:00 | 5258.00 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2026-03-24 13:45:00 | 5286.50 | 2026-03-24 15:15:00 | 5258.00 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2026-03-24 14:30:00 | 5282.00 | 2026-03-24 15:15:00 | 5258.00 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2026-04-01 11:00:00 | 5136.50 | 2026-04-08 09:15:00 | 5335.50 | STOP_HIT | 1.00 | -3.87% |
| SELL | retest2 | 2026-04-01 13:30:00 | 5125.00 | 2026-04-08 09:15:00 | 5335.50 | STOP_HIT | 1.00 | -4.11% |
| SELL | retest2 | 2026-04-17 12:15:00 | 5216.50 | 2026-04-20 09:15:00 | 5276.50 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2026-04-17 13:00:00 | 5216.00 | 2026-04-20 09:15:00 | 5276.50 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-04-17 14:00:00 | 5215.50 | 2026-04-20 09:15:00 | 5276.50 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2026-04-21 10:00:00 | 5295.00 | 2026-04-22 12:15:00 | 5221.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2026-04-21 11:00:00 | 5295.00 | 2026-04-22 12:15:00 | 5221.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2026-04-21 11:45:00 | 5297.00 | 2026-04-22 12:15:00 | 5221.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-04-27 13:30:00 | 5052.00 | 2026-04-28 15:15:00 | 5083.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2026-04-27 14:00:00 | 5054.50 | 2026-04-28 15:15:00 | 5083.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2026-04-28 11:45:00 | 5035.50 | 2026-04-28 15:15:00 | 5083.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2026-04-28 13:30:00 | 5055.50 | 2026-04-28 15:15:00 | 5083.00 | STOP_HIT | 1.00 | -0.54% |
