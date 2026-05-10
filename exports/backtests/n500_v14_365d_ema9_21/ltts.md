# L&T Technology Services Ltd. (LTTS)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 3801.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 67 |
| ALERT1 | 52 |
| ALERT2 | 52 |
| ALERT2_SKIP | 26 |
| ALERT3 | 110 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 61 |
| PARTIAL | 7 |
| TARGET_HIT | 4 |
| STOP_HIT | 58 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 68 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 27 / 41
- **Target hits / Stop hits / Partials:** 4 / 57 / 7
- **Avg / median % per leg:** 0.68% / -0.44%
- **Sum % (uncompounded):** 46.36%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 5 | 33.3% | 0 | 15 | 0 | -0.38% | -5.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 5 | 33.3% | 0 | 15 | 0 | -0.38% | -5.6% |
| SELL (all) | 53 | 22 | 41.5% | 4 | 42 | 7 | 0.98% | 52.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 53 | 22 | 41.5% | 4 | 42 | 7 | 0.98% | 52.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 68 | 27 | 39.7% | 4 | 57 | 7 | 0.68% | 46.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 4276.30 | 4131.04 | 4119.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 4346.70 | 4220.41 | 4167.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 4375.80 | 4379.05 | 4300.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 4375.80 | 4379.05 | 4300.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 11:15:00 | 4510.90 | 4496.63 | 4452.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 11:30:00 | 4517.80 | 4496.63 | 4452.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 4469.30 | 4489.90 | 4460.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 14:45:00 | 4477.00 | 4489.90 | 4460.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 4542.00 | 4498.41 | 4469.14 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 09:15:00 | 4469.90 | 4486.14 | 4487.64 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 4573.30 | 4484.99 | 4477.27 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 10:15:00 | 4465.70 | 4483.27 | 4483.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 4429.40 | 4452.48 | 4466.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 4446.70 | 4417.36 | 4436.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 4446.70 | 4417.36 | 4436.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 4446.70 | 4417.36 | 4436.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 12:30:00 | 4415.10 | 4426.60 | 4436.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-29 14:00:00 | 4416.10 | 4424.50 | 4435.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-09 15:15:00 | 4334.50 | 4330.24 | 4329.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-09 15:15:00 | 4334.50 | 4330.24 | 4329.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 15:15:00 | 4334.50 | 4330.24 | 4329.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 4422.00 | 4348.59 | 4338.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 11:15:00 | 4531.80 | 4532.43 | 4483.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 12:00:00 | 4531.80 | 4532.43 | 4483.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 4456.20 | 4514.84 | 4493.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 4473.20 | 4514.84 | 4493.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 12:15:00 | 4428.10 | 4478.95 | 4480.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 12:15:00 | 4428.10 | 4478.95 | 4480.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 14:15:00 | 4421.10 | 4460.03 | 4471.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 4462.20 | 4458.49 | 4468.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 4462.20 | 4458.49 | 4468.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 4462.20 | 4458.49 | 4468.61 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 4534.00 | 4482.73 | 4477.04 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 4449.00 | 4476.69 | 4477.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 14:15:00 | 4447.90 | 4468.24 | 4473.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 10:15:00 | 4343.90 | 4340.46 | 4379.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 10:45:00 | 4344.70 | 4340.46 | 4379.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 4373.60 | 4347.09 | 4378.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:00:00 | 4373.60 | 4347.09 | 4378.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 4350.00 | 4347.67 | 4376.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:45:00 | 4341.10 | 4359.06 | 4373.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 11:00:00 | 4337.30 | 4354.71 | 4370.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 11:30:00 | 4341.80 | 4352.34 | 4367.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 12:45:00 | 4336.00 | 4350.60 | 4365.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 13:15:00 | 4351.10 | 4350.70 | 4364.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 13:30:00 | 4352.00 | 4350.70 | 4364.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 4375.90 | 4352.38 | 4361.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 13:00:00 | 4355.00 | 4356.07 | 4361.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 12:15:00 | 4350.00 | 4355.94 | 4357.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 13:00:00 | 4354.70 | 4355.69 | 4357.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-25 13:30:00 | 4353.90 | 4356.11 | 4357.42 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 4365.00 | 4357.89 | 4358.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:30:00 | 4363.00 | 4357.89 | 4358.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-25 15:15:00 | 4365.00 | 4359.31 | 4358.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 15:15:00 | 4365.00 | 4359.31 | 4358.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 15:15:00 | 4365.00 | 4359.31 | 4358.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 15:15:00 | 4365.00 | 4359.31 | 4358.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 15:15:00 | 4365.00 | 4359.31 | 4358.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 15:15:00 | 4365.00 | 4359.31 | 4358.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 15:15:00 | 4365.00 | 4359.31 | 4358.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 15:15:00 | 4365.00 | 4359.31 | 4358.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 15:15:00 | 4365.00 | 4359.31 | 4358.74 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 09:15:00 | 4321.80 | 4351.81 | 4355.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 10:15:00 | 4311.20 | 4343.69 | 4351.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 09:15:00 | 4378.00 | 4337.96 | 4343.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 09:15:00 | 4378.00 | 4337.96 | 4343.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 4378.00 | 4337.96 | 4343.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:45:00 | 4357.80 | 4337.96 | 4343.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 4345.50 | 4339.47 | 4343.64 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 12:15:00 | 4358.60 | 4346.97 | 4346.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 14:15:00 | 4381.80 | 4357.12 | 4351.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-01 09:15:00 | 4371.20 | 4384.42 | 4372.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-01 09:15:00 | 4371.20 | 4384.42 | 4372.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 4371.20 | 4384.42 | 4372.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:00:00 | 4371.20 | 4384.42 | 4372.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 4336.80 | 4374.89 | 4369.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 4336.80 | 4374.89 | 4369.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 4325.20 | 4364.95 | 4365.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 13:15:00 | 4313.50 | 4346.67 | 4354.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 10:15:00 | 4348.60 | 4337.68 | 4346.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 10:15:00 | 4348.60 | 4337.68 | 4346.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 4348.60 | 4337.68 | 4346.85 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 11:15:00 | 4368.40 | 4348.59 | 4347.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 12:15:00 | 4379.10 | 4354.69 | 4350.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 11:15:00 | 4362.40 | 4368.25 | 4361.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 11:15:00 | 4362.40 | 4368.25 | 4361.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 4362.40 | 4368.25 | 4361.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:00:00 | 4362.40 | 4368.25 | 4361.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 4368.10 | 4368.22 | 4361.76 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2025-07-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 14:15:00 | 4351.80 | 4361.15 | 4361.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 09:15:00 | 4324.40 | 4351.20 | 4357.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 12:15:00 | 4307.00 | 4306.84 | 4325.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 13:00:00 | 4307.00 | 4306.84 | 4325.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 4368.00 | 4319.10 | 4327.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-10 15:00:00 | 4368.00 | 4319.10 | 4327.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 4348.00 | 4324.88 | 4329.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 4371.50 | 4324.88 | 4329.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 09:15:00 | 4384.80 | 4336.86 | 4334.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-11 14:15:00 | 4415.00 | 4365.23 | 4350.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-14 09:15:00 | 4349.60 | 4368.47 | 4354.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 09:15:00 | 4349.60 | 4368.47 | 4354.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 4349.60 | 4368.47 | 4354.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:00:00 | 4349.60 | 4368.47 | 4354.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 10:15:00 | 4341.40 | 4363.05 | 4353.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-14 10:30:00 | 4342.00 | 4363.05 | 4353.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 14:15:00 | 4339.90 | 4347.00 | 4347.77 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 4363.70 | 4348.42 | 4348.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 11:15:00 | 4391.10 | 4363.03 | 4355.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 13:15:00 | 4352.80 | 4361.24 | 4356.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 13:15:00 | 4352.80 | 4361.24 | 4356.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 4352.80 | 4361.24 | 4356.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:45:00 | 4348.80 | 4361.24 | 4356.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 4350.80 | 4359.15 | 4355.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 15:00:00 | 4350.80 | 4359.15 | 4355.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 4344.00 | 4356.12 | 4354.71 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 4379.40 | 4356.12 | 4354.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:15:00 | 4369.70 | 4395.21 | 4393.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-21 10:15:00 | 4349.00 | 4385.97 | 4389.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-21 10:15:00 | 4349.00 | 4385.97 | 4389.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 10:15:00 | 4349.00 | 4385.97 | 4389.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 10:15:00 | 4337.70 | 4361.21 | 4373.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 10:15:00 | 4320.30 | 4311.89 | 4327.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-24 10:30:00 | 4316.40 | 4311.89 | 4327.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 4244.30 | 4227.94 | 4244.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 4244.30 | 4227.94 | 4244.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 4300.30 | 4242.41 | 4249.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:45:00 | 4307.00 | 4242.41 | 4249.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 4282.00 | 4250.33 | 4252.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 4283.60 | 4250.33 | 4252.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 4273.60 | 4257.47 | 4255.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 11:15:00 | 4280.00 | 4261.98 | 4257.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 4270.00 | 4279.96 | 4269.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 4270.00 | 4279.96 | 4269.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 4270.00 | 4279.96 | 4269.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:45:00 | 4261.00 | 4279.96 | 4269.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 4280.20 | 4280.01 | 4270.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:30:00 | 4300.00 | 4282.73 | 4272.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 10:00:00 | 4298.80 | 4298.77 | 4285.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 09:15:00 | 4260.70 | 4285.38 | 4285.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-04 09:15:00 | 4260.70 | 4285.38 | 4285.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 4260.70 | 4285.38 | 4285.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 4206.60 | 4237.67 | 4247.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 11:15:00 | 4228.80 | 4224.86 | 4235.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-08 11:45:00 | 4232.70 | 4224.86 | 4235.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 4227.60 | 4225.67 | 4233.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:30:00 | 4229.00 | 4225.67 | 4233.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 4175.70 | 4215.68 | 4228.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 14:30:00 | 4220.00 | 4215.68 | 4228.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 4195.00 | 4165.88 | 4186.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:00:00 | 4195.00 | 4165.88 | 4186.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 4177.70 | 4168.25 | 4185.82 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 11:15:00 | 4216.70 | 4196.59 | 4194.49 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2025-08-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 13:15:00 | 4176.90 | 4191.05 | 4192.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 14:15:00 | 4146.60 | 4182.16 | 4188.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 10:15:00 | 4188.10 | 4175.38 | 4182.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 10:15:00 | 4188.10 | 4175.38 | 4182.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 4188.10 | 4175.38 | 4182.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:00:00 | 4188.10 | 4175.38 | 4182.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 4164.30 | 4173.17 | 4180.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 13:00:00 | 4153.80 | 4169.29 | 4178.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 14:30:00 | 4153.10 | 4165.83 | 4175.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 12:00:00 | 4158.20 | 4165.60 | 4172.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 12:30:00 | 4155.00 | 4167.48 | 4172.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 13:15:00 | 4177.40 | 4169.47 | 4172.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 14:45:00 | 4161.50 | 4167.75 | 4171.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-19 09:30:00 | 4161.00 | 4166.74 | 4170.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 10:15:00 | 4195.10 | 4172.41 | 4172.78 | SL hit (close>static) qty=1.00 sl=4192.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-19 10:15:00 | 4195.10 | 4172.41 | 4172.78 | SL hit (close>static) qty=1.00 sl=4192.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-19 10:15:00 | 4195.10 | 4172.41 | 4172.78 | SL hit (close>static) qty=1.00 sl=4192.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-19 10:15:00 | 4195.10 | 4172.41 | 4172.78 | SL hit (close>static) qty=1.00 sl=4192.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-19 10:15:00 | 4195.10 | 4172.41 | 4172.78 | SL hit (close>static) qty=1.00 sl=4185.90 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-19 10:15:00 | 4195.10 | 4172.41 | 4172.78 | SL hit (close>static) qty=1.00 sl=4185.90 alert=retest2 |

### Cycle 23 — BUY (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 11:15:00 | 4204.60 | 4178.85 | 4175.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 11:15:00 | 4232.80 | 4192.67 | 4183.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-22 09:15:00 | 4255.70 | 4270.17 | 4243.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-22 09:30:00 | 4258.30 | 4270.17 | 4243.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 4238.20 | 4263.77 | 4242.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 11:00:00 | 4238.20 | 4263.77 | 4242.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 11:15:00 | 4245.80 | 4260.18 | 4243.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 12:45:00 | 4267.90 | 4263.14 | 4246.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 13:15:00 | 4298.50 | 4314.70 | 4314.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 13:15:00 | 4298.50 | 4314.70 | 4314.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 4215.30 | 4294.82 | 4305.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 4265.50 | 4244.76 | 4265.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 4265.50 | 4244.76 | 4265.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 4265.50 | 4244.76 | 4265.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 11:15:00 | 4216.00 | 4244.13 | 4255.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:15:00 | 4212.00 | 4236.21 | 4249.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 4234.60 | 4132.14 | 4129.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 4234.60 | 4132.14 | 4129.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 4234.60 | 4132.14 | 4129.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 14:15:00 | 4256.30 | 4229.04 | 4214.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-19 10:15:00 | 4358.10 | 4371.73 | 4333.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-19 10:45:00 | 4359.80 | 4371.73 | 4333.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 4336.80 | 4360.47 | 4334.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:00:00 | 4336.80 | 4360.47 | 4334.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 4345.10 | 4357.39 | 4335.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 15:15:00 | 4360.00 | 4353.94 | 4336.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 4249.00 | 4333.92 | 4330.11 | SL hit (close<static) qty=1.00 sl=4330.60 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 10:15:00 | 4239.90 | 4315.12 | 4321.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 4210.70 | 4273.39 | 4298.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 15:15:00 | 4214.00 | 4207.47 | 4241.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 10:00:00 | 4211.60 | 4208.29 | 4239.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 4062.70 | 4073.43 | 4112.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 13:30:00 | 4031.00 | 4049.18 | 4088.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 15:15:00 | 4097.00 | 4083.47 | 4083.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-09-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 15:15:00 | 4097.00 | 4083.47 | 4083.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 11:15:00 | 4140.50 | 4103.55 | 4093.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 11:15:00 | 4284.00 | 4290.60 | 4253.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 11:45:00 | 4276.20 | 4290.60 | 4253.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 4259.90 | 4288.62 | 4270.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:00:00 | 4259.90 | 4288.62 | 4270.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 4254.50 | 4281.80 | 4268.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:30:00 | 4261.60 | 4281.80 | 4268.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 4219.10 | 4261.92 | 4262.33 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 13:15:00 | 4266.50 | 4260.76 | 4260.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 14:15:00 | 4274.60 | 4263.53 | 4261.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 4239.10 | 4289.36 | 4282.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 4239.10 | 4289.36 | 4282.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 4239.10 | 4289.36 | 4282.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:00:00 | 4239.10 | 4289.36 | 4282.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 4211.50 | 4273.79 | 4275.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 14:15:00 | 4198.60 | 4240.39 | 4257.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 4180.00 | 4171.05 | 4202.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 09:45:00 | 4185.10 | 4171.05 | 4202.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 4185.80 | 4158.49 | 4178.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:45:00 | 4194.50 | 4158.49 | 4178.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 4196.00 | 4165.99 | 4180.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 10:45:00 | 4187.20 | 4165.99 | 4180.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 15:15:00 | 4163.50 | 4161.00 | 4171.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 09:15:00 | 4170.00 | 4161.00 | 4171.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 4204.70 | 4169.74 | 4174.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:00:00 | 4204.70 | 4169.74 | 4174.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 4203.60 | 4176.51 | 4176.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:15:00 | 4215.00 | 4176.51 | 4176.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 4209.00 | 4183.01 | 4179.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 11:15:00 | 4236.60 | 4208.70 | 4197.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 4207.50 | 4214.77 | 4203.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 4207.50 | 4214.77 | 4203.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 4207.50 | 4214.77 | 4203.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 4207.50 | 4214.77 | 4203.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 4196.00 | 4211.02 | 4202.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 4191.00 | 4211.02 | 4202.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 4197.60 | 4208.34 | 4202.14 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 4170.00 | 4195.46 | 4197.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 4163.80 | 4185.62 | 4192.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 11:15:00 | 4186.90 | 4174.64 | 4183.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 11:15:00 | 4186.90 | 4174.64 | 4183.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 4186.90 | 4174.64 | 4183.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 11:45:00 | 4189.60 | 4174.64 | 4183.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 4211.20 | 4181.95 | 4185.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:00:00 | 4211.20 | 4181.95 | 4185.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 4213.60 | 4188.28 | 4188.27 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 09:15:00 | 4163.40 | 4185.08 | 4186.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 10:15:00 | 4150.00 | 4178.07 | 4183.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-30 14:15:00 | 4105.00 | 4098.25 | 4116.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-30 14:45:00 | 4105.00 | 4098.25 | 4116.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 4116.00 | 4101.80 | 4116.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 4116.00 | 4101.80 | 4116.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 4098.40 | 4101.12 | 4114.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 10:30:00 | 4095.10 | 4102.61 | 4114.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 4092.70 | 4112.57 | 4115.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 15:15:00 | 4117.00 | 4114.70 | 4114.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-03 15:15:00 | 4117.00 | 4114.70 | 4114.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 15:15:00 | 4117.00 | 4114.70 | 4114.59 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 4084.30 | 4108.62 | 4111.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 4079.20 | 4088.94 | 4098.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 4094.40 | 4086.34 | 4094.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 13:00:00 | 4094.40 | 4086.34 | 4094.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 4106.80 | 4090.43 | 4095.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:30:00 | 4100.00 | 4090.43 | 4095.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 4090.00 | 4090.34 | 4095.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:15:00 | 4115.00 | 4090.34 | 4095.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 4115.00 | 4095.27 | 4096.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 4057.60 | 4095.27 | 4096.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 15:15:00 | 4047.00 | 4070.33 | 4079.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 14:15:00 | 4085.10 | 4079.32 | 4079.75 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 14:15:00 | 4084.50 | 4080.35 | 4080.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-10 14:15:00 | 4084.50 | 4080.35 | 4080.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-10 14:15:00 | 4084.50 | 4080.35 | 4080.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 14:15:00 | 4084.50 | 4080.35 | 4080.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 10:15:00 | 4101.80 | 4085.98 | 4082.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 11:15:00 | 4082.60 | 4085.30 | 4082.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 11:15:00 | 4082.60 | 4085.30 | 4082.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 4082.60 | 4085.30 | 4082.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:00:00 | 4082.60 | 4085.30 | 4082.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 4086.50 | 4085.54 | 4083.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:30:00 | 4081.50 | 4085.54 | 4083.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 13:15:00 | 4135.60 | 4161.59 | 4144.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 14:00:00 | 4135.60 | 4161.59 | 4144.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 4160.00 | 4161.27 | 4145.75 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 4118.70 | 4136.68 | 4137.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 12:15:00 | 4093.60 | 4128.06 | 4133.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 15:15:00 | 4108.40 | 4106.10 | 4114.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-18 09:15:00 | 4122.40 | 4106.10 | 4114.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 4094.00 | 4103.68 | 4112.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 10:45:00 | 4087.90 | 4099.32 | 4109.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 11:30:00 | 4087.50 | 4096.94 | 4107.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 12:00:00 | 4087.40 | 4096.94 | 4107.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 4154.20 | 4098.67 | 4102.93 | SL hit (close>static) qty=1.00 sl=4126.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 4154.20 | 4098.67 | 4102.93 | SL hit (close>static) qty=1.00 sl=4126.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 09:15:00 | 4154.20 | 4098.67 | 4102.93 | SL hit (close>static) qty=1.00 sl=4126.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 10:15:00 | 4272.00 | 4133.33 | 4118.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 11:15:00 | 4376.20 | 4181.91 | 4141.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 15:15:00 | 4321.90 | 4345.23 | 4290.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 15:15:00 | 4321.90 | 4345.23 | 4290.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 4321.90 | 4345.23 | 4290.58 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 11:15:00 | 4254.50 | 4282.54 | 4283.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 12:15:00 | 4242.90 | 4274.61 | 4279.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 13:15:00 | 4310.00 | 4281.69 | 4282.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 13:15:00 | 4310.00 | 4281.69 | 4282.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 4310.00 | 4281.69 | 4282.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:00:00 | 4310.00 | 4281.69 | 4282.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 14:15:00 | 4326.20 | 4290.59 | 4286.61 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 09:15:00 | 4256.20 | 4282.55 | 4283.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 10:15:00 | 4237.40 | 4273.52 | 4279.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 14:15:00 | 4263.80 | 4261.50 | 4270.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 15:00:00 | 4263.80 | 4261.50 | 4270.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 4273.00 | 4265.02 | 4270.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 11:00:00 | 4242.40 | 4260.49 | 4268.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 11:30:00 | 4249.60 | 4257.95 | 4266.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 4309.30 | 4271.99 | 4269.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 4309.30 | 4271.99 | 4269.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 4309.30 | 4271.99 | 4269.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 10:15:00 | 4339.70 | 4285.53 | 4276.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 4410.00 | 4410.33 | 4370.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 10:00:00 | 4410.00 | 4410.33 | 4370.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 4463.70 | 4435.92 | 4404.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:45:00 | 4412.70 | 4435.92 | 4404.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 4460.70 | 4472.70 | 4442.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 09:15:00 | 4560.00 | 4496.64 | 4468.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-08 09:45:00 | 4536.50 | 4598.50 | 4590.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-08 11:00:00 | 4542.30 | 4587.26 | 4586.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 11:15:00 | 4541.00 | 4578.01 | 4582.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 11:15:00 | 4541.00 | 4578.01 | 4582.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-08 11:15:00 | 4541.00 | 4578.01 | 4582.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 4541.00 | 4578.01 | 4582.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 4435.10 | 4532.57 | 4557.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 13:15:00 | 4518.90 | 4518.44 | 4541.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 13:45:00 | 4520.50 | 4518.44 | 4541.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 4555.20 | 4525.79 | 4542.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 4555.20 | 4525.79 | 4542.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 4567.00 | 4534.03 | 4544.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 4545.90 | 4534.03 | 4544.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 4555.10 | 4540.93 | 4546.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 10:30:00 | 4563.80 | 4540.93 | 4546.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 4555.10 | 4543.76 | 4546.99 | EMA400 retest candle locked (from downside) |

### Cycle 45 — BUY (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 14:15:00 | 4580.00 | 4554.29 | 4551.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 15:15:00 | 4600.00 | 4563.43 | 4555.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-11 09:15:00 | 4556.70 | 4562.09 | 4555.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 09:15:00 | 4556.70 | 4562.09 | 4555.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 09:15:00 | 4556.70 | 4562.09 | 4555.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:00:00 | 4556.70 | 4562.09 | 4555.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 10:15:00 | 4576.60 | 4564.99 | 4557.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-11 10:30:00 | 4558.00 | 4564.99 | 4557.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 4648.20 | 4659.28 | 4632.28 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 4581.40 | 4618.47 | 4622.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 4566.50 | 4599.52 | 4611.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 15:15:00 | 4610.00 | 4599.56 | 4608.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 15:15:00 | 4610.00 | 4599.56 | 4608.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 4610.00 | 4599.56 | 4608.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:30:00 | 4633.00 | 4597.43 | 4606.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 4545.70 | 4524.48 | 4549.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:45:00 | 4544.40 | 4524.48 | 4549.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 4555.20 | 4530.62 | 4550.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 4555.20 | 4530.62 | 4550.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 4562.70 | 4537.04 | 4551.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 4601.20 | 4537.04 | 4551.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 4575.50 | 4544.73 | 4553.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 4550.70 | 4548.40 | 4554.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 10:15:00 | 4615.00 | 4561.54 | 4555.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 4615.00 | 4561.54 | 4555.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 13:15:00 | 4640.60 | 4592.35 | 4572.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 10:15:00 | 4653.00 | 4664.35 | 4634.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 11:00:00 | 4653.00 | 4664.35 | 4634.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 13:15:00 | 4634.60 | 4657.76 | 4638.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 14:00:00 | 4634.60 | 4657.76 | 4638.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 4607.00 | 4647.61 | 4635.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 4607.00 | 4647.61 | 4635.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 4601.00 | 4638.29 | 4632.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:15:00 | 4611.00 | 4638.29 | 4632.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 10:15:00 | 4570.10 | 4623.32 | 4626.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 11:15:00 | 4557.10 | 4610.08 | 4620.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 14:15:00 | 4550.00 | 4521.61 | 4553.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 14:15:00 | 4550.00 | 4521.61 | 4553.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 4550.00 | 4521.61 | 4553.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 4550.00 | 4521.61 | 4553.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 4523.50 | 4521.99 | 4550.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 4501.10 | 4521.99 | 4550.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:45:00 | 4499.90 | 4520.47 | 4547.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 10:30:00 | 4496.50 | 4515.64 | 4542.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 14:30:00 | 4486.50 | 4478.62 | 4496.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 4442.70 | 4470.26 | 4489.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 12:00:00 | 4434.00 | 4459.77 | 4481.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 4388.70 | 4374.44 | 4373.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 4388.70 | 4374.44 | 4373.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 4388.70 | 4374.44 | 4373.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 4388.70 | 4374.44 | 4373.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-08 09:15:00 | 4388.70 | 4374.44 | 4373.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 09:15:00 | 4388.70 | 4374.44 | 4373.83 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 4357.30 | 4371.01 | 4372.33 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 12:15:00 | 4377.80 | 4373.49 | 4373.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-08 13:15:00 | 4426.60 | 4384.11 | 4378.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-09 11:15:00 | 4375.50 | 4396.59 | 4388.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 11:15:00 | 4375.50 | 4396.59 | 4388.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 4375.50 | 4396.59 | 4388.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:00:00 | 4375.50 | 4396.59 | 4388.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 4368.90 | 4391.05 | 4386.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:45:00 | 4368.20 | 4391.05 | 4386.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 14:15:00 | 4352.90 | 4379.06 | 4381.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 4313.10 | 4367.62 | 4376.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 4318.80 | 4313.21 | 4341.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 4318.90 | 4313.21 | 4341.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 4312.10 | 4313.52 | 4336.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:45:00 | 4295.40 | 4307.07 | 4326.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:00:00 | 4287.90 | 4306.90 | 4321.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 11:00:00 | 4287.50 | 4303.02 | 4318.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 11:45:00 | 4291.20 | 4300.29 | 4315.85 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 4080.63 | 4235.21 | 4277.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 4073.50 | 4235.21 | 4277.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 4073.12 | 4235.21 | 4277.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 4076.64 | 4235.21 | 4277.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-01-19 09:15:00 | 3865.86 | 4014.29 | 4129.12 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-19 09:15:00 | 3859.11 | 4014.29 | 4129.12 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-19 09:15:00 | 3858.75 | 4014.29 | 4129.12 | Target hit (10%) qty=0.50 alert=retest2 |
| Target hit | 2026-01-19 09:15:00 | 3862.08 | 4014.29 | 4129.12 | Target hit (10%) qty=0.50 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 3876.00 | 3833.31 | 3882.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:30:00 | 3868.40 | 3861.72 | 3883.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 09:15:00 | 3860.80 | 3865.40 | 3876.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 11:15:00 | 3674.98 | 3724.09 | 3744.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 11:15:00 | 3667.76 | 3724.09 | 3744.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-02 15:15:00 | 3708.90 | 3701.21 | 3725.12 | SL hit (close>ema200) qty=0.50 sl=3701.21 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-02 15:15:00 | 3708.90 | 3701.21 | 3725.12 | SL hit (close>ema200) qty=0.50 sl=3701.21 alert=retest2 |

### Cycle 53 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 3815.40 | 3743.01 | 3741.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 12:15:00 | 3850.60 | 3814.30 | 3800.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 3833.10 | 3885.13 | 3872.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 3833.10 | 3885.13 | 3872.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 3833.10 | 3885.13 | 3872.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 3833.10 | 3885.13 | 3872.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 3837.60 | 3875.63 | 3869.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:45:00 | 3836.80 | 3875.63 | 3869.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 3839.10 | 3862.38 | 3863.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 3724.00 | 3823.28 | 3844.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 11:15:00 | 3585.50 | 3553.91 | 3627.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 12:00:00 | 3585.50 | 3553.91 | 3627.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 3440.70 | 3400.76 | 3424.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:00:00 | 3440.70 | 3400.76 | 3424.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 3449.00 | 3410.41 | 3426.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 3311.40 | 3410.41 | 3426.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 13:15:00 | 3145.83 | 3283.33 | 3351.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 3264.40 | 3240.66 | 3311.52 | SL hit (close>ema200) qty=0.50 sl=3240.66 alert=retest2 |

### Cycle 55 — BUY (started 2026-02-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 15:15:00 | 3325.50 | 3298.31 | 3297.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 09:15:00 | 3361.00 | 3310.85 | 3303.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-02 11:15:00 | 3446.90 | 3447.43 | 3395.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-02 12:00:00 | 3446.90 | 3447.43 | 3395.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 09:15:00 | 3386.40 | 3432.38 | 3407.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 09:45:00 | 3353.20 | 3432.38 | 3407.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 10:15:00 | 3368.80 | 3419.66 | 3404.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-04 11:00:00 | 3368.80 | 3419.66 | 3404.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 12:15:00 | 3333.10 | 3389.49 | 3392.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 14:15:00 | 3313.90 | 3367.96 | 3381.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 13:15:00 | 3136.90 | 3124.99 | 3165.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-10 13:45:00 | 3140.10 | 3124.99 | 3165.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 3141.60 | 3136.78 | 3161.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:15:00 | 3137.70 | 3136.78 | 3161.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 12:00:00 | 3137.40 | 3135.71 | 3157.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-12 14:45:00 | 3138.90 | 3127.74 | 3135.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 10:15:00 | 3365.00 | 3181.79 | 3158.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 10:15:00 | 3365.00 | 3181.79 | 3158.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-13 10:15:00 | 3365.00 | 3181.79 | 3158.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 10:15:00 | 3365.00 | 3181.79 | 3158.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-13 14:15:00 | 3441.40 | 3293.21 | 3223.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 09:15:00 | 3274.70 | 3317.79 | 3248.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 3274.70 | 3317.79 | 3248.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 3274.70 | 3317.79 | 3248.90 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 09:15:00 | 3272.00 | 3322.83 | 3329.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 10:15:00 | 3265.60 | 3311.38 | 3323.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 3162.30 | 3122.49 | 3174.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:00:00 | 3162.30 | 3122.49 | 3174.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 3185.30 | 3146.66 | 3170.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 12:45:00 | 3150.30 | 3154.35 | 3168.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 09:15:00 | 3196.80 | 3178.58 | 3177.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-27 09:15:00 | 3196.80 | 3178.58 | 3177.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 11:15:00 | 3231.50 | 3193.07 | 3184.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 14:15:00 | 3196.90 | 3198.27 | 3189.21 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 14:15:00 | 3196.90 | 3198.27 | 3189.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 3196.90 | 3198.27 | 3189.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 3196.90 | 3198.27 | 3189.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 3200.00 | 3198.62 | 3190.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 3143.90 | 3198.62 | 3190.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 3123.70 | 3183.63 | 3184.15 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 3326.40 | 3177.55 | 3174.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 3361.30 | 3309.72 | 3269.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 14:15:00 | 3346.70 | 3346.98 | 3312.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-06 14:30:00 | 3344.60 | 3346.98 | 3312.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 3374.10 | 3352.57 | 3320.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 3418.10 | 3347.07 | 3332.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 10:30:00 | 3393.80 | 3367.00 | 3344.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 14:15:00 | 3332.90 | 3345.79 | 3347.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-09 14:15:00 | 3332.90 | 3345.79 | 3347.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2026-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 14:15:00 | 3332.90 | 3345.79 | 3347.04 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 09:15:00 | 3410.80 | 3356.91 | 3351.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 13:15:00 | 3455.80 | 3406.46 | 3379.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 3418.40 | 3418.73 | 3392.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 3418.40 | 3418.73 | 3392.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 3418.40 | 3418.73 | 3392.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 3430.40 | 3389.41 | 3388.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 11:00:00 | 3433.10 | 3406.15 | 3396.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 15:15:00 | 3441.70 | 3433.09 | 3414.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 3490.00 | 3552.05 | 3552.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 3490.00 | 3552.05 | 3552.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 3490.00 | 3552.05 | 3552.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 3490.00 | 3552.05 | 3552.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 12:15:00 | 3440.00 | 3504.06 | 3528.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 3482.30 | 3420.62 | 3453.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 3482.30 | 3420.62 | 3453.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 3482.30 | 3420.62 | 3453.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 3482.30 | 3420.62 | 3453.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 3513.70 | 3439.24 | 3459.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:00:00 | 3513.70 | 3439.24 | 3459.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 3562.00 | 3477.06 | 3473.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 13:15:00 | 3572.90 | 3496.23 | 3482.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 12:15:00 | 3558.00 | 3566.84 | 3546.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 13:00:00 | 3558.00 | 3566.84 | 3546.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 3556.00 | 3564.91 | 3551.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 3549.00 | 3564.91 | 3551.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 3522.00 | 3556.33 | 3548.46 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 3483.20 | 3541.70 | 3542.53 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 3595.90 | 3544.81 | 3542.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 14:15:00 | 3627.50 | 3561.35 | 3550.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 10:15:00 | 3777.20 | 3777.24 | 3738.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 10:45:00 | 3783.00 | 3777.24 | 3738.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-29 12:30:00 | 4415.10 | 2025-06-09 15:15:00 | 4334.50 | STOP_HIT | 1.00 | 1.83% |
| SELL | retest2 | 2025-05-29 14:00:00 | 4416.10 | 2025-06-09 15:15:00 | 4334.50 | STOP_HIT | 1.00 | 1.85% |
| BUY | retest2 | 2025-06-13 10:15:00 | 4473.20 | 2025-06-13 12:15:00 | 4428.10 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-06-23 09:45:00 | 4341.10 | 2025-06-25 15:15:00 | 4365.00 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-06-23 11:00:00 | 4337.30 | 2025-06-25 15:15:00 | 4365.00 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-06-23 11:30:00 | 4341.80 | 2025-06-25 15:15:00 | 4365.00 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-06-23 12:45:00 | 4336.00 | 2025-06-25 15:15:00 | 4365.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-06-24 13:00:00 | 4355.00 | 2025-06-25 15:15:00 | 4365.00 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-06-25 12:15:00 | 4350.00 | 2025-06-25 15:15:00 | 4365.00 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-06-25 13:00:00 | 4354.70 | 2025-06-25 15:15:00 | 4365.00 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-06-25 13:30:00 | 4353.90 | 2025-06-25 15:15:00 | 4365.00 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-07-17 09:15:00 | 4379.40 | 2025-07-21 10:15:00 | 4349.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-07-21 10:15:00 | 4369.70 | 2025-07-21 10:15:00 | 4349.00 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-07-31 11:30:00 | 4300.00 | 2025-08-04 09:15:00 | 4260.70 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-08-01 10:00:00 | 4298.80 | 2025-08-04 09:15:00 | 4260.70 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-08-14 13:00:00 | 4153.80 | 2025-08-19 10:15:00 | 4195.10 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-08-14 14:30:00 | 4153.10 | 2025-08-19 10:15:00 | 4195.10 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-08-18 12:00:00 | 4158.20 | 2025-08-19 10:15:00 | 4195.10 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-08-18 12:30:00 | 4155.00 | 2025-08-19 10:15:00 | 4195.10 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-08-18 14:45:00 | 4161.50 | 2025-08-19 10:15:00 | 4195.10 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-08-19 09:30:00 | 4161.00 | 2025-08-19 10:15:00 | 4195.10 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-08-22 12:45:00 | 4267.90 | 2025-08-28 13:15:00 | 4298.50 | STOP_HIT | 1.00 | 0.72% |
| SELL | retest2 | 2025-09-02 11:15:00 | 4216.00 | 2025-09-10 09:15:00 | 4234.60 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-09-02 13:15:00 | 4212.00 | 2025-09-10 09:15:00 | 4234.60 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-09-19 15:15:00 | 4360.00 | 2025-09-22 09:15:00 | 4249.00 | STOP_HIT | 1.00 | -2.55% |
| SELL | retest2 | 2025-09-29 13:30:00 | 4031.00 | 2025-09-30 15:15:00 | 4097.00 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-10-31 10:30:00 | 4095.10 | 2025-11-03 15:15:00 | 4117.00 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-11-03 09:15:00 | 4092.70 | 2025-11-03 15:15:00 | 4117.00 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-11-07 09:15:00 | 4057.60 | 2025-11-10 14:15:00 | 4084.50 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-11-07 15:15:00 | 4047.00 | 2025-11-10 14:15:00 | 4084.50 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-11-10 14:15:00 | 4085.10 | 2025-11-10 14:15:00 | 4084.50 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-11-18 10:45:00 | 4087.90 | 2025-11-19 09:15:00 | 4154.20 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-11-18 11:30:00 | 4087.50 | 2025-11-19 09:15:00 | 4154.20 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-11-18 12:00:00 | 4087.40 | 2025-11-19 09:15:00 | 4154.20 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-11-26 11:00:00 | 4242.40 | 2025-11-27 09:15:00 | 4309.30 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-11-26 11:30:00 | 4249.60 | 2025-11-27 09:15:00 | 4309.30 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-12-04 09:15:00 | 4560.00 | 2025-12-08 11:15:00 | 4541.00 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-12-08 09:45:00 | 4536.50 | 2025-12-08 11:15:00 | 4541.00 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2025-12-08 11:00:00 | 4542.30 | 2025-12-08 11:15:00 | 4541.00 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-12-19 11:15:00 | 4550.70 | 2025-12-22 10:15:00 | 4615.00 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2025-12-30 09:15:00 | 4501.10 | 2026-01-08 09:15:00 | 4388.70 | STOP_HIT | 1.00 | 2.50% |
| SELL | retest2 | 2025-12-30 09:45:00 | 4499.90 | 2026-01-08 09:15:00 | 4388.70 | STOP_HIT | 1.00 | 2.47% |
| SELL | retest2 | 2025-12-30 10:30:00 | 4496.50 | 2026-01-08 09:15:00 | 4388.70 | STOP_HIT | 1.00 | 2.40% |
| SELL | retest2 | 2025-12-31 14:30:00 | 4486.50 | 2026-01-08 09:15:00 | 4388.70 | STOP_HIT | 1.00 | 2.18% |
| SELL | retest2 | 2026-01-01 12:00:00 | 4434.00 | 2026-01-08 09:15:00 | 4388.70 | STOP_HIT | 1.00 | 1.02% |
| SELL | retest2 | 2026-01-13 13:45:00 | 4295.40 | 2026-01-16 09:15:00 | 4080.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 10:00:00 | 4287.90 | 2026-01-16 09:15:00 | 4073.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 11:00:00 | 4287.50 | 2026-01-16 09:15:00 | 4073.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 11:45:00 | 4291.20 | 2026-01-16 09:15:00 | 4076.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 13:45:00 | 4295.40 | 2026-01-19 09:15:00 | 3865.86 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 10:00:00 | 4287.90 | 2026-01-19 09:15:00 | 3859.11 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 11:00:00 | 4287.50 | 2026-01-19 09:15:00 | 3858.75 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-14 11:45:00 | 4291.20 | 2026-01-19 09:15:00 | 3862.08 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-01-22 10:30:00 | 3868.40 | 2026-02-02 11:15:00 | 3674.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 09:15:00 | 3860.80 | 2026-02-02 11:15:00 | 3667.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-22 10:30:00 | 3868.40 | 2026-02-02 15:15:00 | 3708.90 | STOP_HIT | 0.50 | 4.12% |
| SELL | retest2 | 2026-01-23 09:15:00 | 3860.80 | 2026-02-02 15:15:00 | 3708.90 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest2 | 2026-02-24 09:15:00 | 3311.40 | 2026-02-24 13:15:00 | 3145.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 3311.40 | 2026-02-25 09:15:00 | 3264.40 | STOP_HIT | 0.50 | 1.42% |
| SELL | retest2 | 2026-03-11 10:15:00 | 3137.70 | 2026-03-13 10:15:00 | 3365.00 | STOP_HIT | 1.00 | -7.24% |
| SELL | retest2 | 2026-03-11 12:00:00 | 3137.40 | 2026-03-13 10:15:00 | 3365.00 | STOP_HIT | 1.00 | -7.25% |
| SELL | retest2 | 2026-03-12 14:45:00 | 3138.90 | 2026-03-13 10:15:00 | 3365.00 | STOP_HIT | 1.00 | -7.20% |
| SELL | retest2 | 2026-03-25 12:45:00 | 3150.30 | 2026-03-27 09:15:00 | 3196.80 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2026-04-08 09:15:00 | 3418.10 | 2026-04-09 14:15:00 | 3332.90 | STOP_HIT | 1.00 | -2.49% |
| BUY | retest2 | 2026-04-08 10:30:00 | 3393.80 | 2026-04-09 14:15:00 | 3332.90 | STOP_HIT | 1.00 | -1.79% |
| BUY | retest2 | 2026-04-15 09:15:00 | 3430.40 | 2026-04-23 09:15:00 | 3490.00 | STOP_HIT | 1.00 | 1.74% |
| BUY | retest2 | 2026-04-15 11:00:00 | 3433.10 | 2026-04-23 09:15:00 | 3490.00 | STOP_HIT | 1.00 | 1.66% |
| BUY | retest2 | 2026-04-15 15:15:00 | 3441.70 | 2026-04-23 09:15:00 | 3490.00 | STOP_HIT | 1.00 | 1.40% |
