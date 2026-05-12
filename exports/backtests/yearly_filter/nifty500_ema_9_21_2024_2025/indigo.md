# InterGlobe Aviation Ltd. (INDIGO)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 4522.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 143 |
| ALERT1 | 107 |
| ALERT2 | 103 |
| ALERT2_SKIP | 50 |
| ALERT3 | 228 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 106 |
| PARTIAL | 6 |
| TARGET_HIT | 3 |
| STOP_HIT | 105 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 114 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 33 / 81
- **Target hits / Stop hits / Partials:** 3 / 105 / 6
- **Avg / median % per leg:** -0.01% / -0.75%
- **Sum % (uncompounded):** -1.09%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 64 | 21 | 32.8% | 2 | 61 | 1 | -0.03% | -2.1% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 3.37% | 10.1% |
| BUY @ 3rd Alert (retest2) | 61 | 19 | 31.1% | 2 | 59 | 0 | -0.20% | -12.2% |
| SELL (all) | 50 | 12 | 24.0% | 1 | 44 | 5 | 0.02% | 1.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 50 | 12 | 24.0% | 1 | 44 | 5 | 0.02% | 1.0% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 3.37% | 10.1% |
| retest2 (combined) | 111 | 31 | 27.9% | 3 | 103 | 5 | -0.10% | -11.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 11:15:00 | 4054.90 | 4030.80 | 4029.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 12:15:00 | 4061.90 | 4037.02 | 4032.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 4338.00 | 4351.33 | 4313.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 4338.00 | 4351.33 | 4313.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 4338.00 | 4351.33 | 4313.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 12:15:00 | 4365.45 | 4325.41 | 4317.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 13:15:00 | 4371.50 | 4332.72 | 4321.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 14:00:00 | 4367.35 | 4339.65 | 4325.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 09:15:00 | 4410.20 | 4345.24 | 4330.82 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 4348.50 | 4362.57 | 4343.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 12:00:00 | 4348.50 | 4362.57 | 4343.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 12:15:00 | 4342.00 | 4358.45 | 4343.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 13:00:00 | 4342.00 | 4358.45 | 4343.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 13:15:00 | 4403.85 | 4367.53 | 4349.18 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-24 11:15:00 | 4274.10 | 4340.21 | 4344.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 11:15:00 | 4274.10 | 4340.21 | 4344.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 14:15:00 | 4260.00 | 4303.78 | 4324.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-27 13:15:00 | 4274.60 | 4273.05 | 4296.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-27 14:15:00 | 4280.80 | 4273.05 | 4296.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 14:15:00 | 4258.20 | 4270.08 | 4293.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 10:00:00 | 4231.95 | 4259.24 | 4284.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-05-29 14:15:00 | 4020.35 | 4094.08 | 4164.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-05-30 09:15:00 | 4092.35 | 4087.00 | 4148.33 | SL hit (close>ema200) qty=0.50 sl=4087.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-05-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 15:15:00 | 4200.00 | 4159.60 | 4156.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 4279.70 | 4183.62 | 4167.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 4266.25 | 4269.14 | 4228.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 4266.25 | 4269.14 | 4228.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 4266.25 | 4269.14 | 4228.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 4203.10 | 4269.14 | 4228.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 4063.95 | 4228.10 | 4213.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 4063.95 | 4228.10 | 4213.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 3882.05 | 4158.89 | 4183.69 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 11:15:00 | 4299.00 | 4185.39 | 4178.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 12:15:00 | 4302.20 | 4208.75 | 4189.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 09:15:00 | 4404.00 | 4495.80 | 4431.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-11 09:15:00 | 4404.00 | 4495.80 | 4431.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 4404.00 | 4495.80 | 4431.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 09:30:00 | 4412.85 | 4495.80 | 4431.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 10:15:00 | 4389.00 | 4474.44 | 4427.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 11:00:00 | 4389.00 | 4474.44 | 4427.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 09:15:00 | 4348.90 | 4398.13 | 4404.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-12 10:15:00 | 4325.35 | 4383.57 | 4397.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-13 14:15:00 | 4300.30 | 4287.16 | 4321.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-13 14:30:00 | 4305.60 | 4287.16 | 4321.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 4288.00 | 4289.38 | 4316.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 10:15:00 | 4232.25 | 4291.78 | 4298.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 11:30:00 | 4249.00 | 4245.25 | 4249.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 13:15:00 | 4312.65 | 4258.21 | 4254.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 13:15:00 | 4312.65 | 4258.21 | 4254.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 09:15:00 | 4321.75 | 4283.63 | 4268.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 09:15:00 | 4306.65 | 4308.61 | 4292.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-25 09:45:00 | 4311.05 | 4308.61 | 4292.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 4250.20 | 4296.93 | 4288.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 10:45:00 | 4272.10 | 4296.93 | 4288.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 4268.00 | 4291.15 | 4286.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-25 12:15:00 | 4274.65 | 4291.15 | 4286.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 13:15:00 | 4263.60 | 4281.79 | 4282.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 13:15:00 | 4263.60 | 4281.79 | 4282.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 14:15:00 | 4228.85 | 4271.20 | 4277.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 14:15:00 | 4225.00 | 4209.61 | 4229.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-27 15:00:00 | 4225.00 | 4209.61 | 4229.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 4231.80 | 4214.05 | 4230.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:15:00 | 4242.00 | 4214.05 | 4230.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 4236.00 | 4218.44 | 4230.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:30:00 | 4244.95 | 4218.44 | 4230.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 10:15:00 | 4249.60 | 4224.67 | 4232.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 11:00:00 | 4249.60 | 4224.67 | 4232.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 4252.15 | 4230.17 | 4234.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 11:30:00 | 4258.35 | 4230.17 | 4234.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 4235.00 | 4233.88 | 4235.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 09:15:00 | 4250.55 | 4233.88 | 4235.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-07-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 09:15:00 | 4254.55 | 4238.02 | 4236.91 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 12:15:00 | 4228.25 | 4235.35 | 4235.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-01 14:15:00 | 4219.80 | 4230.63 | 4233.60 | Break + close below crossover candle low |

### Cycle 11 — BUY (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 09:15:00 | 4263.30 | 4236.74 | 4235.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 4298.05 | 4272.78 | 4259.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 14:15:00 | 4286.15 | 4288.09 | 4273.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 15:00:00 | 4286.15 | 4288.09 | 4273.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 4282.70 | 4312.07 | 4299.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 10:00:00 | 4282.70 | 4312.07 | 4299.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 10:15:00 | 4284.00 | 4306.46 | 4297.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 11:00:00 | 4284.00 | 4306.46 | 4297.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 11:15:00 | 4251.55 | 4295.48 | 4293.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 12:00:00 | 4251.55 | 4295.48 | 4293.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2024-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 12:15:00 | 4256.75 | 4287.73 | 4290.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 14:15:00 | 4233.90 | 4271.21 | 4281.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 10:15:00 | 4261.05 | 4259.96 | 4273.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-09 10:45:00 | 4263.75 | 4259.96 | 4273.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 4277.00 | 4263.37 | 4273.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 12:00:00 | 4277.00 | 4263.37 | 4273.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 12:15:00 | 4269.40 | 4264.57 | 4273.16 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2024-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 15:15:00 | 4297.00 | 4279.12 | 4278.22 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 4245.05 | 4272.30 | 4275.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 10:15:00 | 4207.00 | 4259.24 | 4269.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 14:15:00 | 4281.40 | 4256.67 | 4263.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 14:15:00 | 4281.40 | 4256.67 | 4263.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 4281.40 | 4256.67 | 4263.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 15:00:00 | 4281.40 | 4256.67 | 4263.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 4275.10 | 4260.36 | 4264.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 4302.80 | 4260.36 | 4264.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2024-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 09:15:00 | 4304.75 | 4269.24 | 4268.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-11 14:15:00 | 4319.00 | 4296.98 | 4284.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 4400.85 | 4416.07 | 4391.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 10:15:00 | 4400.85 | 4416.07 | 4391.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 4400.85 | 4416.07 | 4391.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 4400.85 | 4416.07 | 4391.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 4335.25 | 4399.90 | 4395.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 10:00:00 | 4335.25 | 4399.90 | 4395.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 4298.35 | 4379.59 | 4386.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 11:15:00 | 4287.50 | 4361.17 | 4377.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 4323.15 | 4314.88 | 4344.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 09:30:00 | 4310.25 | 4314.88 | 4344.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 4341.95 | 4320.29 | 4344.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 4347.85 | 4320.29 | 4344.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 4338.00 | 4323.83 | 4343.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:45:00 | 4328.35 | 4323.83 | 4343.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 4343.05 | 4327.68 | 4343.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:00:00 | 4343.05 | 4327.68 | 4343.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 4345.00 | 4331.14 | 4343.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:15:00 | 4349.20 | 4331.14 | 4343.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 4335.20 | 4331.95 | 4342.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:30:00 | 4341.05 | 4331.95 | 4342.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 4304.55 | 4326.96 | 4338.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 11:30:00 | 4302.00 | 4322.78 | 4334.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 4206.80 | 4322.78 | 4334.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:00:00 | 4297.45 | 4317.71 | 4331.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 12:15:00 | 4344.35 | 4333.93 | 4333.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 12:15:00 | 4344.35 | 4333.93 | 4333.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 14:15:00 | 4372.00 | 4344.09 | 4338.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 09:15:00 | 4387.40 | 4457.63 | 4431.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 09:15:00 | 4387.40 | 4457.63 | 4431.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 4387.40 | 4457.63 | 4431.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 09:45:00 | 4498.60 | 4450.61 | 4435.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 11:15:00 | 4482.05 | 4452.91 | 4438.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 12:15:00 | 4481.25 | 4457.11 | 4441.60 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 14:15:00 | 4485.75 | 4464.61 | 4447.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 4465.50 | 4469.48 | 4455.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 10:45:00 | 4461.10 | 4469.48 | 4455.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 11:15:00 | 4459.05 | 4467.39 | 4456.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 11:45:00 | 4460.30 | 4467.39 | 4456.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 12:15:00 | 4488.30 | 4471.58 | 4459.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 13:45:00 | 4491.00 | 4475.26 | 4461.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-01 09:15:00 | 4432.75 | 4466.10 | 4460.94 | SL hit (close<static) qty=1.00 sl=4458.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 10:15:00 | 4409.00 | 4454.68 | 4456.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 13:15:00 | 4397.80 | 4428.82 | 4442.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 4310.45 | 4265.88 | 4313.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 4310.45 | 4265.88 | 4313.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 4310.45 | 4265.88 | 4313.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:30:00 | 4270.10 | 4283.68 | 4307.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 11:15:00 | 4323.90 | 4307.29 | 4306.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 11:15:00 | 4323.90 | 4307.29 | 4306.22 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 14:15:00 | 4247.00 | 4296.22 | 4301.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 4233.15 | 4259.17 | 4270.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 4230.00 | 4228.14 | 4243.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 4230.00 | 4228.14 | 4243.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 4230.00 | 4228.14 | 4243.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 10:30:00 | 4220.35 | 4223.15 | 4239.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 11:15:00 | 4266.35 | 4231.79 | 4242.29 | SL hit (close>static) qty=1.00 sl=4258.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 14:15:00 | 4284.95 | 4254.16 | 4250.97 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 11:15:00 | 4227.60 | 4248.70 | 4249.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-19 15:15:00 | 4221.25 | 4240.61 | 4245.62 | Break + close below crossover candle low |

### Cycle 23 — BUY (started 2024-08-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-20 09:15:00 | 4312.70 | 4255.03 | 4251.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 09:15:00 | 4449.00 | 4327.23 | 4299.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-26 09:15:00 | 4616.00 | 4631.54 | 4530.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-26 09:30:00 | 4612.95 | 4631.54 | 4530.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 4757.90 | 4815.86 | 4755.06 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 10:15:00 | 4778.90 | 4809.96 | 4811.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 09:15:00 | 4764.50 | 4786.88 | 4797.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 14:15:00 | 4804.80 | 4775.54 | 4786.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 14:15:00 | 4804.80 | 4775.54 | 4786.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 14:15:00 | 4804.80 | 4775.54 | 4786.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 15:00:00 | 4804.80 | 4775.54 | 4786.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 15:15:00 | 4805.50 | 4781.53 | 4788.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:15:00 | 4797.05 | 4781.53 | 4788.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 11:15:00 | 4797.35 | 4783.62 | 4787.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 12:00:00 | 4797.35 | 4783.62 | 4787.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 12:15:00 | 4820.00 | 4790.90 | 4790.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 13:15:00 | 4834.00 | 4799.52 | 4794.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 11:15:00 | 4963.85 | 4966.22 | 4928.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 11:30:00 | 4959.55 | 4966.22 | 4928.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 14:15:00 | 4940.15 | 4954.55 | 4932.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 09:30:00 | 4964.05 | 4958.84 | 4937.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 15:15:00 | 4953.70 | 4957.05 | 4945.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 09:15:00 | 4927.00 | 4950.50 | 4944.26 | SL hit (close<static) qty=1.00 sl=4929.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 12:15:00 | 4937.10 | 4939.89 | 4940.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 12:15:00 | 4906.90 | 4931.29 | 4935.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 14:15:00 | 4929.60 | 4928.25 | 4933.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-18 15:00:00 | 4929.60 | 4928.25 | 4933.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 4928.65 | 4928.37 | 4932.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:45:00 | 4938.50 | 4928.37 | 4932.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 10:15:00 | 4859.35 | 4914.57 | 4925.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 13:00:00 | 4846.45 | 4891.43 | 4912.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-19 13:45:00 | 4836.35 | 4879.61 | 4905.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 09:15:00 | 4806.90 | 4877.37 | 4899.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 09:15:00 | 4932.55 | 4880.17 | 4883.88 | SL hit (close>static) qty=1.00 sl=4931.40 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 11:15:00 | 4913.65 | 4891.72 | 4888.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 12:15:00 | 4933.10 | 4900.00 | 4892.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 09:15:00 | 4903.70 | 4911.70 | 4901.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 09:15:00 | 4903.70 | 4911.70 | 4901.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 4903.70 | 4911.70 | 4901.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 10:15:00 | 4878.95 | 4911.70 | 4901.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2024-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 10:15:00 | 4809.00 | 4891.16 | 4893.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 11:15:00 | 4765.00 | 4814.19 | 4844.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 09:15:00 | 4794.20 | 4792.97 | 4820.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 10:15:00 | 4812.60 | 4796.90 | 4819.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 4812.60 | 4796.90 | 4819.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 11:00:00 | 4812.60 | 4796.90 | 4819.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 4868.25 | 4811.17 | 4824.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 12:00:00 | 4868.25 | 4811.17 | 4824.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 12:15:00 | 4835.30 | 4815.99 | 4825.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-26 13:45:00 | 4831.20 | 4819.37 | 4825.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 15:15:00 | 4865.00 | 4835.82 | 4832.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2024-09-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 15:15:00 | 4865.00 | 4835.82 | 4832.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-27 09:15:00 | 4917.45 | 4852.15 | 4840.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 10:15:00 | 4910.90 | 4915.24 | 4887.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-30 10:30:00 | 4907.75 | 4915.24 | 4887.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 11:15:00 | 4885.20 | 4909.23 | 4887.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 11:45:00 | 4886.90 | 4909.23 | 4887.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 12:15:00 | 4879.45 | 4903.27 | 4886.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 13:00:00 | 4879.45 | 4903.27 | 4886.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 4831.25 | 4888.87 | 4881.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 14:00:00 | 4831.25 | 4888.87 | 4881.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 14:15:00 | 4788.30 | 4868.76 | 4872.91 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2024-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 12:15:00 | 4904.95 | 4871.98 | 4870.61 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 10:15:00 | 4791.50 | 4857.02 | 4865.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 11:15:00 | 4712.55 | 4828.13 | 4851.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 4547.30 | 4526.81 | 4601.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 11:00:00 | 4547.30 | 4526.81 | 4601.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 4603.70 | 4554.23 | 4591.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 4603.70 | 4554.23 | 4591.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 4609.40 | 4565.27 | 4593.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 4706.00 | 4565.27 | 4593.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 4732.10 | 4629.40 | 4619.66 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-14 10:15:00 | 4654.05 | 4669.32 | 4670.77 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 09:15:00 | 4731.50 | 4681.38 | 4674.78 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-10-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 10:15:00 | 4659.25 | 4699.96 | 4704.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 11:15:00 | 4645.50 | 4689.07 | 4699.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 10:15:00 | 4667.05 | 4654.05 | 4673.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 11:00:00 | 4667.05 | 4654.05 | 4673.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 11:15:00 | 4682.80 | 4659.80 | 4674.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 12:00:00 | 4682.80 | 4659.80 | 4674.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 12:15:00 | 4672.00 | 4662.24 | 4674.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 14:00:00 | 4662.50 | 4662.29 | 4673.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 09:15:00 | 4429.38 | 4503.30 | 4533.85 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-28 09:15:00 | 4196.25 | 4326.18 | 4421.72 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 37 — BUY (started 2024-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 12:15:00 | 4039.25 | 3989.17 | 3986.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 14:15:00 | 4060.20 | 4010.87 | 3997.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 4002.45 | 4020.09 | 4004.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 4002.45 | 4020.09 | 4004.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 4002.45 | 4020.09 | 4004.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 4002.45 | 4020.09 | 4004.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 3997.15 | 4015.50 | 4004.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 11:00:00 | 3997.15 | 4015.50 | 4004.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 4010.00 | 4014.40 | 4004.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 12:15:00 | 4013.20 | 4014.40 | 4004.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 09:15:00 | 4035.90 | 4006.96 | 4003.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 11:30:00 | 4021.05 | 4015.89 | 4009.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-08 14:30:00 | 4020.30 | 4015.15 | 4010.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 3990.35 | 4010.19 | 4008.95 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-11-08 15:15:00 | 3990.35 | 4010.19 | 4008.95 | SL hit (close<static) qty=1.00 sl=3992.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 09:15:00 | 3954.60 | 4002.90 | 4007.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 11:15:00 | 3937.15 | 3981.80 | 3996.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 3890.55 | 3880.29 | 3914.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-14 10:00:00 | 3890.55 | 3880.29 | 3914.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 3895.50 | 3883.33 | 3912.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 3910.55 | 3883.33 | 3912.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 12:15:00 | 3910.00 | 3890.45 | 3910.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 13:00:00 | 3910.00 | 3890.45 | 3910.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 13:15:00 | 3903.60 | 3893.08 | 3910.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 15:15:00 | 3889.90 | 3893.82 | 3909.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-18 09:15:00 | 3949.20 | 3904.27 | 3911.12 | SL hit (close>static) qty=1.00 sl=3919.00 alert=retest2 |

### Cycle 39 — BUY (started 2024-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 11:15:00 | 3991.35 | 3930.23 | 3922.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-18 12:15:00 | 4009.30 | 3946.05 | 3930.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 15:15:00 | 4040.00 | 4045.64 | 4005.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-21 09:15:00 | 4053.50 | 4045.64 | 4005.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 4032.40 | 4042.99 | 4008.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:45:00 | 4020.30 | 4042.99 | 4008.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 4229.00 | 4225.52 | 4194.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 11:15:00 | 4261.30 | 4228.89 | 4199.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 09:15:00 | 4325.40 | 4378.45 | 4383.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 09:15:00 | 4325.40 | 4378.45 | 4383.25 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-12-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 10:15:00 | 4420.00 | 4384.23 | 4380.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 14:15:00 | 4474.95 | 4415.05 | 4397.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 09:15:00 | 4469.50 | 4473.00 | 4446.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 13:15:00 | 4476.15 | 4473.73 | 4455.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 13:15:00 | 4476.15 | 4473.73 | 4455.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 13:45:00 | 4472.30 | 4473.73 | 4455.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 4483.25 | 4476.18 | 4461.31 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 4411.35 | 4461.28 | 4461.41 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-12 12:15:00 | 4475.15 | 4461.37 | 4461.07 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-13 09:15:00 | 4405.10 | 4450.87 | 4456.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 09:15:00 | 4398.40 | 4426.91 | 4439.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-17 09:15:00 | 4414.85 | 4409.53 | 4421.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-17 09:15:00 | 4414.85 | 4409.53 | 4421.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 4414.85 | 4409.53 | 4421.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 11:15:00 | 4381.40 | 4407.83 | 4419.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 14:30:00 | 4387.10 | 4401.79 | 4413.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 09:45:00 | 4387.10 | 4394.05 | 4407.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 12:30:00 | 4384.75 | 4392.29 | 4403.33 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 10:15:00 | 4398.75 | 4389.68 | 4397.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 11:00:00 | 4398.75 | 4389.68 | 4397.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 4422.00 | 4396.14 | 4399.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 11:30:00 | 4417.65 | 4396.14 | 4399.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 4421.85 | 4401.28 | 4401.81 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-19 13:15:00 | 4425.60 | 4406.15 | 4403.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 13:15:00 | 4425.60 | 4406.15 | 4403.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 14:15:00 | 4431.00 | 4411.12 | 4406.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 12:15:00 | 4436.00 | 4441.61 | 4425.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-20 13:00:00 | 4436.00 | 4441.61 | 4425.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 4417.40 | 4436.77 | 4424.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 14:00:00 | 4417.40 | 4436.77 | 4424.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 4405.25 | 4430.47 | 4423.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 4405.25 | 4430.47 | 4423.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 4390.75 | 4422.52 | 4420.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 09:15:00 | 4440.30 | 4422.52 | 4420.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-30 15:15:00 | 4613.05 | 4633.60 | 4634.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2024-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 15:15:00 | 4613.05 | 4633.60 | 4634.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 09:15:00 | 4572.05 | 4621.29 | 4629.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 10:15:00 | 4602.60 | 4578.25 | 4595.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-01 10:15:00 | 4602.60 | 4578.25 | 4595.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 10:15:00 | 4602.60 | 4578.25 | 4595.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 10:45:00 | 4609.10 | 4578.25 | 4595.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 11:15:00 | 4633.40 | 4589.28 | 4599.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 12:00:00 | 4633.40 | 4589.28 | 4599.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 12:15:00 | 4599.70 | 4591.37 | 4599.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-02 09:15:00 | 4570.95 | 4595.40 | 4599.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 10:15:00 | 4342.40 | 4435.63 | 4492.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-07 10:15:00 | 4327.00 | 4325.49 | 4395.09 | SL hit (close>ema200) qty=0.50 sl=4325.49 alert=retest2 |

### Cycle 47 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 4111.20 | 4088.07 | 4084.97 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-01-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 15:15:00 | 4086.15 | 4092.95 | 4093.08 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 09:15:00 | 4109.95 | 4096.35 | 4094.61 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 09:15:00 | 4058.00 | 4097.80 | 4097.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 12:15:00 | 4026.20 | 4070.09 | 4084.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 15:15:00 | 4016.50 | 4014.45 | 4037.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 09:15:00 | 4076.80 | 4014.45 | 4037.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 4112.70 | 4034.10 | 4044.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 4119.65 | 4034.10 | 4044.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 4156.45 | 4058.57 | 4054.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-24 09:15:00 | 4197.65 | 4130.72 | 4097.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-24 13:15:00 | 4147.05 | 4151.90 | 4119.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-24 13:45:00 | 4156.75 | 4151.90 | 4119.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 4200.00 | 4165.18 | 4134.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-27 11:00:00 | 4230.00 | 4178.14 | 4142.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-28 10:15:00 | 4227.10 | 4188.38 | 4163.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-28 12:00:00 | 4228.55 | 4198.46 | 4172.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-30 15:00:00 | 4232.00 | 4263.21 | 4253.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 4249.00 | 4254.74 | 4250.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 11:00:00 | 4273.55 | 4258.50 | 4252.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-04 12:15:00 | 4320.50 | 4354.76 | 4359.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-02-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 12:15:00 | 4320.50 | 4354.76 | 4359.16 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 4403.30 | 4359.52 | 4359.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 11:15:00 | 4423.45 | 4381.46 | 4369.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 10:15:00 | 4390.80 | 4401.90 | 4387.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 10:15:00 | 4390.80 | 4401.90 | 4387.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 4390.80 | 4401.90 | 4387.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:45:00 | 4400.00 | 4401.90 | 4387.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 4412.75 | 4404.07 | 4389.91 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2025-02-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 13:15:00 | 4359.10 | 4383.91 | 4386.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 15:15:00 | 4351.75 | 4375.25 | 4381.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 09:15:00 | 4334.40 | 4332.62 | 4350.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-11 10:15:00 | 4385.80 | 4332.62 | 4350.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 10:15:00 | 4363.75 | 4338.84 | 4351.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 12:45:00 | 4328.15 | 4338.15 | 4348.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 14:15:00 | 4322.00 | 4337.69 | 4347.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-13 11:15:00 | 4340.60 | 4325.98 | 4325.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 11:15:00 | 4340.60 | 4325.98 | 4325.89 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-02-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 10:15:00 | 4242.20 | 4319.58 | 4325.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 11:15:00 | 4233.70 | 4302.40 | 4317.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 12:15:00 | 4252.30 | 4246.01 | 4272.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 13:00:00 | 4252.30 | 4246.01 | 4272.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 13:15:00 | 4263.95 | 4249.60 | 4271.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 13:45:00 | 4253.75 | 4249.60 | 4271.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 14:15:00 | 4305.05 | 4260.69 | 4274.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-17 15:00:00 | 4305.05 | 4260.69 | 4274.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 15:15:00 | 4312.00 | 4270.95 | 4277.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 09:15:00 | 4276.00 | 4270.95 | 4277.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-18 12:15:00 | 4305.30 | 4283.67 | 4281.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 12:15:00 | 4305.30 | 4283.67 | 4281.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-18 13:15:00 | 4344.50 | 4295.83 | 4287.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 15:15:00 | 4492.50 | 4499.09 | 4459.36 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-24 09:15:00 | 4518.00 | 4499.09 | 4459.36 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 4487.95 | 4526.69 | 4501.63 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-25 09:15:00 | 4487.95 | 4526.69 | 4501.63 | SL hit (close<ema400) qty=1.00 sl=4501.63 alert=retest1 |

### Cycle 58 — SELL (started 2025-02-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 12:15:00 | 4430.10 | 4485.63 | 4487.43 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-02-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-28 15:15:00 | 4479.95 | 4467.47 | 4466.52 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-03-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-03 09:15:00 | 4319.15 | 4437.81 | 4453.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 10:15:00 | 4308.00 | 4411.85 | 4439.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 4438.00 | 4413.09 | 4435.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 12:15:00 | 4438.00 | 4413.09 | 4435.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 4438.00 | 4413.09 | 4435.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:00:00 | 4438.00 | 4413.09 | 4435.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 4430.25 | 4416.52 | 4434.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:45:00 | 4432.90 | 4416.52 | 4434.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 14:15:00 | 4470.00 | 4427.22 | 4438.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 15:00:00 | 4470.00 | 4427.22 | 4438.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 4457.65 | 4433.31 | 4439.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 09:15:00 | 4480.80 | 4433.31 | 4439.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 09:15:00 | 4544.90 | 4455.62 | 4449.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 10:15:00 | 4572.00 | 4478.90 | 4460.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 10:15:00 | 4718.85 | 4729.64 | 4678.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 11:00:00 | 4718.85 | 4729.64 | 4678.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 12:15:00 | 4664.10 | 4711.80 | 4679.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 12:45:00 | 4669.40 | 4711.80 | 4679.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 4686.75 | 4706.79 | 4679.89 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 4625.15 | 4662.57 | 4666.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 09:15:00 | 4586.00 | 4642.01 | 4656.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 13:15:00 | 4646.05 | 4629.83 | 4644.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 13:15:00 | 4646.05 | 4629.83 | 4644.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 13:15:00 | 4646.05 | 4629.83 | 4644.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 14:00:00 | 4646.05 | 4629.83 | 4644.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 14:15:00 | 4645.80 | 4633.02 | 4644.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 15:15:00 | 4668.00 | 4633.02 | 4644.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 4668.00 | 4640.02 | 4646.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 4700.10 | 4640.02 | 4646.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — BUY (started 2025-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 09:15:00 | 4701.20 | 4652.25 | 4651.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-17 09:15:00 | 4738.90 | 4707.10 | 4692.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-18 10:15:00 | 4755.40 | 4761.83 | 4736.22 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-03-18 14:15:00 | 4784.55 | 4767.09 | 4745.11 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-19 12:15:00 | 5023.78 | 4885.60 | 4818.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-03-21 12:15:00 | 5060.95 | 5065.39 | 5001.22 | SL hit (close<ema200) qty=0.50 sl=5065.39 alert=retest1 |

### Cycle 64 — SELL (started 2025-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 09:15:00 | 4976.45 | 5002.08 | 5002.91 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 09:15:00 | 5009.90 | 5001.48 | 5000.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 10:15:00 | 5082.80 | 5042.29 | 5026.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 12:15:00 | 5099.90 | 5111.15 | 5081.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-28 13:00:00 | 5099.90 | 5111.15 | 5081.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 5084.90 | 5105.90 | 5081.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 14:00:00 | 5084.90 | 5105.90 | 5081.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 14:15:00 | 5119.20 | 5108.56 | 5084.99 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2025-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 11:15:00 | 5018.40 | 5064.93 | 5070.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-03 10:15:00 | 4985.50 | 5036.54 | 5046.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-04 09:15:00 | 5076.80 | 5023.59 | 5031.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 5076.80 | 5023.59 | 5031.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 5076.80 | 5023.59 | 5031.73 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2025-04-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-04 10:15:00 | 5118.90 | 5042.65 | 5039.66 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 4854.15 | 5036.82 | 5044.26 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 11:15:00 | 5088.75 | 5029.67 | 5022.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 12:15:00 | 5157.45 | 5055.22 | 5034.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 10:15:00 | 5175.15 | 5183.48 | 5141.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-11 10:30:00 | 5162.70 | 5183.48 | 5141.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 13:15:00 | 5151.05 | 5182.50 | 5152.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-11 14:00:00 | 5151.05 | 5182.50 | 5152.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 14:15:00 | 5148.70 | 5175.74 | 5152.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 09:15:00 | 5274.50 | 5172.79 | 5152.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 5355.00 | 5488.83 | 5490.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 5355.00 | 5488.83 | 5490.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 5245.00 | 5440.06 | 5468.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 10:15:00 | 5353.00 | 5344.13 | 5393.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 11:00:00 | 5353.00 | 5344.13 | 5393.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 5306.00 | 5338.48 | 5370.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 10:15:00 | 5258.00 | 5338.48 | 5370.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 11:15:00 | 5272.00 | 5331.68 | 5364.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 13:15:00 | 5344.00 | 5309.76 | 5307.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2025-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 13:15:00 | 5344.00 | 5309.76 | 5307.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 09:15:00 | 5546.00 | 5366.29 | 5334.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 09:15:00 | 5453.50 | 5479.52 | 5422.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-06 10:00:00 | 5453.50 | 5479.52 | 5422.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 10:15:00 | 5413.00 | 5466.21 | 5421.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 10:45:00 | 5420.00 | 5466.21 | 5421.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 5406.50 | 5454.27 | 5420.23 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2025-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 15:15:00 | 5320.00 | 5387.95 | 5396.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 5246.50 | 5359.66 | 5383.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 15:15:00 | 5310.00 | 5303.34 | 5337.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-08 09:15:00 | 5319.50 | 5303.34 | 5337.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 5335.00 | 5309.68 | 5337.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-08 13:30:00 | 5277.50 | 5308.09 | 5329.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 5013.62 | 5198.62 | 5270.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 15:15:00 | 5120.00 | 5110.82 | 5184.68 | SL hit (close>ema200) qty=0.50 sl=5110.82 alert=retest2 |

### Cycle 73 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 5474.50 | 5236.15 | 5232.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 5516.00 | 5292.12 | 5257.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 12:15:00 | 5493.50 | 5498.30 | 5444.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 13:00:00 | 5493.50 | 5498.30 | 5444.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 5449.00 | 5481.51 | 5445.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:15:00 | 5479.50 | 5475.01 | 5445.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 12:15:00 | 5505.00 | 5550.70 | 5553.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 5505.00 | 5550.70 | 5553.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 5452.50 | 5531.06 | 5544.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 5473.00 | 5460.35 | 5491.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 15:00:00 | 5473.00 | 5460.35 | 5491.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 5457.50 | 5459.16 | 5485.92 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 5515.00 | 5497.43 | 5495.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 13:15:00 | 5526.00 | 5503.15 | 5498.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 15:15:00 | 5503.50 | 5506.47 | 5500.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 5489.00 | 5502.98 | 5499.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 5489.00 | 5502.98 | 5499.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:00:00 | 5489.00 | 5502.98 | 5499.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 10:15:00 | 5443.00 | 5490.98 | 5494.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 09:15:00 | 5289.00 | 5418.75 | 5454.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 12:15:00 | 5345.00 | 5328.80 | 5369.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-28 13:00:00 | 5345.00 | 5328.80 | 5369.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 5333.50 | 5315.91 | 5332.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 09:45:00 | 5302.00 | 5319.39 | 5328.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 15:15:00 | 5346.00 | 5331.69 | 5331.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 15:15:00 | 5346.00 | 5331.69 | 5331.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 09:15:00 | 5423.00 | 5349.95 | 5339.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 15:15:00 | 5379.00 | 5381.48 | 5364.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 09:15:00 | 5405.50 | 5381.48 | 5364.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 5402.50 | 5385.68 | 5367.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 10:45:00 | 5419.00 | 5392.54 | 5372.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 5460.00 | 5404.51 | 5387.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 5502.50 | 5582.11 | 5582.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 5502.50 | 5582.11 | 5582.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 5497.50 | 5565.19 | 5574.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 5320.00 | 5310.85 | 5389.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:00:00 | 5320.00 | 5310.85 | 5389.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 5404.00 | 5334.73 | 5375.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 5404.00 | 5334.73 | 5375.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 5389.50 | 5345.68 | 5376.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 5370.50 | 5345.68 | 5376.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 12:15:00 | 5359.50 | 5316.06 | 5312.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 79 — BUY (started 2025-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 12:15:00 | 5359.50 | 5316.06 | 5312.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 5388.50 | 5337.90 | 5323.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 5331.00 | 5344.86 | 5329.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 5331.00 | 5344.86 | 5329.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 5331.00 | 5344.86 | 5329.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 10:30:00 | 5363.00 | 5347.99 | 5332.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:00:00 | 5360.50 | 5347.99 | 5332.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-30 09:15:00 | 5899.30 | 5812.72 | 5733.84 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 80 — SELL (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 11:15:00 | 5826.50 | 5887.61 | 5893.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 13:15:00 | 5783.50 | 5854.05 | 5876.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 09:15:00 | 5776.00 | 5774.21 | 5808.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 12:15:00 | 5783.00 | 5761.53 | 5777.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 5783.00 | 5761.53 | 5777.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:45:00 | 5782.00 | 5761.53 | 5777.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 5782.00 | 5765.62 | 5777.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:30:00 | 5785.00 | 5765.62 | 5777.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 5799.00 | 5772.30 | 5779.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 5799.00 | 5772.30 | 5779.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 5800.00 | 5777.84 | 5781.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 5801.50 | 5777.84 | 5781.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 5836.50 | 5789.57 | 5786.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 10:15:00 | 5840.50 | 5799.76 | 5791.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 12:15:00 | 5842.50 | 5849.52 | 5829.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 12:45:00 | 5846.50 | 5849.52 | 5829.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 5923.00 | 5860.77 | 5840.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 5940.00 | 5902.82 | 5873.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 11:00:00 | 5944.00 | 5913.16 | 5883.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 14:30:00 | 5942.00 | 5925.11 | 5899.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:30:00 | 5945.50 | 5930.21 | 5906.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 5917.00 | 5932.22 | 5919.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:15:00 | 5905.50 | 5932.22 | 5919.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 5929.00 | 5931.58 | 5919.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 10:15:00 | 5940.50 | 5931.58 | 5919.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:30:00 | 5932.00 | 5925.24 | 5920.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 14:15:00 | 5943.50 | 5925.24 | 5920.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 09:15:00 | 5806.00 | 5906.99 | 5913.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 09:15:00 | 5806.00 | 5906.99 | 5913.83 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 12:15:00 | 5885.50 | 5867.35 | 5866.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 10:15:00 | 5915.00 | 5884.69 | 5875.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 09:15:00 | 5869.00 | 5910.63 | 5895.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 5869.00 | 5910.63 | 5895.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 5869.00 | 5910.63 | 5895.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:45:00 | 5826.50 | 5910.63 | 5895.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 5848.00 | 5898.11 | 5891.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:00:00 | 5848.00 | 5898.11 | 5891.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 5882.00 | 5889.61 | 5888.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:00:00 | 5882.00 | 5889.61 | 5888.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 5895.00 | 5890.69 | 5889.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:30:00 | 5883.50 | 5890.69 | 5889.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 5890.50 | 5890.65 | 5889.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 5915.00 | 5890.65 | 5889.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 5918.00 | 5896.12 | 5892.01 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 5868.00 | 5887.12 | 5888.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 5839.00 | 5874.87 | 5882.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 5845.00 | 5817.36 | 5838.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 5845.00 | 5817.36 | 5838.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 5845.00 | 5817.36 | 5838.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:00:00 | 5845.00 | 5817.36 | 5838.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 5843.50 | 5822.59 | 5839.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:45:00 | 5809.50 | 5820.07 | 5836.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 12:00:00 | 5820.00 | 5768.64 | 5770.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 12:15:00 | 5872.50 | 5789.41 | 5779.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 5872.50 | 5789.41 | 5779.40 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 10:15:00 | 5759.00 | 5799.24 | 5804.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 5730.00 | 5775.38 | 5789.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 12:15:00 | 5778.00 | 5770.15 | 5782.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 12:15:00 | 5778.00 | 5770.15 | 5782.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 5778.00 | 5770.15 | 5782.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:45:00 | 5780.50 | 5770.15 | 5782.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 5762.00 | 5768.52 | 5781.07 | EMA400 retest candle locked (from downside) |

### Cycle 87 — BUY (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 09:15:00 | 5855.00 | 5787.20 | 5786.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 10:15:00 | 5885.00 | 5806.76 | 5795.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 09:15:00 | 5833.00 | 5850.86 | 5828.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 5833.00 | 5850.86 | 5828.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 5833.00 | 5850.86 | 5828.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:00:00 | 5833.00 | 5850.86 | 5828.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 5843.00 | 5849.29 | 5829.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 15:15:00 | 5884.00 | 5845.77 | 5833.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 09:15:00 | 5825.50 | 5847.83 | 5837.00 | SL hit (close<static) qty=1.00 sl=5826.50 alert=retest2 |

### Cycle 88 — SELL (started 2025-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 12:15:00 | 5792.00 | 5827.47 | 5829.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 5772.50 | 5810.64 | 5821.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 5849.00 | 5812.45 | 5819.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 5849.00 | 5812.45 | 5819.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 5849.00 | 5812.45 | 5819.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:00:00 | 5849.00 | 5812.45 | 5819.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 5875.50 | 5825.06 | 5824.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 11:15:00 | 5888.50 | 5837.75 | 5830.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 10:15:00 | 5965.00 | 5965.44 | 5936.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 11:00:00 | 5965.00 | 5965.44 | 5936.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 6055.00 | 6097.32 | 6051.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:45:00 | 6026.50 | 6097.32 | 6051.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 6050.50 | 6087.96 | 6051.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 6050.50 | 6087.96 | 6051.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 6059.50 | 6082.26 | 6051.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 09:30:00 | 6097.00 | 6073.92 | 6057.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:30:00 | 6086.00 | 6118.19 | 6092.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 15:15:00 | 6080.00 | 6093.28 | 6093.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 15:15:00 | 6080.00 | 6093.28 | 6093.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 6036.50 | 6081.93 | 6088.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 12:15:00 | 6092.00 | 6078.04 | 6084.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 12:15:00 | 6092.00 | 6078.04 | 6084.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 6092.00 | 6078.04 | 6084.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:00:00 | 6092.00 | 6078.04 | 6084.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 6103.00 | 6083.03 | 6085.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:45:00 | 6102.50 | 6083.03 | 6085.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 14:15:00 | 6115.00 | 6089.43 | 6088.62 | EMA200 above EMA400 |

### Cycle 92 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 6067.00 | 6086.45 | 6087.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 6036.00 | 6072.01 | 6080.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 10:15:00 | 5689.50 | 5686.23 | 5775.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 11:00:00 | 5689.50 | 5686.23 | 5775.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 5712.50 | 5683.66 | 5700.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 12:45:00 | 5711.00 | 5683.66 | 5700.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 5731.50 | 5693.23 | 5703.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:00:00 | 5731.50 | 5693.23 | 5703.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 15:15:00 | 5740.50 | 5710.25 | 5709.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 5772.00 | 5722.60 | 5715.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 13:15:00 | 5711.00 | 5722.10 | 5717.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 13:15:00 | 5711.00 | 5722.10 | 5717.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 5711.00 | 5722.10 | 5717.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 5711.00 | 5722.10 | 5717.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 5689.00 | 5715.48 | 5715.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 5689.00 | 5715.48 | 5715.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 94 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 5691.00 | 5710.59 | 5712.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 5678.50 | 5704.17 | 5709.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 5707.00 | 5681.78 | 5691.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 5707.00 | 5681.78 | 5691.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 5707.00 | 5681.78 | 5691.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 5707.00 | 5681.78 | 5691.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 5715.50 | 5688.52 | 5693.47 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 12:15:00 | 5735.00 | 5702.77 | 5699.40 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 09:15:00 | 5673.00 | 5694.76 | 5696.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 10:15:00 | 5659.50 | 5687.71 | 5693.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 5645.00 | 5633.30 | 5649.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 5645.00 | 5633.30 | 5649.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 5645.00 | 5633.30 | 5649.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 5642.00 | 5633.30 | 5649.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 5613.00 | 5629.24 | 5646.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:30:00 | 5605.00 | 5627.79 | 5644.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 13:15:00 | 5666.50 | 5638.69 | 5646.35 | SL hit (close>static) qty=1.00 sl=5650.00 alert=retest2 |

### Cycle 97 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 5749.50 | 5668.73 | 5658.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 10:15:00 | 5765.00 | 5687.99 | 5668.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 14:15:00 | 5730.00 | 5734.42 | 5715.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 15:15:00 | 5721.00 | 5734.42 | 5715.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 5715.50 | 5728.49 | 5715.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:30:00 | 5750.50 | 5727.31 | 5720.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 11:00:00 | 5748.50 | 5731.55 | 5722.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 11:30:00 | 5746.00 | 5734.74 | 5725.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 12:30:00 | 5746.50 | 5736.59 | 5726.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 5733.50 | 5738.50 | 5731.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:45:00 | 5736.00 | 5738.50 | 5731.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 5725.50 | 5735.90 | 5731.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:00:00 | 5725.50 | 5735.90 | 5731.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 5718.00 | 5732.32 | 5729.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:45:00 | 5721.50 | 5732.32 | 5729.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-18 14:15:00 | 5716.00 | 5728.60 | 5728.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 14:15:00 | 5716.00 | 5728.60 | 5728.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 5693.50 | 5718.77 | 5724.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 09:15:00 | 5743.00 | 5698.30 | 5706.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 5743.00 | 5698.30 | 5706.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 5743.00 | 5698.30 | 5706.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:00:00 | 5743.00 | 5698.30 | 5706.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 5745.00 | 5707.64 | 5709.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:30:00 | 5740.00 | 5707.64 | 5709.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 11:15:00 | 5747.00 | 5715.51 | 5713.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 12:15:00 | 5768.50 | 5726.11 | 5718.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 13:15:00 | 5748.00 | 5753.41 | 5740.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-23 14:15:00 | 5745.50 | 5753.41 | 5740.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 5732.00 | 5749.13 | 5739.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 15:00:00 | 5732.00 | 5749.13 | 5739.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 5730.50 | 5745.40 | 5738.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:15:00 | 5725.00 | 5745.40 | 5738.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 5722.50 | 5740.20 | 5737.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 5722.50 | 5740.20 | 5737.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 5700.50 | 5732.26 | 5734.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 5691.00 | 5716.48 | 5725.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 13:15:00 | 5697.00 | 5693.08 | 5707.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 13:15:00 | 5697.00 | 5693.08 | 5707.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 5697.00 | 5693.08 | 5707.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:00:00 | 5697.00 | 5693.08 | 5707.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 5665.50 | 5687.57 | 5703.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 5654.00 | 5685.65 | 5701.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 10:00:00 | 5637.50 | 5676.02 | 5695.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 5756.50 | 5596.08 | 5615.96 | SL hit (close>static) qty=1.00 sl=5708.50 alert=retest2 |

### Cycle 101 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 5665.00 | 5602.66 | 5601.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 12:15:00 | 5688.00 | 5649.09 | 5633.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 5639.50 | 5665.80 | 5648.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 09:15:00 | 5639.50 | 5665.80 | 5648.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 5639.50 | 5665.80 | 5648.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:45:00 | 5634.00 | 5665.80 | 5648.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 5647.50 | 5662.14 | 5648.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 15:15:00 | 5665.50 | 5657.76 | 5650.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 10:15:00 | 5671.50 | 5656.25 | 5650.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 5633.00 | 5648.83 | 5649.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 5633.00 | 5648.83 | 5649.09 | EMA200 below EMA400 |

### Cycle 103 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 5714.00 | 5661.00 | 5654.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 12:15:00 | 5729.00 | 5684.68 | 5666.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 09:15:00 | 5772.00 | 5774.20 | 5749.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 10:15:00 | 5762.00 | 5774.20 | 5749.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 5750.50 | 5767.19 | 5750.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:00:00 | 5750.50 | 5767.19 | 5750.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 5771.00 | 5767.95 | 5752.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 13:30:00 | 5777.00 | 5770.06 | 5754.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 5797.50 | 5767.60 | 5756.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 12:15:00 | 5841.00 | 5881.89 | 5884.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 104 — SELL (started 2025-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 12:15:00 | 5841.00 | 5881.89 | 5884.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 13:15:00 | 5829.50 | 5871.41 | 5879.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 14:15:00 | 5774.50 | 5767.71 | 5810.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-24 15:00:00 | 5774.50 | 5767.71 | 5810.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 5810.00 | 5775.81 | 5806.74 | EMA400 retest candle locked (from downside) |

### Cycle 105 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 5861.00 | 5820.71 | 5820.60 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 09:15:00 | 5793.50 | 5820.41 | 5820.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 13:15:00 | 5750.00 | 5795.85 | 5808.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 14:15:00 | 5816.50 | 5799.98 | 5809.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 14:15:00 | 5816.50 | 5799.98 | 5809.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 5816.50 | 5799.98 | 5809.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 5816.50 | 5799.98 | 5809.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 5834.50 | 5806.88 | 5811.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 5791.00 | 5806.88 | 5811.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:30:00 | 5804.00 | 5810.08 | 5810.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 15:15:00 | 5822.00 | 5812.47 | 5811.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 15:15:00 | 5822.00 | 5812.47 | 5811.32 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 5776.50 | 5805.27 | 5808.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 11:15:00 | 5762.00 | 5791.94 | 5801.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 10:15:00 | 5675.50 | 5662.51 | 5701.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 11:00:00 | 5675.50 | 5662.51 | 5701.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 5692.00 | 5669.44 | 5692.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 5692.00 | 5669.44 | 5692.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 5700.00 | 5675.55 | 5692.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 5679.50 | 5675.55 | 5692.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 10:15:00 | 5736.50 | 5667.88 | 5672.32 | SL hit (close>static) qty=1.00 sl=5703.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 11:15:00 | 5767.00 | 5687.70 | 5680.93 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 10:15:00 | 5641.50 | 5677.29 | 5680.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 11:15:00 | 5608.00 | 5663.43 | 5674.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 11:15:00 | 5624.50 | 5612.67 | 5636.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 11:15:00 | 5624.50 | 5612.67 | 5636.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 5624.50 | 5612.67 | 5636.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:45:00 | 5631.00 | 5612.67 | 5636.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 5631.00 | 5616.34 | 5636.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:30:00 | 5636.50 | 5616.34 | 5636.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 5604.50 | 5613.97 | 5633.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 13:30:00 | 5614.00 | 5613.97 | 5633.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 5649.00 | 5613.35 | 5627.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:45:00 | 5638.00 | 5613.35 | 5627.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 5670.00 | 5624.68 | 5631.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:45:00 | 5672.50 | 5624.68 | 5631.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 11:15:00 | 5698.00 | 5639.35 | 5637.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 12:15:00 | 5722.00 | 5655.88 | 5645.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 10:15:00 | 5878.50 | 5883.25 | 5830.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 10:45:00 | 5884.50 | 5883.25 | 5830.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 5842.50 | 5880.18 | 5855.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:00:00 | 5842.50 | 5880.18 | 5855.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 5867.00 | 5877.55 | 5856.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 13:15:00 | 5875.00 | 5876.54 | 5858.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 14:00:00 | 5876.00 | 5876.43 | 5859.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 15:15:00 | 5878.00 | 5874.54 | 5860.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 5796.00 | 5859.39 | 5856.05 | SL hit (close<static) qty=1.00 sl=5841.50 alert=retest2 |

### Cycle 112 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 5767.50 | 5841.01 | 5848.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 5740.50 | 5792.12 | 5819.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 14:15:00 | 5764.50 | 5750.99 | 5779.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 15:00:00 | 5764.50 | 5750.99 | 5779.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 5798.50 | 5763.21 | 5780.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 5803.00 | 5763.21 | 5780.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 5791.00 | 5768.77 | 5781.40 | EMA400 retest candle locked (from downside) |

### Cycle 113 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 5830.00 | 5794.84 | 5791.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 12:15:00 | 5847.00 | 5812.68 | 5801.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 13:15:00 | 5846.50 | 5847.94 | 5830.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-24 14:00:00 | 5846.50 | 5847.94 | 5830.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 5807.00 | 5839.75 | 5827.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 5807.00 | 5839.75 | 5827.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 5805.50 | 5832.90 | 5825.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:15:00 | 5759.00 | 5832.90 | 5825.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — SELL (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 10:15:00 | 5785.00 | 5814.06 | 5817.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 11:15:00 | 5763.50 | 5803.95 | 5813.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 5838.50 | 5798.56 | 5805.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 5838.50 | 5798.56 | 5805.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 5838.50 | 5798.56 | 5805.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:45:00 | 5841.50 | 5798.56 | 5805.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 5881.00 | 5815.05 | 5812.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 11:15:00 | 5890.50 | 5830.14 | 5819.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 5902.00 | 5908.69 | 5883.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 13:15:00 | 5891.00 | 5905.07 | 5890.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 5891.00 | 5905.07 | 5890.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:00:00 | 5891.00 | 5905.07 | 5890.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 5899.00 | 5903.86 | 5890.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:30:00 | 5895.00 | 5903.86 | 5890.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 5880.00 | 5898.95 | 5890.93 | EMA400 retest candle locked (from upside) |

### Cycle 116 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 11:15:00 | 5792.50 | 5871.99 | 5879.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 15:15:00 | 5772.50 | 5824.15 | 5852.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 4974.00 | 4948.25 | 5089.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 15:00:00 | 4974.00 | 4948.25 | 5089.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 4894.00 | 4835.25 | 4889.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:45:00 | 4882.00 | 4835.25 | 4889.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 4881.00 | 4844.40 | 4888.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 11:45:00 | 4842.00 | 4842.62 | 4883.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 4941.00 | 4868.51 | 4880.52 | SL hit (close>static) qty=1.00 sl=4894.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 10:15:00 | 4994.00 | 4893.60 | 4890.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 10:15:00 | 5055.00 | 4991.33 | 4971.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 15:15:00 | 5113.00 | 5121.91 | 5079.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 09:15:00 | 5160.00 | 5121.91 | 5079.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 5155.00 | 5153.08 | 5135.68 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 5064.50 | 5115.04 | 5121.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 5059.50 | 5094.54 | 5109.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 14:15:00 | 5071.00 | 5068.93 | 5088.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 15:00:00 | 5071.00 | 5068.93 | 5088.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 5089.50 | 5072.57 | 5087.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:30:00 | 5092.00 | 5072.57 | 5087.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 5087.00 | 5075.46 | 5087.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:15:00 | 5083.00 | 5075.46 | 5087.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 14:45:00 | 5083.50 | 5079.99 | 5085.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 5066.50 | 5080.99 | 5085.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 09:15:00 | 5112.50 | 5054.81 | 5050.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 09:15:00 | 5112.50 | 5054.81 | 5050.66 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 5051.50 | 5092.21 | 5094.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 5026.00 | 5064.51 | 5079.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 10:15:00 | 4977.00 | 4968.68 | 5002.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-08 10:30:00 | 4984.50 | 4968.68 | 5002.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 4902.50 | 4772.19 | 4772.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 4902.50 | 4772.19 | 4772.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 10:15:00 | 4924.00 | 4802.55 | 4786.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 14:15:00 | 4943.00 | 4877.16 | 4831.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 4861.00 | 4883.50 | 4842.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 09:45:00 | 4854.00 | 4883.50 | 4842.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 4856.50 | 4878.10 | 4843.84 | EMA400 retest candle locked (from upside) |

### Cycle 122 — SELL (started 2026-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 15:15:00 | 4795.00 | 4825.75 | 4828.30 | EMA200 below EMA400 |

### Cycle 123 — BUY (started 2026-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 12:15:00 | 4896.00 | 4840.58 | 4834.13 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 4762.00 | 4860.02 | 4861.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 4705.00 | 4812.53 | 4838.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 4761.00 | 4739.38 | 4772.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 4761.00 | 4739.38 | 4772.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 4761.00 | 4739.38 | 4772.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 14:45:00 | 4776.50 | 4739.38 | 4772.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 4763.50 | 4744.21 | 4771.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 4775.50 | 4744.21 | 4771.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 4769.00 | 4749.16 | 4771.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 13:15:00 | 4729.50 | 4757.55 | 4770.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 4676.50 | 4752.29 | 4764.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 14:15:00 | 4684.70 | 4638.82 | 4634.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 4684.70 | 4638.82 | 4634.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 4914.80 | 4702.20 | 4664.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 4854.00 | 4922.15 | 4863.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 4854.00 | 4922.15 | 4863.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 4854.00 | 4922.15 | 4863.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:30:00 | 4883.80 | 4896.13 | 4867.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 10:15:00 | 4883.90 | 4903.03 | 4878.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:00:00 | 4887.30 | 4899.88 | 4879.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:30:00 | 4902.50 | 4900.61 | 4881.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 4955.90 | 4913.27 | 4894.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 10:15:00 | 4963.60 | 4913.27 | 4894.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 12:00:00 | 4960.00 | 4931.49 | 4906.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 13:15:00 | 4964.00 | 4936.87 | 4911.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 14:15:00 | 4963.00 | 4940.22 | 4915.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 4963.50 | 4954.77 | 4937.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:30:00 | 4942.70 | 4954.77 | 4937.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 4998.50 | 4998.98 | 4975.96 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 4916.70 | 4963.66 | 4969.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 4916.70 | 4963.66 | 4969.18 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 4968.40 | 4956.74 | 4955.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 4991.80 | 4966.06 | 4960.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 15:15:00 | 4966.40 | 4969.05 | 4963.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 4971.80 | 4969.60 | 4964.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 4971.80 | 4969.60 | 4964.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:45:00 | 4968.90 | 4969.60 | 4964.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 4980.00 | 4980.61 | 4972.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:45:00 | 4975.50 | 4980.61 | 4972.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 4972.00 | 4978.89 | 4972.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 4891.00 | 4978.89 | 4972.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 4892.40 | 4961.59 | 4965.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 4851.50 | 4914.07 | 4939.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 12:15:00 | 4861.00 | 4853.47 | 4891.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 13:00:00 | 4861.00 | 4853.47 | 4891.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 4866.20 | 4851.84 | 4867.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:00:00 | 4866.20 | 4851.84 | 4867.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 4855.00 | 4852.47 | 4866.47 | EMA400 retest candle locked (from downside) |

### Cycle 129 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 4940.90 | 4869.30 | 4865.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 10:15:00 | 4955.90 | 4886.62 | 4874.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 4924.80 | 4931.90 | 4910.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 12:00:00 | 4924.80 | 4931.90 | 4910.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 4936.90 | 4936.92 | 4918.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:45:00 | 4927.50 | 4936.92 | 4918.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 4876.10 | 4924.61 | 4915.94 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 10:15:00 | 4817.50 | 4903.19 | 4906.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 4640.80 | 4808.46 | 4854.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 4471.20 | 4399.30 | 4476.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 14:15:00 | 4471.20 | 4399.30 | 4476.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 4471.20 | 4399.30 | 4476.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 4540.30 | 4399.30 | 4476.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 4495.00 | 4418.44 | 4478.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 4402.60 | 4418.44 | 4478.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 4182.47 | 4355.51 | 4417.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 4360.00 | 4278.77 | 4333.47 | SL hit (close>ema200) qty=0.50 sl=4278.77 alert=retest2 |

### Cycle 131 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 4383.50 | 4357.22 | 4356.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 4423.90 | 4370.55 | 4362.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 4352.30 | 4376.41 | 4367.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 12:15:00 | 4352.30 | 4376.41 | 4367.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 4352.30 | 4376.41 | 4367.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 4352.30 | 4376.41 | 4367.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 4351.90 | 4371.51 | 4366.39 | EMA400 retest candle locked (from upside) |

### Cycle 132 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 4350.20 | 4362.80 | 4363.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 4256.10 | 4341.46 | 4353.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 09:15:00 | 4207.50 | 4186.71 | 4232.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 4207.50 | 4186.71 | 4232.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 4207.50 | 4186.71 | 4232.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:45:00 | 4234.00 | 4186.71 | 4232.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 4208.60 | 4195.69 | 4229.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 12:15:00 | 4193.30 | 4195.69 | 4229.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 09:15:00 | 4289.40 | 4221.01 | 4228.11 | SL hit (close>static) qty=1.00 sl=4246.80 alert=retest2 |

### Cycle 133 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 4256.70 | 4234.24 | 4233.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 14:15:00 | 4292.30 | 4249.66 | 4240.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 4223.50 | 4312.58 | 4290.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 4223.50 | 4312.58 | 4290.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 4223.50 | 4312.58 | 4290.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 4223.50 | 4312.58 | 4290.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 4206.00 | 4291.27 | 4283.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 4206.00 | 4291.27 | 4283.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 4202.60 | 4263.35 | 4271.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 4184.50 | 4247.58 | 4263.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 4040.00 | 4000.71 | 4076.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 4040.00 | 4000.71 | 4076.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 4040.00 | 4000.71 | 4076.35 | EMA400 retest candle locked (from downside) |

### Cycle 135 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 4157.00 | 4100.04 | 4099.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 4240.70 | 4128.18 | 4112.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 4168.70 | 4231.22 | 4187.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 4168.70 | 4231.22 | 4187.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 4168.70 | 4231.22 | 4187.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 4168.70 | 4231.22 | 4187.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 4148.60 | 4214.69 | 4183.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 4148.60 | 4214.69 | 4183.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 4097.40 | 4159.52 | 4164.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 4055.00 | 4130.98 | 4149.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 4276.00 | 4085.51 | 4108.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 4276.00 | 4085.51 | 4108.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 4276.00 | 4085.51 | 4108.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 4276.00 | 4085.51 | 4108.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 4240.00 | 4116.41 | 4120.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:30:00 | 4312.30 | 4116.41 | 4120.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 4210.40 | 4135.21 | 4128.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 4262.30 | 4160.63 | 4140.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 4018.80 | 4144.40 | 4141.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 4018.80 | 4144.40 | 4141.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 4018.80 | 4144.40 | 4141.47 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 4015.30 | 4118.58 | 4130.00 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 4170.00 | 4134.66 | 4131.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 4251.10 | 4168.84 | 4149.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 4219.30 | 4220.81 | 4185.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 4219.30 | 4220.81 | 4185.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 4451.50 | 4504.55 | 4451.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 4445.80 | 4504.55 | 4451.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 4448.00 | 4493.24 | 4451.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 4486.00 | 4493.24 | 4451.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 4395.20 | 4503.81 | 4483.52 | SL hit (close<static) qty=1.00 sl=4443.30 alert=retest2 |

### Cycle 140 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 4439.10 | 4467.40 | 4470.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 14:15:00 | 4431.30 | 4460.18 | 4466.75 | Break + close below crossover candle low |

### Cycle 141 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 4643.00 | 4492.54 | 4480.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 10:15:00 | 4674.70 | 4622.59 | 4568.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 13:15:00 | 4623.30 | 4627.48 | 4584.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-17 09:15:00 | 4612.60 | 4620.69 | 4591.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 4612.60 | 4620.69 | 4591.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 10:00:00 | 4612.60 | 4620.69 | 4591.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 4626.20 | 4630.78 | 4612.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:45:00 | 4605.60 | 4630.78 | 4612.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 4641.60 | 4678.28 | 4664.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:00:00 | 4641.60 | 4678.28 | 4664.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 4663.00 | 4675.23 | 4663.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 11:45:00 | 4678.10 | 4675.58 | 4665.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 4578.40 | 4648.07 | 4655.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 142 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 4578.40 | 4648.07 | 4655.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 11:15:00 | 4458.90 | 4527.80 | 4549.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 14:15:00 | 4306.80 | 4293.11 | 4363.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 15:00:00 | 4306.80 | 4293.11 | 4363.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 4338.30 | 4303.25 | 4355.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 4288.70 | 4303.96 | 4347.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 4376.20 | 4309.44 | 4303.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 4376.20 | 4309.44 | 4303.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 4378.50 | 4328.96 | 4313.62 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-22 12:15:00 | 4365.45 | 2024-05-24 11:15:00 | 4274.10 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2024-05-22 13:15:00 | 4371.50 | 2024-05-24 11:15:00 | 4274.10 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2024-05-22 14:00:00 | 4367.35 | 2024-05-24 11:15:00 | 4274.10 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2024-05-23 09:15:00 | 4410.20 | 2024-05-24 11:15:00 | 4274.10 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2024-05-28 10:00:00 | 4231.95 | 2024-05-29 14:15:00 | 4020.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-28 10:00:00 | 4231.95 | 2024-05-30 09:15:00 | 4092.35 | STOP_HIT | 0.50 | 3.30% |
| SELL | retest2 | 2024-06-19 10:15:00 | 4232.25 | 2024-06-21 13:15:00 | 4312.65 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2024-06-21 11:30:00 | 4249.00 | 2024-06-21 13:15:00 | 4312.65 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2024-06-25 12:15:00 | 4274.65 | 2024-06-25 13:15:00 | 4263.60 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2024-07-23 11:30:00 | 4302.00 | 2024-07-24 12:15:00 | 4344.35 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-07-23 12:15:00 | 4206.80 | 2024-07-24 12:15:00 | 4344.35 | STOP_HIT | 1.00 | -3.27% |
| SELL | retest2 | 2024-07-23 13:00:00 | 4297.45 | 2024-07-24 12:15:00 | 4344.35 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-07-30 09:45:00 | 4498.60 | 2024-08-01 09:15:00 | 4432.75 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2024-07-30 11:15:00 | 4482.05 | 2024-08-01 10:15:00 | 4409.00 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-07-30 12:15:00 | 4481.25 | 2024-08-01 10:15:00 | 4409.00 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2024-07-30 14:15:00 | 4485.75 | 2024-08-01 10:15:00 | 4409.00 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2024-07-31 13:45:00 | 4491.00 | 2024-08-01 10:15:00 | 4409.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-08-06 14:30:00 | 4270.10 | 2024-08-08 11:15:00 | 4323.90 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-08-16 10:30:00 | 4220.35 | 2024-08-16 11:15:00 | 4266.35 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-09-16 09:30:00 | 4964.05 | 2024-09-17 09:15:00 | 4927.00 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-09-16 15:15:00 | 4953.70 | 2024-09-17 09:15:00 | 4927.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-09-19 13:00:00 | 4846.45 | 2024-09-23 09:15:00 | 4932.55 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-09-19 13:45:00 | 4836.35 | 2024-09-23 09:15:00 | 4932.55 | STOP_HIT | 1.00 | -1.99% |
| SELL | retest2 | 2024-09-20 09:15:00 | 4806.90 | 2024-09-23 09:15:00 | 4932.55 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2024-09-26 13:45:00 | 4831.20 | 2024-09-26 15:15:00 | 4865.00 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2024-10-18 14:00:00 | 4662.50 | 2024-10-25 09:15:00 | 4429.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 14:00:00 | 4662.50 | 2024-10-28 09:15:00 | 4196.25 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-11-07 12:15:00 | 4013.20 | 2024-11-08 15:15:00 | 3990.35 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2024-11-08 09:15:00 | 4035.90 | 2024-11-08 15:15:00 | 3990.35 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2024-11-08 11:30:00 | 4021.05 | 2024-11-08 15:15:00 | 3990.35 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-11-08 14:30:00 | 4020.30 | 2024-11-08 15:15:00 | 3990.35 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2024-11-14 15:15:00 | 3889.90 | 2024-11-18 09:15:00 | 3949.20 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-11-27 11:15:00 | 4261.30 | 2024-12-05 09:15:00 | 4325.40 | STOP_HIT | 1.00 | 1.50% |
| SELL | retest2 | 2024-12-17 11:15:00 | 4381.40 | 2024-12-19 13:15:00 | 4425.60 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-12-17 14:30:00 | 4387.10 | 2024-12-19 13:15:00 | 4425.60 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-12-18 09:45:00 | 4387.10 | 2024-12-19 13:15:00 | 4425.60 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2024-12-18 12:30:00 | 4384.75 | 2024-12-19 13:15:00 | 4425.60 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-12-23 09:15:00 | 4440.30 | 2024-12-30 15:15:00 | 4613.05 | STOP_HIT | 1.00 | 3.89% |
| SELL | retest2 | 2025-01-02 09:15:00 | 4570.95 | 2025-01-06 10:15:00 | 4342.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-02 09:15:00 | 4570.95 | 2025-01-07 10:15:00 | 4327.00 | STOP_HIT | 0.50 | 5.34% |
| BUY | retest2 | 2025-01-27 11:00:00 | 4230.00 | 2025-02-04 12:15:00 | 4320.50 | STOP_HIT | 1.00 | 2.14% |
| BUY | retest2 | 2025-01-28 10:15:00 | 4227.10 | 2025-02-04 12:15:00 | 4320.50 | STOP_HIT | 1.00 | 2.21% |
| BUY | retest2 | 2025-01-28 12:00:00 | 4228.55 | 2025-02-04 12:15:00 | 4320.50 | STOP_HIT | 1.00 | 2.17% |
| BUY | retest2 | 2025-01-30 15:00:00 | 4232.00 | 2025-02-04 12:15:00 | 4320.50 | STOP_HIT | 1.00 | 2.09% |
| BUY | retest2 | 2025-01-31 11:00:00 | 4273.55 | 2025-02-04 12:15:00 | 4320.50 | STOP_HIT | 1.00 | 1.10% |
| SELL | retest2 | 2025-02-11 12:45:00 | 4328.15 | 2025-02-13 11:15:00 | 4340.60 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-02-11 14:15:00 | 4322.00 | 2025-02-13 11:15:00 | 4340.60 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-02-18 09:15:00 | 4276.00 | 2025-02-18 12:15:00 | 4305.30 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest1 | 2025-02-24 09:15:00 | 4518.00 | 2025-02-25 09:15:00 | 4487.95 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest1 | 2025-03-18 14:15:00 | 4784.55 | 2025-03-19 12:15:00 | 5023.78 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-03-18 14:15:00 | 4784.55 | 2025-03-21 12:15:00 | 5060.95 | STOP_HIT | 0.50 | 5.78% |
| BUY | retest2 | 2025-04-15 09:15:00 | 5274.50 | 2025-04-25 09:15:00 | 5355.00 | STOP_HIT | 1.00 | 1.53% |
| SELL | retest2 | 2025-04-29 10:15:00 | 5258.00 | 2025-05-02 13:15:00 | 5344.00 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-04-29 11:15:00 | 5272.00 | 2025-05-02 13:15:00 | 5344.00 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-05-08 13:30:00 | 5277.50 | 2025-05-09 09:15:00 | 5013.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-05-08 13:30:00 | 5277.50 | 2025-05-09 15:15:00 | 5120.00 | STOP_HIT | 0.50 | 2.98% |
| BUY | retest2 | 2025-05-15 09:15:00 | 5479.50 | 2025-05-20 12:15:00 | 5505.00 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2025-06-02 09:45:00 | 5302.00 | 2025-06-02 15:15:00 | 5346.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-06-04 10:45:00 | 5419.00 | 2025-06-12 10:15:00 | 5502.50 | STOP_HIT | 1.00 | 1.54% |
| BUY | retest2 | 2025-06-05 09:15:00 | 5460.00 | 2025-06-12 10:15:00 | 5502.50 | STOP_HIT | 1.00 | 0.78% |
| SELL | retest2 | 2025-06-17 09:15:00 | 5370.50 | 2025-06-20 12:15:00 | 5359.50 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-06-23 10:30:00 | 5363.00 | 2025-06-30 09:15:00 | 5899.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-23 11:00:00 | 5360.50 | 2025-06-30 09:15:00 | 5896.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-14 09:15:00 | 5940.00 | 2025-07-17 09:15:00 | 5806.00 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-07-14 11:00:00 | 5944.00 | 2025-07-17 09:15:00 | 5806.00 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-07-14 14:30:00 | 5942.00 | 2025-07-17 09:15:00 | 5806.00 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-07-15 09:30:00 | 5945.50 | 2025-07-17 09:15:00 | 5806.00 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-07-16 10:15:00 | 5940.50 | 2025-07-17 09:15:00 | 5806.00 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-07-16 13:30:00 | 5932.00 | 2025-07-17 09:15:00 | 5806.00 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-07-16 14:15:00 | 5943.50 | 2025-07-17 09:15:00 | 5806.00 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-07-28 11:45:00 | 5809.50 | 2025-07-31 12:15:00 | 5872.50 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-07-31 12:00:00 | 5820.00 | 2025-07-31 12:15:00 | 5872.50 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-08-07 15:15:00 | 5884.00 | 2025-08-08 09:15:00 | 5825.50 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-08-20 09:30:00 | 6097.00 | 2025-08-22 15:15:00 | 6080.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-08-21 09:30:00 | 6086.00 | 2025-08-22 15:15:00 | 6080.00 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-09-11 11:30:00 | 5605.00 | 2025-09-11 13:15:00 | 5666.50 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-09-17 09:30:00 | 5750.50 | 2025-09-18 14:15:00 | 5716.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-09-17 11:00:00 | 5748.50 | 2025-09-18 14:15:00 | 5716.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-09-17 11:30:00 | 5746.00 | 2025-09-18 14:15:00 | 5716.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-09-17 12:30:00 | 5746.50 | 2025-09-18 14:15:00 | 5716.00 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-09-26 09:15:00 | 5654.00 | 2025-09-29 14:15:00 | 5756.50 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-09-26 10:00:00 | 5637.50 | 2025-09-29 14:15:00 | 5756.50 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-09-29 15:15:00 | 5572.50 | 2025-10-03 09:15:00 | 5665.00 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-10-07 15:15:00 | 5665.50 | 2025-10-08 14:15:00 | 5633.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-10-08 10:15:00 | 5671.50 | 2025-10-08 14:15:00 | 5633.00 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-10-14 13:30:00 | 5777.00 | 2025-10-23 12:15:00 | 5841.00 | STOP_HIT | 1.00 | 1.11% |
| BUY | retest2 | 2025-10-15 09:15:00 | 5797.50 | 2025-10-23 12:15:00 | 5841.00 | STOP_HIT | 1.00 | 0.75% |
| SELL | retest2 | 2025-10-29 09:15:00 | 5791.00 | 2025-10-29 15:15:00 | 5822.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-10-29 14:30:00 | 5804.00 | 2025-10-29 15:15:00 | 5822.00 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-11-04 09:15:00 | 5679.50 | 2025-11-06 10:15:00 | 5736.50 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-11-17 13:15:00 | 5875.00 | 2025-11-18 09:15:00 | 5796.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-11-17 14:00:00 | 5876.00 | 2025-11-18 09:15:00 | 5796.00 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-11-17 15:15:00 | 5878.00 | 2025-11-18 09:15:00 | 5796.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-12-12 11:45:00 | 4842.00 | 2025-12-15 09:15:00 | 4941.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-12-29 11:15:00 | 5083.00 | 2026-01-01 09:15:00 | 5112.50 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-12-29 14:45:00 | 5083.50 | 2026-01-01 09:15:00 | 5112.50 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-12-30 09:15:00 | 5066.50 | 2026-01-01 09:15:00 | 5112.50 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-01-28 13:15:00 | 4729.50 | 2026-02-02 14:15:00 | 4684.70 | STOP_HIT | 1.00 | 0.95% |
| SELL | retest2 | 2026-01-29 09:15:00 | 4676.50 | 2026-02-02 14:15:00 | 4684.70 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2026-02-05 13:30:00 | 4883.80 | 2026-02-13 10:15:00 | 4916.70 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2026-02-06 10:15:00 | 4883.90 | 2026-02-13 10:15:00 | 4916.70 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2026-02-06 11:00:00 | 4887.30 | 2026-02-13 10:15:00 | 4916.70 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest2 | 2026-02-06 11:30:00 | 4902.50 | 2026-02-13 10:15:00 | 4916.70 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2026-02-09 10:15:00 | 4963.60 | 2026-02-13 10:15:00 | 4916.70 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-02-09 12:00:00 | 4960.00 | 2026-02-13 10:15:00 | 4916.70 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-02-09 13:15:00 | 4964.00 | 2026-02-13 10:15:00 | 4916.70 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2026-02-09 14:15:00 | 4963.00 | 2026-02-13 10:15:00 | 4916.70 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-03-06 09:15:00 | 4402.60 | 2026-03-09 09:15:00 | 4182.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 09:15:00 | 4402.60 | 2026-03-10 09:15:00 | 4360.00 | STOP_HIT | 0.50 | 0.97% |
| SELL | retest2 | 2026-03-16 12:15:00 | 4193.30 | 2026-03-17 09:15:00 | 4289.40 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2026-04-10 09:15:00 | 4486.00 | 2026-04-13 09:15:00 | 4395.20 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2026-04-13 12:15:00 | 4459.20 | 2026-04-13 13:15:00 | 4439.10 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2026-04-22 11:45:00 | 4678.10 | 2026-04-23 09:15:00 | 4578.40 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2026-05-04 12:00:00 | 4288.70 | 2026-05-06 11:15:00 | 4376.20 | STOP_HIT | 1.00 | -2.04% |
