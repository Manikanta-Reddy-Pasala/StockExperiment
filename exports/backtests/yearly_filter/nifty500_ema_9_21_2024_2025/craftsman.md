# Craftsman Automation Ltd. (CRAFTSMAN)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 8990.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 159 |
| ALERT1 | 97 |
| ALERT2 | 96 |
| ALERT2_SKIP | 54 |
| ALERT3 | 262 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 122 |
| PARTIAL | 20 |
| TARGET_HIT | 27 |
| STOP_HIT | 105 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 150 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 68 / 82
- **Target hits / Stop hits / Partials:** 27 / 103 / 20
- **Avg / median % per leg:** 1.74% / -0.43%
- **Sum % (uncompounded):** 260.75%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 65 | 24 | 36.9% | 17 | 48 | 0 | 1.96% | 127.4% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.07% | -4.3% |
| BUY @ 3rd Alert (retest2) | 61 | 24 | 39.3% | 17 | 44 | 0 | 2.16% | 131.7% |
| SELL (all) | 85 | 44 | 51.8% | 10 | 55 | 20 | 1.57% | 133.3% |
| SELL @ 2nd Alert (retest1) | 5 | 2 | 40.0% | 0 | 4 | 1 | -1.15% | -5.7% |
| SELL @ 3rd Alert (retest2) | 80 | 42 | 52.5% | 10 | 51 | 19 | 1.74% | 139.1% |
| retest1 (combined) | 9 | 2 | 22.2% | 0 | 8 | 1 | -1.11% | -10.0% |
| retest2 (combined) | 141 | 66 | 46.8% | 27 | 95 | 19 | 1.92% | 270.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 15:15:00 | 4350.00 | 4317.43 | 4317.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 10:15:00 | 4422.80 | 4340.84 | 4328.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 12:15:00 | 4332.50 | 4345.56 | 4332.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 12:15:00 | 4332.50 | 4345.56 | 4332.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 12:15:00 | 4332.50 | 4345.56 | 4332.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 13:00:00 | 4332.50 | 4345.56 | 4332.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 13:15:00 | 4340.00 | 4344.44 | 4333.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 14:00:00 | 4340.00 | 4344.44 | 4333.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 14:15:00 | 4355.05 | 4346.57 | 4335.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 14:30:00 | 4334.85 | 4346.57 | 4335.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 15:15:00 | 4341.00 | 4345.45 | 4335.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-15 09:15:00 | 4424.95 | 4345.45 | 4335.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-15 13:30:00 | 4373.65 | 4364.36 | 4352.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 14:15:00 | 4361.75 | 4381.32 | 4368.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-22 09:15:00 | 4307.00 | 4389.69 | 4400.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 09:15:00 | 4307.00 | 4389.69 | 4400.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 14:15:00 | 4294.95 | 4315.94 | 4339.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 11:15:00 | 4329.05 | 4313.17 | 4330.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 11:15:00 | 4329.05 | 4313.17 | 4330.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 4329.05 | 4313.17 | 4330.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 11:45:00 | 4336.35 | 4313.17 | 4330.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 12:15:00 | 4335.65 | 4317.67 | 4330.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 13:15:00 | 4354.80 | 4317.67 | 4330.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 13:15:00 | 4400.20 | 4334.17 | 4337.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 13:45:00 | 4401.25 | 4334.17 | 4337.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 14:15:00 | 4410.00 | 4349.34 | 4343.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 15:15:00 | 4450.00 | 4369.47 | 4353.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-27 09:15:00 | 4330.00 | 4361.58 | 4351.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-27 09:15:00 | 4330.00 | 4361.58 | 4351.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 4330.00 | 4361.58 | 4351.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:00:00 | 4330.00 | 4361.58 | 4351.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 10:15:00 | 4352.05 | 4359.67 | 4351.27 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 09:15:00 | 4305.80 | 4348.98 | 4349.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 11:15:00 | 4261.15 | 4323.59 | 4337.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 14:15:00 | 4210.05 | 4185.51 | 4222.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-30 15:00:00 | 4210.05 | 4185.51 | 4222.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 15:15:00 | 4155.00 | 4179.41 | 4216.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 09:30:00 | 4141.00 | 4171.53 | 4209.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 4242.80 | 4196.31 | 4202.09 | SL hit (close>static) qty=1.00 sl=4240.10 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 4220.30 | 4207.19 | 4206.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 13:15:00 | 4270.00 | 4219.75 | 4211.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 09:15:00 | 4215.40 | 4240.75 | 4225.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 4215.40 | 4240.75 | 4225.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 4215.40 | 4240.75 | 4225.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:30:00 | 4163.40 | 4240.75 | 4225.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 4144.70 | 4221.54 | 4217.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 4144.70 | 4221.54 | 4217.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 4125.00 | 4202.23 | 4209.44 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 09:15:00 | 4292.80 | 4195.81 | 4190.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 10:15:00 | 4311.10 | 4218.87 | 4201.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 09:15:00 | 4412.55 | 4418.00 | 4370.27 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 12:00:00 | 4472.80 | 4430.25 | 4384.20 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 4408.20 | 4437.03 | 4399.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 15:00:00 | 4408.20 | 4437.03 | 4399.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 15:15:00 | 4421.00 | 4433.82 | 4401.75 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-12 09:15:00 | 4398.90 | 4426.84 | 4401.49 | SL hit (close<ema400) qty=1.00 sl=4401.49 alert=retest1 |

### Cycle 8 — SELL (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 10:15:00 | 5564.00 | 5667.18 | 5676.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 09:15:00 | 5414.70 | 5544.90 | 5603.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 11:15:00 | 5413.05 | 5354.25 | 5408.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 11:15:00 | 5413.05 | 5354.25 | 5408.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 5413.05 | 5354.25 | 5408.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 11:45:00 | 5407.90 | 5354.25 | 5408.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 5347.15 | 5352.83 | 5402.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 15:15:00 | 5334.80 | 5358.41 | 5397.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 09:30:00 | 5317.20 | 5347.95 | 5385.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 09:30:00 | 5333.15 | 5280.40 | 5299.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 10:15:00 | 5341.15 | 5280.40 | 5299.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 5350.65 | 5294.45 | 5304.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 11:00:00 | 5350.65 | 5294.45 | 5304.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 5333.80 | 5295.13 | 5300.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 15:00:00 | 5333.80 | 5295.13 | 5300.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 5336.00 | 5303.31 | 5303.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 5333.05 | 5303.31 | 5303.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 09:15:00 | 5098.45 | 5204.82 | 5245.59 | EMA400 retest candle locked (from downside) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 10:15:00 | 5074.09 | 5182.86 | 5231.90 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-22 09:15:00 | 5068.06 | 5155.01 | 5192.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-22 09:15:00 | 5218.55 | 5155.01 | 5192.19 | SL hit (close>static) qty=0.50 sl=5155.01 alert=retest2 |

### Cycle 9 — BUY (started 2024-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 15:15:00 | 5275.00 | 5219.49 | 5212.77 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-23 12:15:00 | 5173.15 | 5210.20 | 5210.83 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 14:15:00 | 5236.45 | 5214.66 | 5212.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 09:15:00 | 5277.95 | 5232.97 | 5221.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 5207.25 | 5273.57 | 5253.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 09:15:00 | 5207.25 | 5273.57 | 5253.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 5207.25 | 5273.57 | 5253.83 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2024-07-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 13:15:00 | 5227.95 | 5240.50 | 5241.88 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 11:15:00 | 5280.05 | 5241.18 | 5239.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 13:15:00 | 5319.15 | 5263.38 | 5250.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 09:15:00 | 5523.00 | 5535.07 | 5469.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-01 09:30:00 | 5555.55 | 5535.07 | 5469.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 5514.40 | 5530.93 | 5473.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 15:00:00 | 5539.00 | 5522.63 | 5487.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-02 09:15:00 | 5466.00 | 5511.12 | 5488.03 | SL hit (close<static) qty=1.00 sl=5469.35 alert=retest2 |

### Cycle 14 — SELL (started 2024-08-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 13:15:00 | 5456.75 | 5477.14 | 5477.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 14:15:00 | 5356.70 | 5453.05 | 5466.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 5292.45 | 5245.10 | 5318.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 5292.45 | 5245.10 | 5318.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 5292.45 | 5245.10 | 5318.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:45:00 | 5332.00 | 5245.10 | 5318.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 5201.00 | 5230.22 | 5276.47 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2024-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-08 15:15:00 | 5300.00 | 5251.31 | 5250.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 09:15:00 | 5353.70 | 5271.79 | 5259.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-09 12:15:00 | 5285.00 | 5292.25 | 5273.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-09 12:45:00 | 5280.00 | 5292.25 | 5273.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 5271.75 | 5288.15 | 5273.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 13:45:00 | 5260.05 | 5288.15 | 5273.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 5280.00 | 5286.52 | 5274.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-09 14:30:00 | 5271.15 | 5286.52 | 5274.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 5260.00 | 5281.22 | 5272.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-12 09:15:00 | 5359.10 | 5281.22 | 5272.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 5336.60 | 5292.29 | 5278.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-13 09:15:00 | 5477.65 | 5338.53 | 5309.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 12:15:00 | 5287.90 | 5325.37 | 5328.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 12:15:00 | 5287.90 | 5325.37 | 5328.64 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 14:15:00 | 5354.70 | 5331.67 | 5330.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 09:15:00 | 5476.05 | 5365.08 | 5346.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-16 10:15:00 | 5340.65 | 5360.19 | 5346.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 10:15:00 | 5340.65 | 5360.19 | 5346.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 5340.65 | 5360.19 | 5346.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 11:00:00 | 5340.65 | 5360.19 | 5346.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 5321.55 | 5352.47 | 5343.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-16 11:45:00 | 5309.70 | 5352.47 | 5343.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 5420.00 | 5365.22 | 5351.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 15:00:00 | 5447.55 | 5381.68 | 5359.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 10:15:00 | 5441.50 | 5403.01 | 5374.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-19 14:30:00 | 5478.50 | 5431.07 | 5401.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 09:15:00 | 5504.50 | 5425.13 | 5401.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 09:15:00 | 5534.80 | 5447.07 | 5413.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 10:15:00 | 5586.95 | 5447.07 | 5413.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 12:00:00 | 5595.00 | 5495.20 | 5442.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 14:45:00 | 5579.05 | 5516.29 | 5466.42 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-20 15:15:00 | 5577.90 | 5516.29 | 5466.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 5553.65 | 5551.15 | 5521.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 09:45:00 | 5579.10 | 5551.15 | 5521.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 5560.05 | 5577.86 | 5551.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 09:15:00 | 5622.00 | 5577.86 | 5551.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-08-23 15:15:00 | 5992.31 | 5772.51 | 5673.97 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 11:15:00 | 6023.55 | 6099.84 | 6100.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 12:15:00 | 6006.35 | 6081.14 | 6091.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 15:15:00 | 6137.00 | 6042.86 | 6052.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 15:15:00 | 6137.00 | 6042.86 | 6052.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 6137.00 | 6042.86 | 6052.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 10:30:00 | 5987.05 | 6010.34 | 6036.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-04 13:15:00 | 6009.15 | 5942.22 | 5934.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 13:15:00 | 6009.15 | 5942.22 | 5934.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 09:15:00 | 6316.80 | 6044.29 | 5985.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-06 12:15:00 | 6304.15 | 6312.47 | 6208.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-06 12:30:00 | 6306.00 | 6312.47 | 6208.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 14:15:00 | 6239.00 | 6296.07 | 6219.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 14:45:00 | 6225.00 | 6296.07 | 6219.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 6090.50 | 6254.62 | 6213.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:30:00 | 6118.00 | 6254.62 | 6213.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 6120.55 | 6227.81 | 6205.16 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2024-09-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 12:15:00 | 5997.95 | 6155.97 | 6174.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-10 11:15:00 | 5972.55 | 6066.60 | 6115.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 09:15:00 | 6046.30 | 5978.35 | 6012.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 09:15:00 | 6046.30 | 5978.35 | 6012.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 6046.30 | 5978.35 | 6012.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 09:45:00 | 6088.00 | 5978.35 | 6012.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 10:15:00 | 6028.50 | 5988.38 | 6014.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 11:15:00 | 6047.50 | 5988.38 | 6014.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 11:15:00 | 6093.00 | 6009.31 | 6021.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 12:00:00 | 6093.00 | 6009.31 | 6021.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2024-09-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 14:15:00 | 6131.40 | 6043.07 | 6034.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 14:15:00 | 6310.90 | 6146.78 | 6096.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 6328.15 | 6332.97 | 6247.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 10:00:00 | 6328.15 | 6332.97 | 6247.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 6420.90 | 6341.53 | 6282.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 14:45:00 | 6245.00 | 6341.53 | 6282.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 6340.00 | 6350.91 | 6308.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 15:00:00 | 6433.95 | 6355.40 | 6319.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-19 09:15:00 | 6279.00 | 6341.80 | 6319.79 | SL hit (close<static) qty=1.00 sl=6300.05 alert=retest2 |

### Cycle 22 — SELL (started 2024-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 11:15:00 | 6228.90 | 6306.13 | 6306.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 13:15:00 | 6167.30 | 6265.38 | 6287.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 12:15:00 | 6210.00 | 6205.59 | 6240.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-20 13:00:00 | 6210.00 | 6205.59 | 6240.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 23 — BUY (started 2024-09-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 14:15:00 | 6498.75 | 6267.33 | 6262.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 15:15:00 | 6542.00 | 6474.77 | 6411.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 15:15:00 | 6600.50 | 6617.17 | 6533.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-26 09:15:00 | 6560.65 | 6617.17 | 6533.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 6742.00 | 6642.14 | 6552.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:30:00 | 6551.85 | 6642.14 | 6552.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 6570.50 | 6667.87 | 6605.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 6570.50 | 6667.87 | 6605.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 6589.00 | 6652.10 | 6603.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 09:15:00 | 6614.90 | 6652.10 | 6603.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 09:15:00 | 6539.70 | 6629.62 | 6598.00 | SL hit (close<static) qty=1.00 sl=6550.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-09-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 12:15:00 | 6470.90 | 6562.54 | 6572.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 14:15:00 | 6399.95 | 6516.21 | 6548.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 09:15:00 | 6501.00 | 6497.76 | 6533.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-30 10:00:00 | 6501.00 | 6497.76 | 6533.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 6523.20 | 6479.76 | 6502.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:45:00 | 6530.00 | 6479.76 | 6502.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 6550.15 | 6493.84 | 6506.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 14:30:00 | 6485.70 | 6512.63 | 6512.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 6463.10 | 6511.50 | 6512.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 10:45:00 | 6499.60 | 6502.29 | 6507.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 11:30:00 | 6448.65 | 6490.78 | 6501.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 6161.41 | 6373.14 | 6434.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 6139.94 | 6373.14 | 6434.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 6174.62 | 6373.14 | 6434.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 6363.45 | 6304.71 | 6370.67 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-10-04 14:15:00 | 6363.45 | 6304.71 | 6370.67 | SL hit (close>ema200) qty=0.50 sl=6304.71 alert=retest2 |

### Cycle 25 — BUY (started 2024-10-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 11:15:00 | 6265.55 | 6230.12 | 6226.20 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 10:15:00 | 6195.90 | 6220.65 | 6223.86 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2024-10-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 15:15:00 | 6244.00 | 6217.73 | 6217.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 11:15:00 | 6262.00 | 6228.60 | 6222.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-14 15:15:00 | 6240.00 | 6242.58 | 6231.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 15:15:00 | 6240.00 | 6242.58 | 6231.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 15:15:00 | 6240.00 | 6242.58 | 6231.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-15 09:15:00 | 6300.40 | 6242.58 | 6231.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 10:15:00 | 6192.75 | 6233.62 | 6229.68 | SL hit (close<static) qty=1.00 sl=6231.50 alert=retest2 |

### Cycle 28 — SELL (started 2024-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 11:15:00 | 6197.00 | 6226.30 | 6226.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-15 12:15:00 | 6178.05 | 6216.65 | 6222.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 14:15:00 | 6126.90 | 6096.14 | 6132.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 14:15:00 | 6126.90 | 6096.14 | 6132.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 6126.90 | 6096.14 | 6132.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 15:00:00 | 6126.90 | 6096.14 | 6132.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 15:15:00 | 6095.00 | 6095.91 | 6128.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 09:15:00 | 6065.00 | 6095.91 | 6128.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 11:30:00 | 6058.55 | 6071.42 | 6108.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:15:00 | 5761.75 | 5915.40 | 5980.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 09:15:00 | 5755.62 | 5915.40 | 5980.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-22 14:15:00 | 5458.50 | 5686.17 | 5830.33 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 14:15:00 | 5099.95 | 5050.61 | 5049.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 5164.00 | 5077.99 | 5062.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 5019.60 | 5088.88 | 5071.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 5019.60 | 5088.88 | 5071.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 5019.60 | 5088.88 | 5071.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 09:45:00 | 5009.60 | 5088.88 | 5071.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 5010.70 | 5073.24 | 5066.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 5010.70 | 5073.24 | 5066.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2024-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 11:15:00 | 5012.05 | 5061.00 | 5061.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 09:15:00 | 5007.50 | 5031.32 | 5044.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 14:15:00 | 5002.00 | 4996.28 | 5018.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 14:15:00 | 5002.00 | 4996.28 | 5018.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 5002.00 | 4996.28 | 5018.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 5002.00 | 4996.28 | 5018.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 5009.15 | 4998.86 | 5017.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:15:00 | 5072.30 | 4998.86 | 5017.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 5045.20 | 5008.13 | 5020.44 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2024-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 12:15:00 | 5072.55 | 5033.07 | 5029.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 09:15:00 | 5129.75 | 5053.36 | 5040.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 13:15:00 | 5063.00 | 5069.33 | 5053.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 14:00:00 | 5063.00 | 5069.33 | 5053.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 14:15:00 | 5029.40 | 5061.35 | 5051.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 15:00:00 | 5029.40 | 5061.35 | 5051.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 5011.00 | 5051.28 | 5047.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 5039.65 | 5051.28 | 5047.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 09:15:00 | 4980.60 | 5037.14 | 5041.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 15:15:00 | 4940.00 | 4963.22 | 4987.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-12 09:15:00 | 5017.60 | 4974.09 | 4990.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-12 09:15:00 | 5017.60 | 4974.09 | 4990.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 5017.60 | 4974.09 | 4990.15 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2024-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 11:15:00 | 5049.10 | 5001.24 | 5000.46 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 12:15:00 | 4991.45 | 4999.28 | 4999.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 14:15:00 | 4982.00 | 4994.40 | 4997.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 4810.00 | 4771.79 | 4851.93 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-14 11:30:00 | 4690.00 | 4746.60 | 4826.05 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-14 13:30:00 | 4684.80 | 4728.20 | 4803.44 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-18 09:15:00 | 4645.00 | 4720.43 | 4786.43 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 4715.55 | 4719.46 | 4779.99 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-19 09:15:00 | 4897.75 | 4725.16 | 4747.08 | SL hit (close>ema400) qty=1.00 sl=4747.08 alert=retest1 |

### Cycle 35 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 4906.40 | 4781.35 | 4770.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 12:15:00 | 4987.75 | 4822.63 | 4789.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 09:15:00 | 4854.45 | 4894.87 | 4841.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 09:15:00 | 4854.45 | 4894.87 | 4841.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 4854.45 | 4894.87 | 4841.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:00:00 | 4854.45 | 4894.87 | 4841.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 4858.45 | 4887.58 | 4843.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 14:00:00 | 4913.95 | 4871.65 | 4845.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-27 10:15:00 | 4996.70 | 5071.53 | 5074.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-27 10:15:00 | 4996.70 | 5071.53 | 5074.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 13:15:00 | 4990.10 | 5034.04 | 5055.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-28 09:15:00 | 5050.00 | 5024.84 | 5044.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-28 09:15:00 | 5050.00 | 5024.84 | 5044.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 5050.00 | 5024.84 | 5044.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 10:30:00 | 4969.55 | 5014.75 | 5038.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 12:30:00 | 4969.00 | 5000.90 | 5027.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 14:00:00 | 4970.20 | 4994.76 | 5022.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 15:15:00 | 4955.05 | 4997.37 | 5021.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 4984.60 | 4988.04 | 5012.29 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-29 13:15:00 | 5084.85 | 5015.83 | 5017.39 | SL hit (close>static) qty=1.00 sl=5084.70 alert=retest2 |

### Cycle 37 — BUY (started 2024-11-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 14:15:00 | 5110.00 | 5034.66 | 5025.81 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 10:15:00 | 4958.55 | 5019.76 | 5027.46 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 09:15:00 | 5068.75 | 4944.91 | 4944.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 14:15:00 | 5097.30 | 5002.55 | 4975.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-11 13:15:00 | 4968.40 | 5026.22 | 5004.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 13:15:00 | 4968.40 | 5026.22 | 5004.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 13:15:00 | 4968.40 | 5026.22 | 5004.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 14:00:00 | 4968.40 | 5026.22 | 5004.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 4994.00 | 5019.78 | 5003.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 14:30:00 | 4992.20 | 5019.78 | 5003.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 15:15:00 | 4990.00 | 5013.82 | 5002.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-12 09:15:00 | 5062.85 | 5013.82 | 5002.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-12 15:15:00 | 4960.00 | 5003.32 | 5003.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 15:15:00 | 4960.00 | 5003.32 | 5003.96 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-12-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 09:15:00 | 5040.20 | 5010.70 | 5007.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-13 11:15:00 | 5131.60 | 5047.17 | 5025.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-16 10:15:00 | 5039.45 | 5068.29 | 5049.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 10:15:00 | 5039.45 | 5068.29 | 5049.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 5039.45 | 5068.29 | 5049.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:00:00 | 5039.45 | 5068.29 | 5049.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 5052.70 | 5065.17 | 5049.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 13:15:00 | 5074.90 | 5063.66 | 5050.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 14:00:00 | 5066.05 | 5064.14 | 5051.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-17 09:30:00 | 5072.00 | 5058.01 | 5051.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 09:15:00 | 5195.30 | 5124.02 | 5097.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 5236.65 | 5146.55 | 5110.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-18 11:15:00 | 5283.25 | 5173.35 | 5125.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-18 14:15:00 | 5582.39 | 5329.83 | 5219.69 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 5151.10 | 5292.73 | 5295.80 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 13:15:00 | 5348.85 | 5250.32 | 5240.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 14:15:00 | 5375.25 | 5275.31 | 5252.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 15:15:00 | 5352.10 | 5380.28 | 5333.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-30 09:15:00 | 5324.05 | 5380.28 | 5333.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 5310.95 | 5366.41 | 5331.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:00:00 | 5310.95 | 5366.41 | 5331.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 5277.85 | 5348.70 | 5326.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 11:00:00 | 5277.85 | 5348.70 | 5326.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 5322.10 | 5375.93 | 5349.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 5322.10 | 5375.93 | 5349.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 5341.00 | 5368.95 | 5348.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:15:00 | 5286.00 | 5368.95 | 5348.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 5278.00 | 5350.76 | 5341.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:00:00 | 5278.00 | 5350.76 | 5341.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 5295.40 | 5339.69 | 5337.70 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2024-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 11:15:00 | 5253.10 | 5322.37 | 5330.01 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2024-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 15:15:00 | 5375.00 | 5339.49 | 5335.19 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 15:15:00 | 5319.00 | 5335.81 | 5336.28 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-01-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 11:15:00 | 5336.35 | 5333.75 | 5333.59 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-01-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 12:15:00 | 5331.20 | 5333.24 | 5333.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 13:15:00 | 5325.00 | 5331.59 | 5332.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-03 14:15:00 | 5331.85 | 5331.64 | 5332.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 14:15:00 | 5331.85 | 5331.64 | 5332.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 5331.85 | 5331.64 | 5332.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 5331.85 | 5331.64 | 5332.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-06 09:15:00 | 5355.30 | 5335.31 | 5333.99 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 10:15:00 | 5283.75 | 5325.00 | 5329.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 5265.05 | 5313.01 | 5323.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 5260.50 | 5255.99 | 5286.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 5260.50 | 5255.99 | 5286.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 5260.50 | 5255.99 | 5286.76 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 11:15:00 | 5323.65 | 5288.71 | 5285.74 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 13:15:00 | 5241.85 | 5281.15 | 5282.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 14:15:00 | 5214.10 | 5267.74 | 5276.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 15:15:00 | 5212.00 | 5201.08 | 5219.21 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 09:15:00 | 5120.00 | 5201.08 | 5219.21 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-15 09:15:00 | 4864.00 | 4952.50 | 5022.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 13:15:00 | 4931.60 | 4869.73 | 4911.51 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-16 13:15:00 | 4931.60 | 4869.73 | 4911.51 | SL hit (close>ema200) qty=0.50 sl=4869.73 alert=retest1 |

### Cycle 53 — BUY (started 2025-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 13:15:00 | 4320.95 | 4269.07 | 4264.31 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 4230.35 | 4258.72 | 4262.31 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 09:15:00 | 4425.80 | 4283.60 | 4270.75 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 10:15:00 | 4174.10 | 4304.64 | 4321.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 11:15:00 | 4152.90 | 4274.29 | 4306.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 14:15:00 | 4119.00 | 4112.67 | 4179.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 15:00:00 | 4119.00 | 4112.67 | 4179.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 4017.30 | 4071.43 | 4116.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 09:15:00 | 3917.25 | 4027.96 | 4071.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 3943.35 | 4014.73 | 4026.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-17 14:15:00 | 4055.80 | 3977.84 | 3975.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-02-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 14:15:00 | 4055.80 | 3977.84 | 3975.88 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 09:15:00 | 3928.40 | 3974.54 | 3975.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 10:15:00 | 3906.00 | 3960.83 | 3968.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-18 14:15:00 | 3949.00 | 3938.05 | 3953.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-18 14:15:00 | 3949.00 | 3938.05 | 3953.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 3949.00 | 3938.05 | 3953.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 15:00:00 | 3949.00 | 3938.05 | 3953.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 15:15:00 | 3932.50 | 3936.94 | 3951.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:30:00 | 4000.00 | 3951.98 | 3957.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 10:15:00 | 4045.15 | 3970.62 | 3965.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 12:15:00 | 4059.40 | 4001.85 | 3981.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 11:15:00 | 4044.00 | 4059.48 | 4025.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 11:45:00 | 4042.45 | 4059.48 | 4025.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 12:15:00 | 4022.05 | 4051.99 | 4025.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 13:00:00 | 4022.05 | 4051.99 | 4025.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 13:15:00 | 4040.35 | 4049.66 | 4026.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 13:30:00 | 4010.00 | 4049.66 | 4026.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 14:15:00 | 4030.05 | 4045.74 | 4026.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 15:00:00 | 4030.05 | 4045.74 | 4026.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 15:15:00 | 4022.05 | 4041.00 | 4026.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 09:15:00 | 4069.95 | 4041.00 | 4026.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-24 09:15:00 | 3997.55 | 4055.69 | 4049.04 | SL hit (close<static) qty=1.00 sl=4010.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 11:15:00 | 4026.80 | 4041.77 | 4043.36 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-24 12:15:00 | 4059.35 | 4045.29 | 4044.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-27 09:15:00 | 4110.00 | 4073.94 | 4061.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-05 09:15:00 | 4481.50 | 4504.41 | 4431.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-05 09:15:00 | 4481.50 | 4504.41 | 4431.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 4481.50 | 4504.41 | 4431.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-05 12:45:00 | 4613.20 | 4534.75 | 4463.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-05 14:15:00 | 5074.52 | 4688.43 | 4546.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 09:15:00 | 4840.05 | 4914.09 | 4921.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-12 11:15:00 | 4830.10 | 4895.83 | 4911.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 14:15:00 | 4900.95 | 4892.79 | 4906.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 14:15:00 | 4900.95 | 4892.79 | 4906.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 4900.95 | 4892.79 | 4906.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 09:15:00 | 4820.55 | 4891.83 | 4904.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-18 11:15:00 | 4906.55 | 4838.28 | 4830.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 11:15:00 | 4906.55 | 4838.28 | 4830.21 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-19 11:15:00 | 4786.55 | 4833.30 | 4837.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-19 14:15:00 | 4751.05 | 4801.33 | 4820.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-21 09:15:00 | 4737.20 | 4736.66 | 4765.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-21 10:15:00 | 4805.85 | 4750.50 | 4769.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 10:15:00 | 4805.85 | 4750.50 | 4769.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 11:00:00 | 4805.85 | 4750.50 | 4769.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 11:15:00 | 4820.80 | 4764.56 | 4774.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 12:00:00 | 4820.80 | 4764.56 | 4774.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 12:15:00 | 4796.55 | 4770.96 | 4776.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-21 13:30:00 | 4771.55 | 4773.02 | 4776.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-21 14:15:00 | 4818.15 | 4782.04 | 4780.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-03-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 14:15:00 | 4818.15 | 4782.04 | 4780.35 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-24 13:15:00 | 4775.25 | 4781.00 | 4781.26 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2025-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-24 14:15:00 | 4799.00 | 4784.60 | 4782.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-25 11:15:00 | 4822.80 | 4793.42 | 4787.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-26 15:15:00 | 4850.00 | 4882.62 | 4855.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 15:15:00 | 4850.00 | 4882.62 | 4855.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 15:15:00 | 4850.00 | 4882.62 | 4855.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:15:00 | 4810.10 | 4882.62 | 4855.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 09:15:00 | 4784.60 | 4863.02 | 4849.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 09:45:00 | 4763.00 | 4863.02 | 4849.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 10:15:00 | 4770.00 | 4844.41 | 4842.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-27 11:00:00 | 4770.00 | 4844.41 | 4842.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 11:15:00 | 4815.00 | 4838.53 | 4839.76 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 14:15:00 | 4880.70 | 4825.80 | 4825.43 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 10:15:00 | 4763.80 | 4816.28 | 4822.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 12:15:00 | 4709.15 | 4783.42 | 4805.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 15:15:00 | 4802.20 | 4769.78 | 4792.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 15:15:00 | 4802.20 | 4769.78 | 4792.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 15:15:00 | 4802.20 | 4769.78 | 4792.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 4688.00 | 4769.78 | 4792.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-03 11:30:00 | 4713.80 | 4749.36 | 4764.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-04-07 09:15:00 | 4219.20 | 4553.84 | 4635.83 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 4623.80 | 4402.10 | 4390.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 11:15:00 | 4650.30 | 4483.57 | 4431.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 14:15:00 | 4595.70 | 4595.80 | 4540.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 15:00:00 | 4595.70 | 4595.80 | 4540.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 13:15:00 | 4844.40 | 4824.94 | 4800.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 14:15:00 | 4906.00 | 4824.94 | 4800.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 4752.20 | 4829.18 | 4809.78 | SL hit (close<static) qty=1.00 sl=4795.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 4713.70 | 4787.46 | 4793.03 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 09:15:00 | 4888.20 | 4782.89 | 4780.42 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 11:15:00 | 4720.30 | 4804.97 | 4807.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 12:15:00 | 4709.00 | 4785.77 | 4798.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 13:15:00 | 4664.50 | 4653.03 | 4706.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-02 14:00:00 | 4664.50 | 4653.03 | 4706.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 15:15:00 | 4620.00 | 4648.92 | 4695.64 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2025-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 14:15:00 | 4802.60 | 4710.54 | 4708.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 15:15:00 | 4832.00 | 4734.83 | 4719.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 12:15:00 | 4734.10 | 4752.17 | 4734.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 12:15:00 | 4734.10 | 4752.17 | 4734.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 4734.10 | 4752.17 | 4734.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 12:30:00 | 4736.60 | 4752.17 | 4734.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 4728.60 | 4747.45 | 4733.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:15:00 | 4716.10 | 4747.45 | 4733.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 4754.70 | 4748.90 | 4735.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:30:00 | 4702.80 | 4748.90 | 4735.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 4749.90 | 4749.10 | 4736.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-07 09:15:00 | 4716.50 | 4749.10 | 4736.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2025-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 09:15:00 | 4642.10 | 4727.70 | 4728.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 14:15:00 | 4630.20 | 4675.67 | 4699.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 09:15:00 | 4615.50 | 4609.92 | 4646.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 09:15:00 | 4615.50 | 4609.92 | 4646.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 4615.50 | 4609.92 | 4646.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:30:00 | 4654.80 | 4609.92 | 4646.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 10:15:00 | 4577.90 | 4603.52 | 4640.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 10:30:00 | 4792.10 | 4603.52 | 4640.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 11:15:00 | 4684.60 | 4619.73 | 4644.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 12:00:00 | 4684.60 | 4619.73 | 4644.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 12:15:00 | 4793.50 | 4654.49 | 4657.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 12:45:00 | 4813.50 | 4654.49 | 4657.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2025-05-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 13:15:00 | 4824.90 | 4688.57 | 4672.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 5175.50 | 4840.43 | 4750.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 14:15:00 | 5339.10 | 5364.68 | 5273.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 15:00:00 | 5339.10 | 5364.68 | 5273.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 09:15:00 | 5273.00 | 5342.40 | 5279.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:00:00 | 5273.00 | 5342.40 | 5279.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 5284.20 | 5330.76 | 5279.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 11:30:00 | 5399.00 | 5322.01 | 5280.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 14:15:00 | 5249.30 | 5293.56 | 5276.40 | SL hit (close<static) qty=1.00 sl=5264.30 alert=retest2 |

### Cycle 78 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 5266.90 | 5301.14 | 5304.84 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 5430.10 | 5326.94 | 5316.23 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 14:15:00 | 5285.90 | 5310.47 | 5313.01 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 5388.00 | 5328.94 | 5321.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 10:15:00 | 5579.90 | 5379.13 | 5344.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 12:15:00 | 5521.40 | 5526.98 | 5464.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-23 13:00:00 | 5521.40 | 5526.98 | 5464.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 5477.00 | 5514.59 | 5474.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:30:00 | 5518.60 | 5512.18 | 5476.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 13:15:00 | 5430.00 | 5482.25 | 5473.00 | SL hit (close<static) qty=1.00 sl=5455.60 alert=retest2 |

### Cycle 82 — SELL (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 15:15:00 | 5609.90 | 5626.81 | 5628.31 | EMA200 below EMA400 |

### Cycle 83 — BUY (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 12:15:00 | 5638.00 | 5630.34 | 5629.58 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 13:15:00 | 5600.00 | 5624.27 | 5626.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 09:15:00 | 5500.50 | 5589.65 | 5609.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 10:15:00 | 5571.50 | 5533.29 | 5560.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 10:15:00 | 5571.50 | 5533.29 | 5560.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 5571.50 | 5533.29 | 5560.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 5571.50 | 5533.29 | 5560.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 5551.50 | 5536.93 | 5559.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:30:00 | 5570.00 | 5536.93 | 5559.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 5517.00 | 5534.90 | 5553.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:30:00 | 5550.00 | 5534.90 | 5553.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 5483.00 | 5518.94 | 5542.60 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 5682.50 | 5570.33 | 5555.11 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 11:15:00 | 5533.50 | 5569.84 | 5572.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 12:15:00 | 5500.00 | 5555.87 | 5565.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 5302.50 | 5290.36 | 5345.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 5302.50 | 5290.36 | 5345.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 5332.50 | 5277.06 | 5299.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 09:45:00 | 5251.50 | 5285.36 | 5295.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:45:00 | 5245.00 | 5266.61 | 5283.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 15:00:00 | 5247.50 | 5258.69 | 5276.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:15:00 | 5237.00 | 5254.74 | 5271.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 5233.00 | 5216.98 | 5240.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 15:00:00 | 5163.50 | 5194.87 | 5219.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 5268.50 | 5213.92 | 5222.40 | SL hit (close>static) qty=1.00 sl=5249.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 5278.00 | 5234.91 | 5230.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 12:15:00 | 5295.50 | 5255.85 | 5246.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 09:15:00 | 5429.00 | 5436.90 | 5382.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 10:00:00 | 5429.00 | 5436.90 | 5382.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 5414.00 | 5432.32 | 5385.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:30:00 | 5392.50 | 5432.32 | 5385.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 5833.00 | 5867.18 | 5839.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 5833.00 | 5867.18 | 5839.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 5790.00 | 5851.75 | 5834.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 5790.00 | 5851.75 | 5834.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 5867.00 | 5854.80 | 5837.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 5930.00 | 5845.59 | 5837.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 10:15:00 | 5985.00 | 5852.77 | 5841.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 13:15:00 | 5928.50 | 5931.26 | 5927.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 14:45:00 | 5937.00 | 5929.61 | 5927.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 5933.50 | 5931.89 | 5928.93 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-14 12:15:00 | 5910.00 | 5924.55 | 5926.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 12:15:00 | 5910.00 | 5924.55 | 5926.07 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 5942.00 | 5929.39 | 5928.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 6124.00 | 5971.53 | 5947.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 10:15:00 | 6361.00 | 6403.15 | 6300.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 10:30:00 | 6396.00 | 6403.15 | 6300.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 6453.00 | 6544.86 | 6499.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 6453.00 | 6544.86 | 6499.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 6400.50 | 6515.99 | 6490.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:00:00 | 6400.50 | 6515.99 | 6490.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 12:15:00 | 6400.00 | 6474.55 | 6474.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 13:15:00 | 6386.50 | 6456.94 | 6466.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 6440.00 | 6438.08 | 6454.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 6440.00 | 6438.08 | 6454.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 6440.00 | 6438.08 | 6454.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 10:30:00 | 6395.00 | 6416.17 | 6442.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 13:15:00 | 6075.25 | 6180.22 | 6240.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 14:15:00 | 6475.00 | 6239.17 | 6261.92 | SL hit (close>ema200) qty=0.50 sl=6239.17 alert=retest2 |

### Cycle 91 — BUY (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 15:15:00 | 6465.00 | 6284.34 | 6280.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 6815.00 | 6390.47 | 6328.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 11:15:00 | 6698.00 | 6698.92 | 6573.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 12:00:00 | 6698.00 | 6698.92 | 6573.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 6568.00 | 6662.07 | 6602.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 6568.00 | 6662.07 | 6602.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 6576.00 | 6644.86 | 6600.25 | EMA400 retest candle locked (from upside) |

### Cycle 92 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 6492.00 | 6581.00 | 6581.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 15:15:00 | 6425.00 | 6549.80 | 6567.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 6549.50 | 6534.28 | 6552.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 13:00:00 | 6549.50 | 6534.28 | 6552.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 6626.00 | 6552.63 | 6559.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 6626.00 | 6552.63 | 6559.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 6646.00 | 6571.30 | 6567.31 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 6486.00 | 6564.35 | 6565.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 10:15:00 | 6445.50 | 6540.58 | 6554.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 12:15:00 | 6374.00 | 6352.33 | 6399.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 12:15:00 | 6374.00 | 6352.33 | 6399.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 6374.00 | 6352.33 | 6399.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 12:45:00 | 6372.00 | 6352.33 | 6399.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 6403.50 | 6364.83 | 6397.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 6403.50 | 6364.83 | 6397.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 6423.50 | 6376.57 | 6399.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 6425.00 | 6376.57 | 6399.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 6445.50 | 6390.35 | 6403.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:45:00 | 6440.00 | 6390.35 | 6403.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 6439.50 | 6400.18 | 6406.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:15:00 | 6484.50 | 6400.18 | 6406.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — BUY (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 11:15:00 | 6517.50 | 6423.65 | 6417.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 09:15:00 | 6615.00 | 6492.85 | 6456.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 15:15:00 | 7033.00 | 7066.94 | 6970.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 09:15:00 | 6997.00 | 7066.94 | 6970.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 7014.00 | 7056.36 | 6974.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:45:00 | 6995.00 | 7056.36 | 6974.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 6917.00 | 7019.47 | 6971.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:00:00 | 6917.00 | 7019.47 | 6971.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 6850.50 | 6985.67 | 6960.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:30:00 | 6873.50 | 6985.67 | 6960.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 96 — SELL (started 2025-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 14:15:00 | 6867.50 | 6934.17 | 6939.68 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 15:15:00 | 6990.00 | 6945.34 | 6944.25 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 09:15:00 | 6805.00 | 6917.27 | 6931.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 11:15:00 | 6786.00 | 6826.76 | 6847.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 12:15:00 | 6880.50 | 6837.51 | 6850.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 12:15:00 | 6880.50 | 6837.51 | 6850.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 6880.50 | 6837.51 | 6850.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:00:00 | 6880.50 | 6837.51 | 6850.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 6856.50 | 6841.31 | 6850.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 14:15:00 | 6900.00 | 6841.31 | 6850.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — BUY (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 14:15:00 | 7162.50 | 6905.55 | 6879.03 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 12:15:00 | 6877.00 | 6922.91 | 6923.90 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 7027.50 | 6930.05 | 6922.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 7180.00 | 6980.04 | 6946.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 09:15:00 | 6985.50 | 7010.17 | 6967.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 6985.50 | 7010.17 | 6967.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 6985.50 | 7010.17 | 6967.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 6972.00 | 7010.17 | 6967.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 11:15:00 | 6993.00 | 7002.55 | 6971.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 11:30:00 | 6977.00 | 7002.55 | 6971.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 6972.00 | 6996.44 | 6971.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:00:00 | 6972.00 | 6996.44 | 6971.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 6935.00 | 6984.15 | 6968.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:45:00 | 6931.00 | 6984.15 | 6968.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 6985.00 | 6984.32 | 6969.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:15:00 | 6948.00 | 6984.32 | 6969.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 6948.00 | 6977.06 | 6967.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 6867.50 | 6977.06 | 6967.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 09:15:00 | 6869.00 | 6955.44 | 6958.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 6829.50 | 6883.29 | 6907.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 14:15:00 | 6896.50 | 6854.64 | 6875.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 14:15:00 | 6896.50 | 6854.64 | 6875.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 6896.50 | 6854.64 | 6875.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:45:00 | 6877.50 | 6854.64 | 6875.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 6859.50 | 6855.61 | 6873.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:15:00 | 6848.50 | 6855.61 | 6873.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 13:00:00 | 6842.00 | 6858.61 | 6870.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 14:15:00 | 6844.00 | 6857.39 | 6868.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 10:15:00 | 6802.50 | 6770.05 | 6766.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 10:15:00 | 6802.50 | 6770.05 | 6766.20 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 13:15:00 | 6700.00 | 6757.43 | 6761.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 09:15:00 | 6571.00 | 6702.37 | 6734.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 6700.00 | 6615.47 | 6659.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 6700.00 | 6615.47 | 6659.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 6700.00 | 6615.47 | 6659.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:00:00 | 6700.00 | 6615.47 | 6659.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 6680.50 | 6628.47 | 6661.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:30:00 | 6666.00 | 6640.78 | 6664.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 13:15:00 | 6713.00 | 6661.50 | 6669.93 | SL hit (close>static) qty=1.00 sl=6712.50 alert=retest2 |

### Cycle 105 — BUY (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 14:15:00 | 6752.50 | 6679.70 | 6677.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 6783.50 | 6711.63 | 6693.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 11:15:00 | 6766.50 | 6770.83 | 6744.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 12:00:00 | 6766.50 | 6770.83 | 6744.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 6799.50 | 6777.69 | 6757.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 10:45:00 | 6836.50 | 6782.36 | 6761.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 13:15:00 | 6734.50 | 6772.31 | 6762.07 | SL hit (close<static) qty=1.00 sl=6740.50 alert=retest2 |

### Cycle 106 — SELL (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 10:15:00 | 6746.50 | 6781.37 | 6781.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 12:15:00 | 6692.50 | 6756.74 | 6770.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 13:15:00 | 6798.50 | 6765.09 | 6772.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 13:15:00 | 6798.50 | 6765.09 | 6772.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 6798.50 | 6765.09 | 6772.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 14:00:00 | 6798.50 | 6765.09 | 6772.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 6829.00 | 6777.87 | 6777.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 15:00:00 | 6829.00 | 6777.87 | 6777.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — BUY (started 2025-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 15:15:00 | 6820.00 | 6786.30 | 6781.74 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 11:15:00 | 6757.50 | 6776.76 | 6778.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 6740.00 | 6767.64 | 6773.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 15:15:00 | 6778.00 | 6769.71 | 6773.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 15:15:00 | 6778.00 | 6769.71 | 6773.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 6778.00 | 6769.71 | 6773.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 6589.50 | 6769.71 | 6773.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 6623.00 | 6740.37 | 6760.13 | EMA400 retest candle locked (from downside) |

### Cycle 109 — BUY (started 2025-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 13:15:00 | 6815.00 | 6747.39 | 6739.52 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 15:15:00 | 6671.00 | 6728.21 | 6731.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 09:15:00 | 6626.50 | 6707.87 | 6722.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 14:15:00 | 6800.00 | 6691.67 | 6704.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 14:15:00 | 6800.00 | 6691.67 | 6704.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 6800.00 | 6691.67 | 6704.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 6800.00 | 6691.67 | 6704.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 6752.50 | 6703.83 | 6709.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 6712.50 | 6703.83 | 6709.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 6718.00 | 6696.17 | 6702.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:00:00 | 6718.00 | 6696.17 | 6702.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 6741.00 | 6705.13 | 6706.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 6741.00 | 6705.13 | 6706.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 6723.00 | 6708.28 | 6707.56 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 10:15:00 | 6694.00 | 6707.30 | 6707.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-03 12:15:00 | 6683.00 | 6700.55 | 6704.16 | Break + close below crossover candle low |

### Cycle 113 — BUY (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 13:15:00 | 6738.50 | 6708.14 | 6707.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 15:15:00 | 6798.50 | 6736.43 | 6720.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 15:15:00 | 6780.00 | 6785.96 | 6759.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 09:15:00 | 6775.00 | 6785.96 | 6759.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 6773.50 | 6783.47 | 6761.22 | EMA400 retest candle locked (from upside) |

### Cycle 114 — SELL (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 13:15:00 | 6708.00 | 6746.61 | 6748.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 14:15:00 | 6692.50 | 6735.79 | 6743.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 14:15:00 | 6563.00 | 6552.60 | 6591.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 14:45:00 | 6551.00 | 6552.60 | 6591.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 6506.00 | 6487.09 | 6523.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:30:00 | 6574.50 | 6487.09 | 6523.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 6462.00 | 6466.53 | 6498.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:45:00 | 6592.00 | 6466.53 | 6498.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 6465.00 | 6459.55 | 6486.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 6500.00 | 6459.55 | 6486.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 6477.50 | 6463.14 | 6485.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 6491.00 | 6463.14 | 6485.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 6441.00 | 6458.71 | 6481.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 14:00:00 | 6425.00 | 6451.97 | 6476.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 15:15:00 | 6409.00 | 6448.67 | 6472.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 6790.00 | 6510.59 | 6496.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 6790.00 | 6510.59 | 6496.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 11:15:00 | 6800.50 | 6611.92 | 6547.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 10:15:00 | 6677.50 | 6711.09 | 6638.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 11:00:00 | 6677.50 | 6711.09 | 6638.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 6668.50 | 6705.99 | 6671.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:00:00 | 6668.50 | 6705.99 | 6671.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 6674.00 | 6699.59 | 6671.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:30:00 | 6692.50 | 6699.59 | 6671.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 6670.00 | 6691.98 | 6676.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 6723.00 | 6695.58 | 6679.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 6683.00 | 6693.77 | 6681.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 6683.00 | 6693.77 | 6681.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 6676.00 | 6690.22 | 6681.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:00:00 | 6676.00 | 6690.22 | 6681.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 6677.00 | 6687.57 | 6680.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 12:30:00 | 6705.00 | 6691.76 | 6683.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 14:15:00 | 6614.00 | 6677.69 | 6678.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 6614.00 | 6677.69 | 6678.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 11:15:00 | 6592.00 | 6637.52 | 6657.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 6504.00 | 6500.13 | 6546.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 09:45:00 | 6513.50 | 6500.13 | 6546.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 6549.00 | 6509.91 | 6546.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:00:00 | 6549.00 | 6509.91 | 6546.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 6487.50 | 6505.43 | 6541.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 12:15:00 | 6480.00 | 6505.43 | 6541.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 13:30:00 | 6468.00 | 6487.87 | 6526.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:00:00 | 6434.00 | 6487.87 | 6526.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 15:00:00 | 6463.00 | 6441.58 | 6473.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 6452.00 | 6443.66 | 6471.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 6590.00 | 6443.66 | 6471.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 6550.00 | 6464.93 | 6478.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:30:00 | 6540.50 | 6464.93 | 6478.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-30 10:15:00 | 6583.00 | 6488.54 | 6488.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 10:15:00 | 6583.00 | 6488.54 | 6488.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 11:15:00 | 6584.50 | 6507.73 | 6496.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 6666.50 | 6682.86 | 6619.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 15:00:00 | 6666.50 | 6682.86 | 6619.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 6815.00 | 6840.01 | 6787.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:15:00 | 6798.50 | 6840.01 | 6787.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 6794.00 | 6830.81 | 6787.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:45:00 | 6773.50 | 6830.81 | 6787.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 6789.50 | 6822.55 | 6788.05 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 6681.50 | 6768.42 | 6773.01 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2025-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 15:15:00 | 6799.50 | 6769.97 | 6769.34 | EMA200 above EMA400 |

### Cycle 120 — SELL (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 09:15:00 | 6657.50 | 6747.48 | 6759.17 | EMA200 below EMA400 |

### Cycle 121 — BUY (started 2025-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 15:15:00 | 6793.50 | 6762.29 | 6758.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 09:15:00 | 7112.00 | 6832.23 | 6791.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 09:15:00 | 7031.50 | 7084.90 | 6973.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 09:30:00 | 7034.50 | 7084.90 | 6973.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 6927.50 | 7033.84 | 6999.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 6927.50 | 7033.84 | 6999.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 6880.50 | 7003.17 | 6989.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 6880.50 | 7003.17 | 6989.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 11:15:00 | 6867.50 | 6976.04 | 6978.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 15:15:00 | 6812.00 | 6898.99 | 6936.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 12:15:00 | 6800.00 | 6789.17 | 6833.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 13:00:00 | 6800.00 | 6789.17 | 6833.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 6691.00 | 6771.64 | 6812.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 10:30:00 | 6678.50 | 6754.31 | 6800.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 11:45:00 | 6678.00 | 6743.25 | 6791.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 11:00:00 | 6640.00 | 6715.58 | 6736.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:30:00 | 6588.50 | 6680.05 | 6710.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 6684.50 | 6658.04 | 6687.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 13:45:00 | 6684.00 | 6658.04 | 6687.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 6662.00 | 6658.83 | 6685.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:15:00 | 6759.00 | 6658.83 | 6685.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 6759.00 | 6678.87 | 6692.19 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 6942.50 | 6731.59 | 6714.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 123 — BUY (started 2025-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 09:15:00 | 6942.50 | 6731.59 | 6714.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 10:15:00 | 7065.00 | 6798.28 | 6746.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 7025.50 | 7058.16 | 6979.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 09:30:00 | 7020.00 | 7058.16 | 6979.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 6973.50 | 7041.23 | 6978.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 6973.50 | 7041.23 | 6978.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 6985.00 | 7029.98 | 6979.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 12:15:00 | 7042.00 | 7029.98 | 6979.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 11:45:00 | 7013.50 | 7036.50 | 7009.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:00:00 | 7005.00 | 7030.20 | 7008.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 12:30:00 | 6995.50 | 7018.57 | 7013.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 7048.00 | 7022.77 | 7015.97 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-02 10:15:00 | 6980.00 | 7011.32 | 7012.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 6980.00 | 7011.32 | 7012.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 6953.00 | 6999.66 | 7007.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 14:15:00 | 7069.50 | 6961.07 | 6970.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 14:15:00 | 7069.50 | 6961.07 | 6970.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 7069.50 | 6961.07 | 6970.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 7069.50 | 6961.07 | 6970.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 7011.00 | 6971.06 | 6974.51 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 7063.50 | 6989.55 | 6982.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 09:15:00 | 7098.50 | 7047.73 | 7021.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 11:15:00 | 7085.50 | 7090.50 | 7064.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 12:00:00 | 7085.50 | 7090.50 | 7064.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 6970.00 | 7066.40 | 7055.70 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 6968.50 | 7046.82 | 7047.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 14:15:00 | 6914.50 | 7020.36 | 7035.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 6941.50 | 6932.15 | 6981.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 6941.50 | 6932.15 | 6981.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 7024.50 | 6950.62 | 6985.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 7024.50 | 6950.62 | 6985.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 7029.00 | 6966.30 | 6989.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 7007.50 | 6966.30 | 6989.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 7059.00 | 7012.34 | 7007.50 | EMA200 above EMA400 |

### Cycle 128 — SELL (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 12:15:00 | 6975.50 | 7003.05 | 7004.43 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 13:15:00 | 7044.50 | 7011.34 | 7008.07 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 6935.00 | 6992.74 | 6999.97 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 7086.50 | 7013.49 | 7007.84 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 7042.50 | 7050.91 | 7051.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 6982.00 | 7035.59 | 7043.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 15:15:00 | 7047.00 | 7012.96 | 7025.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 15:15:00 | 7047.00 | 7012.96 | 7025.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 7047.00 | 7012.96 | 7025.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 6922.50 | 7012.96 | 7025.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 7076.50 | 7005.57 | 7007.82 | SL hit (close>static) qty=1.00 sl=7065.00 alert=retest2 |

### Cycle 133 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 7069.00 | 7018.26 | 7013.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 11:15:00 | 7103.00 | 7068.72 | 7046.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 14:15:00 | 7051.50 | 7077.20 | 7056.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 14:15:00 | 7051.50 | 7077.20 | 7056.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 7051.50 | 7077.20 | 7056.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 15:00:00 | 7051.50 | 7077.20 | 7056.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 7060.00 | 7073.76 | 7057.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:15:00 | 7076.50 | 7073.76 | 7057.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:45:00 | 7066.50 | 7073.11 | 7058.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 11:15:00 | 7098.00 | 7065.89 | 7056.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 7078.50 | 7079.46 | 7069.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 7085.00 | 7080.57 | 7070.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 13:45:00 | 7102.50 | 7098.18 | 7082.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 15:15:00 | 7002.50 | 7071.66 | 7072.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — SELL (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 15:15:00 | 7002.50 | 7071.66 | 7072.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 10:15:00 | 6959.50 | 7040.00 | 7057.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 12:15:00 | 7068.00 | 7032.80 | 7050.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 12:15:00 | 7068.00 | 7032.80 | 7050.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 7068.00 | 7032.80 | 7050.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 13:00:00 | 7068.00 | 7032.80 | 7050.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 13:15:00 | 7279.00 | 7082.04 | 7071.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 14:15:00 | 7392.00 | 7144.03 | 7100.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 10:15:00 | 7158.00 | 7189.10 | 7136.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 10:15:00 | 7158.00 | 7189.10 | 7136.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 7158.00 | 7189.10 | 7136.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:00:00 | 7158.00 | 7189.10 | 7136.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 7139.50 | 7179.18 | 7136.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 12:15:00 | 7122.00 | 7179.18 | 7136.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 7089.50 | 7161.25 | 7132.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 12:45:00 | 7099.50 | 7161.25 | 7132.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 7096.00 | 7148.20 | 7129.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:15:00 | 7080.50 | 7148.20 | 7129.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 7071.00 | 7132.76 | 7123.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:45:00 | 7097.00 | 7132.76 | 7123.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 15:15:00 | 7004.50 | 7107.11 | 7112.89 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 10:15:00 | 7138.00 | 7119.11 | 7117.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 10:15:00 | 7545.00 | 7236.74 | 7177.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 7466.00 | 7524.24 | 7379.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 10:00:00 | 7466.00 | 7524.24 | 7379.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 7991.50 | 7998.54 | 7931.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:30:00 | 7977.50 | 7998.54 | 7931.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 7809.00 | 7960.63 | 7919.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 7809.00 | 7960.63 | 7919.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 7839.00 | 7936.31 | 7912.58 | EMA400 retest candle locked (from upside) |

### Cycle 138 — SELL (started 2026-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 14:15:00 | 7834.00 | 7890.19 | 7895.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 15:15:00 | 7788.50 | 7869.85 | 7885.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 7881.50 | 7872.18 | 7885.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 7881.50 | 7872.18 | 7885.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 7881.50 | 7872.18 | 7885.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 7881.50 | 7872.18 | 7885.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 7735.50 | 7844.84 | 7871.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:15:00 | 7703.00 | 7844.84 | 7871.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 12:45:00 | 7721.00 | 7800.50 | 7845.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 13:15:00 | 7709.00 | 7800.50 | 7845.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 14:00:00 | 7716.00 | 7783.60 | 7833.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 7770.00 | 7745.19 | 7800.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:00:00 | 7770.00 | 7745.19 | 7800.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 7745.00 | 7745.15 | 7795.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:30:00 | 7804.00 | 7745.15 | 7795.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 7334.95 | 7556.93 | 7660.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 12:15:00 | 7566.00 | 7558.74 | 7652.32 | SL hit (close>ema200) qty=0.50 sl=7558.74 alert=retest2 |

### Cycle 139 — BUY (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 09:15:00 | 7754.00 | 7616.38 | 7613.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 7814.50 | 7695.21 | 7654.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 7756.50 | 7789.63 | 7739.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 14:00:00 | 7756.50 | 7789.63 | 7739.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 7739.50 | 7776.46 | 7742.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 7674.50 | 7776.46 | 7742.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 7725.00 | 7766.17 | 7740.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 7684.00 | 7766.17 | 7740.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 7674.00 | 7747.74 | 7734.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:30:00 | 7669.50 | 7747.74 | 7734.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 7678.50 | 7733.89 | 7729.51 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 7633.50 | 7713.81 | 7720.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 13:15:00 | 7601.00 | 7691.25 | 7709.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 11:15:00 | 7439.50 | 7425.26 | 7521.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 12:00:00 | 7439.50 | 7425.26 | 7521.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 7582.00 | 7442.78 | 7491.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 7582.00 | 7442.78 | 7491.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 7505.00 | 7455.22 | 7492.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 7488.00 | 7455.22 | 7492.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 7616.50 | 7519.50 | 7513.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 7616.50 | 7519.50 | 7513.64 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 7415.50 | 7497.31 | 7506.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 7361.50 | 7470.15 | 7493.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 7420.00 | 7367.89 | 7428.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 7420.00 | 7367.89 | 7428.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 7420.00 | 7367.89 | 7428.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:45:00 | 7433.00 | 7367.89 | 7428.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 7554.00 | 7405.11 | 7440.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 11:00:00 | 7554.00 | 7405.11 | 7440.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 7541.50 | 7432.39 | 7449.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 11:30:00 | 7558.00 | 7432.39 | 7449.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — BUY (started 2026-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 13:15:00 | 7544.00 | 7470.01 | 7464.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 14:15:00 | 7568.00 | 7489.61 | 7473.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 7492.50 | 7657.87 | 7598.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 7492.50 | 7657.87 | 7598.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 7492.50 | 7657.87 | 7598.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:30:00 | 7522.00 | 7657.87 | 7598.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 7357.50 | 7597.79 | 7576.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:30:00 | 7349.50 | 7597.79 | 7576.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 7255.50 | 7529.33 | 7547.34 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 7512.00 | 7401.14 | 7390.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 7606.50 | 7459.23 | 7419.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 7761.50 | 7779.75 | 7684.77 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 11:15:00 | 7828.00 | 7784.80 | 7695.70 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 11:45:00 | 7822.00 | 7793.34 | 7707.68 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 13:45:00 | 7823.50 | 7802.24 | 7726.76 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 7805.00 | 7812.55 | 7750.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:30:00 | 7753.00 | 7812.55 | 7750.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 7755.50 | 7807.73 | 7764.65 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-06 12:15:00 | 7755.50 | 7807.73 | 7764.65 | SL hit (close<ema400) qty=1.00 sl=7764.65 alert=retest1 |

### Cycle 146 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 7734.00 | 7950.23 | 7954.36 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 11:15:00 | 7937.00 | 7815.08 | 7805.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 09:15:00 | 8085.00 | 7948.11 | 7881.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 09:15:00 | 7961.00 | 8015.00 | 7958.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 09:15:00 | 7961.00 | 8015.00 | 7958.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 7961.00 | 8015.00 | 7958.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:30:00 | 7984.00 | 8015.00 | 7958.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 8052.50 | 8022.50 | 7967.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:45:00 | 8099.50 | 8046.15 | 8003.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 09:15:00 | 7909.00 | 7987.20 | 7990.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 148 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 7909.00 | 7987.20 | 7990.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 09:15:00 | 7730.00 | 7874.05 | 7924.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 14:15:00 | 7799.50 | 7797.95 | 7861.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 15:00:00 | 7799.50 | 7797.95 | 7861.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 7859.00 | 7805.69 | 7853.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:00:00 | 7859.00 | 7805.69 | 7853.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 7767.50 | 7798.05 | 7845.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:15:00 | 7750.50 | 7798.05 | 7845.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 7362.97 | 7548.53 | 7647.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 10:15:00 | 7634.50 | 7565.72 | 7646.18 | SL hit (close>ema200) qty=0.50 sl=7565.72 alert=retest2 |

### Cycle 149 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 6849.50 | 6759.23 | 6747.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 7003.50 | 6808.08 | 6770.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 6915.50 | 6971.06 | 6895.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 6915.50 | 6971.06 | 6895.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 6915.50 | 6971.06 | 6895.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:45:00 | 6900.50 | 6971.06 | 6895.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 6950.50 | 6966.95 | 6900.13 | EMA400 retest candle locked (from upside) |

### Cycle 150 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 6749.50 | 6875.50 | 6885.45 | EMA200 below EMA400 |

### Cycle 151 — BUY (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 13:15:00 | 6987.00 | 6907.61 | 6897.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 09:15:00 | 7069.50 | 6935.19 | 6912.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-01 11:15:00 | 6913.00 | 6940.96 | 6920.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 11:15:00 | 6913.00 | 6940.96 | 6920.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 6913.00 | 6940.96 | 6920.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:45:00 | 6889.50 | 6940.96 | 6920.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 7030.00 | 6958.77 | 6930.08 | EMA400 retest candle locked (from upside) |

### Cycle 152 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 6767.50 | 6889.65 | 6904.61 | EMA200 below EMA400 |

### Cycle 153 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 7023.00 | 6823.16 | 6796.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 12:15:00 | 7269.50 | 6951.04 | 6861.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 7369.00 | 7422.33 | 7326.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 7369.00 | 7422.33 | 7326.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 7369.00 | 7422.33 | 7326.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 7390.50 | 7415.56 | 7332.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:15:00 | 7405.50 | 7415.56 | 7332.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 7466.00 | 7408.10 | 7360.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 14:15:00 | 7411.50 | 7414.54 | 7383.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 7431.50 | 7417.93 | 7388.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 15:15:00 | 7450.00 | 7417.93 | 7388.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 10:15:00 | 7611.50 | 7681.58 | 7686.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 154 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 7611.50 | 7681.58 | 7686.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 7595.50 | 7664.36 | 7677.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 7686.00 | 7618.94 | 7644.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 7686.00 | 7618.94 | 7644.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 7686.00 | 7618.94 | 7644.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 7686.00 | 7618.94 | 7644.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 7593.00 | 7613.75 | 7640.23 | EMA400 retest candle locked (from downside) |

### Cycle 155 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 7709.00 | 7653.93 | 7653.69 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 14:15:00 | 7621.50 | 7647.44 | 7650.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 09:15:00 | 7557.00 | 7623.84 | 7639.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-28 15:15:00 | 7596.00 | 7582.71 | 7606.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-29 09:15:00 | 7596.00 | 7582.71 | 7606.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 7583.50 | 7582.86 | 7604.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 10:30:00 | 7553.50 | 7583.99 | 7602.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 11:15:00 | 7644.50 | 7596.09 | 7606.61 | SL hit (close>static) qty=1.00 sl=7636.00 alert=retest2 |

### Cycle 157 — BUY (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 12:15:00 | 7719.50 | 7620.77 | 7616.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 13:15:00 | 7763.00 | 7649.22 | 7630.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 10:15:00 | 7679.50 | 7694.93 | 7661.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 10:15:00 | 7679.50 | 7694.93 | 7661.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 7679.50 | 7694.93 | 7661.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:00:00 | 7679.50 | 7694.93 | 7661.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 7675.00 | 7690.94 | 7662.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 7785.00 | 7674.08 | 7662.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 10:15:00 | 7616.50 | 7672.61 | 7676.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 7616.50 | 7672.61 | 7676.89 | EMA200 below EMA400 |

### Cycle 159 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 7743.50 | 7691.81 | 7685.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 7812.00 | 7736.77 | 7710.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 10:15:00 | 7710.50 | 7745.50 | 7731.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 10:15:00 | 7710.50 | 7745.50 | 7731.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 7710.50 | 7745.50 | 7731.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 10:30:00 | 7722.00 | 7745.50 | 7731.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 7807.50 | 7757.90 | 7738.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:30:00 | 7704.50 | 7757.90 | 7738.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 8348.00 | 7875.92 | 7793.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 15:15:00 | 8777.00 | 8162.19 | 7946.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-08 10:15:00 | 9654.70 | 8688.18 | 8261.93 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-05-13 09:30:00 | 4298.80 | 2024-05-13 15:15:00 | 4350.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2024-05-13 11:45:00 | 4298.00 | 2024-05-13 15:15:00 | 4350.00 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2024-05-15 09:15:00 | 4424.95 | 2024-05-22 09:15:00 | 4307.00 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2024-05-15 13:30:00 | 4373.65 | 2024-05-22 09:15:00 | 4307.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2024-05-16 14:15:00 | 4361.75 | 2024-05-22 09:15:00 | 4307.00 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-05-31 09:30:00 | 4141.00 | 2024-06-03 09:15:00 | 4242.80 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest1 | 2024-06-11 12:00:00 | 4472.80 | 2024-06-12 09:15:00 | 4398.90 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-06-14 09:15:00 | 4539.90 | 2024-06-21 09:15:00 | 4993.89 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-11 15:15:00 | 5334.80 | 2024-07-19 10:15:00 | 5074.09 | PARTIAL | 0.50 | 4.89% |
| SELL | retest2 | 2024-07-12 09:30:00 | 5317.20 | 2024-07-22 09:15:00 | 5068.06 | PARTIAL | 0.50 | 4.69% |
| SELL | retest2 | 2024-07-11 15:15:00 | 5334.80 | 2024-07-22 09:15:00 | 5218.55 | STOP_HIT | 0.50 | 2.18% |
| SELL | retest2 | 2024-07-16 09:30:00 | 5333.15 | 2024-07-22 09:15:00 | 5051.34 | PARTIAL | 0.50 | 5.28% |
| SELL | retest2 | 2024-07-12 09:30:00 | 5317.20 | 2024-07-22 09:15:00 | 5218.55 | STOP_HIT | 0.50 | 1.86% |
| SELL | retest2 | 2024-07-16 10:15:00 | 5341.15 | 2024-07-22 09:15:00 | 5066.49 | PARTIAL | 0.50 | 5.14% |
| SELL | retest2 | 2024-07-16 09:30:00 | 5333.15 | 2024-07-22 09:15:00 | 5218.55 | STOP_HIT | 0.50 | 2.15% |
| SELL | retest2 | 2024-07-16 10:15:00 | 5341.15 | 2024-07-22 09:15:00 | 5218.55 | STOP_HIT | 0.50 | 2.30% |
| BUY | retest2 | 2024-08-01 15:00:00 | 5539.00 | 2024-08-02 09:15:00 | 5466.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-08-13 09:15:00 | 5477.65 | 2024-08-14 12:15:00 | 5287.90 | STOP_HIT | 1.00 | -3.46% |
| BUY | retest2 | 2024-08-16 15:00:00 | 5447.55 | 2024-08-23 15:15:00 | 5992.31 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-19 10:15:00 | 5441.50 | 2024-08-23 15:15:00 | 5985.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-19 14:30:00 | 5478.50 | 2024-08-23 15:15:00 | 6026.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-20 09:15:00 | 5504.50 | 2024-08-26 09:15:00 | 6054.95 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-20 10:15:00 | 5586.95 | 2024-08-26 09:15:00 | 6145.65 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-20 12:00:00 | 5595.00 | 2024-08-26 09:15:00 | 6154.50 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-20 14:45:00 | 5579.05 | 2024-08-26 09:15:00 | 6136.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-20 15:15:00 | 5577.90 | 2024-08-26 09:15:00 | 6135.69 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-23 09:15:00 | 5622.00 | 2024-08-26 09:15:00 | 6184.20 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-09-02 10:30:00 | 5987.05 | 2024-09-04 13:15:00 | 6009.15 | STOP_HIT | 1.00 | -0.37% |
| BUY | retest2 | 2024-09-18 15:00:00 | 6433.95 | 2024-09-19 09:15:00 | 6279.00 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2024-09-27 09:15:00 | 6614.90 | 2024-09-27 09:15:00 | 6539.70 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2024-10-01 14:30:00 | 6485.70 | 2024-10-04 09:15:00 | 6161.41 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 6463.10 | 2024-10-04 09:15:00 | 6139.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 10:45:00 | 6499.60 | 2024-10-04 09:15:00 | 6174.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 14:30:00 | 6485.70 | 2024-10-04 14:15:00 | 6363.45 | STOP_HIT | 0.50 | 1.88% |
| SELL | retest2 | 2024-10-03 09:15:00 | 6463.10 | 2024-10-04 14:15:00 | 6363.45 | STOP_HIT | 0.50 | 1.54% |
| SELL | retest2 | 2024-10-03 10:45:00 | 6499.60 | 2024-10-04 14:15:00 | 6363.45 | STOP_HIT | 0.50 | 2.09% |
| SELL | retest2 | 2024-10-03 11:30:00 | 6448.65 | 2024-10-07 10:15:00 | 6126.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 11:30:00 | 6448.65 | 2024-10-08 10:15:00 | 6155.50 | STOP_HIT | 0.50 | 4.55% |
| SELL | retest2 | 2024-10-07 10:30:00 | 6140.45 | 2024-10-09 11:15:00 | 6265.55 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2024-10-08 11:30:00 | 6167.30 | 2024-10-09 11:15:00 | 6265.55 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2024-10-08 14:45:00 | 6164.55 | 2024-10-09 11:15:00 | 6265.55 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2024-10-15 09:15:00 | 6300.40 | 2024-10-15 10:15:00 | 6192.75 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-10-18 09:15:00 | 6065.00 | 2024-10-22 09:15:00 | 5761.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 11:30:00 | 6058.55 | 2024-10-22 09:15:00 | 5755.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-18 09:15:00 | 6065.00 | 2024-10-22 14:15:00 | 5458.50 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-18 11:30:00 | 6058.55 | 2024-10-22 14:15:00 | 5452.70 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest1 | 2024-11-14 11:30:00 | 4690.00 | 2024-11-19 09:15:00 | 4897.75 | STOP_HIT | 1.00 | -4.43% |
| SELL | retest1 | 2024-11-14 13:30:00 | 4684.80 | 2024-11-19 09:15:00 | 4897.75 | STOP_HIT | 1.00 | -4.55% |
| SELL | retest1 | 2024-11-18 09:15:00 | 4645.00 | 2024-11-19 09:15:00 | 4897.75 | STOP_HIT | 1.00 | -5.44% |
| BUY | retest2 | 2024-11-21 14:00:00 | 4913.95 | 2024-11-27 10:15:00 | 4996.70 | STOP_HIT | 1.00 | 1.68% |
| SELL | retest2 | 2024-11-28 10:30:00 | 4969.55 | 2024-11-29 13:15:00 | 5084.85 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2024-11-28 12:30:00 | 4969.00 | 2024-11-29 13:15:00 | 5084.85 | STOP_HIT | 1.00 | -2.33% |
| SELL | retest2 | 2024-11-28 14:00:00 | 4970.20 | 2024-11-29 13:15:00 | 5084.85 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2024-11-28 15:15:00 | 4955.05 | 2024-11-29 13:15:00 | 5084.85 | STOP_HIT | 1.00 | -2.62% |
| BUY | retest2 | 2024-12-12 09:15:00 | 5062.85 | 2024-12-12 15:15:00 | 4960.00 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-12-16 13:15:00 | 5074.90 | 2024-12-18 14:15:00 | 5582.39 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-16 14:00:00 | 5066.05 | 2024-12-18 14:15:00 | 5572.66 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-17 09:30:00 | 5072.00 | 2024-12-18 14:15:00 | 5579.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-18 09:15:00 | 5195.30 | 2024-12-18 15:15:00 | 5714.83 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-18 11:15:00 | 5283.25 | 2024-12-18 15:15:00 | 5811.58 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-20 10:00:00 | 5300.90 | 2024-12-20 13:15:00 | 5151.10 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest1 | 2025-01-13 09:15:00 | 5120.00 | 2025-01-15 09:15:00 | 4864.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-01-13 09:15:00 | 5120.00 | 2025-01-16 13:15:00 | 4931.60 | STOP_HIT | 0.50 | 3.68% |
| SELL | retest2 | 2025-01-21 10:15:00 | 4655.10 | 2025-01-28 09:15:00 | 4422.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 10:45:00 | 4660.00 | 2025-01-28 09:15:00 | 4427.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 12:00:00 | 4658.00 | 2025-01-28 09:15:00 | 4425.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 10:45:00 | 4652.95 | 2025-01-28 09:15:00 | 4420.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-21 10:15:00 | 4655.10 | 2025-01-29 13:15:00 | 4189.59 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-21 10:45:00 | 4660.00 | 2025-01-29 13:15:00 | 4194.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-21 12:00:00 | 4658.00 | 2025-01-29 13:15:00 | 4192.20 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-23 10:45:00 | 4652.95 | 2025-01-29 13:15:00 | 4187.65 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 13:00:00 | 4599.00 | 2025-01-29 13:15:00 | 4369.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 15:00:00 | 4569.90 | 2025-01-29 13:15:00 | 4341.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 13:00:00 | 4599.00 | 2025-01-30 10:15:00 | 4139.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 15:00:00 | 4569.90 | 2025-01-30 13:15:00 | 4112.91 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-12 09:15:00 | 3917.25 | 2025-02-17 14:15:00 | 4055.80 | STOP_HIT | 1.00 | -3.54% |
| SELL | retest2 | 2025-02-14 09:15:00 | 3943.35 | 2025-02-17 14:15:00 | 4055.80 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2025-02-21 09:15:00 | 4069.95 | 2025-02-24 09:15:00 | 3997.55 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2025-03-05 12:45:00 | 4613.20 | 2025-03-05 14:15:00 | 5074.52 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-03-13 09:15:00 | 4820.55 | 2025-03-18 11:15:00 | 4906.55 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-03-21 13:30:00 | 4771.55 | 2025-03-21 14:15:00 | 4818.15 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-04-02 09:15:00 | 4688.00 | 2025-04-07 09:15:00 | 4219.20 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-03 11:30:00 | 4713.80 | 2025-04-07 09:15:00 | 4242.42 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-24 14:15:00 | 4906.00 | 2025-04-25 09:15:00 | 4752.20 | STOP_HIT | 1.00 | -3.13% |
| BUY | retest2 | 2025-05-16 11:30:00 | 5399.00 | 2025-05-16 14:15:00 | 5249.30 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2025-05-19 09:15:00 | 5365.50 | 2025-05-20 13:15:00 | 5263.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-05-19 13:30:00 | 5313.20 | 2025-05-20 13:15:00 | 5263.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-05-19 14:30:00 | 5313.50 | 2025-05-20 13:15:00 | 5263.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-05-26 09:30:00 | 5518.60 | 2025-05-26 13:15:00 | 5430.00 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-05-27 09:45:00 | 5520.10 | 2025-05-30 15:15:00 | 5609.90 | STOP_HIT | 1.00 | 1.63% |
| SELL | retest2 | 2025-06-18 09:45:00 | 5251.50 | 2025-06-23 10:15:00 | 5268.50 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-06-18 12:45:00 | 5245.00 | 2025-06-23 12:15:00 | 5278.00 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-06-18 15:00:00 | 5247.50 | 2025-06-23 12:15:00 | 5278.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-06-19 10:15:00 | 5237.00 | 2025-06-23 12:15:00 | 5278.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-06-20 15:00:00 | 5163.50 | 2025-06-23 12:15:00 | 5278.00 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-07-09 09:15:00 | 5930.00 | 2025-07-14 12:15:00 | 5910.00 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-07-09 10:15:00 | 5985.00 | 2025-07-14 12:15:00 | 5910.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-07-11 13:15:00 | 5928.50 | 2025-07-14 12:15:00 | 5910.00 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-07-11 14:45:00 | 5937.00 | 2025-07-14 12:15:00 | 5910.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-07-24 10:30:00 | 6395.00 | 2025-07-29 13:15:00 | 6075.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 10:30:00 | 6395.00 | 2025-07-29 14:15:00 | 6475.00 | STOP_HIT | 0.50 | -1.25% |
| SELL | retest2 | 2025-09-08 09:15:00 | 6848.50 | 2025-09-12 10:15:00 | 6802.50 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2025-09-08 13:00:00 | 6842.00 | 2025-09-12 10:15:00 | 6802.50 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2025-09-08 14:15:00 | 6844.00 | 2025-09-12 10:15:00 | 6802.50 | STOP_HIT | 1.00 | 0.61% |
| SELL | retest2 | 2025-09-16 11:30:00 | 6666.00 | 2025-09-16 13:15:00 | 6713.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-09-19 10:45:00 | 6836.50 | 2025-09-19 13:15:00 | 6734.50 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-09-23 09:15:00 | 6846.00 | 2025-09-24 10:15:00 | 6746.50 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-09-23 14:30:00 | 6816.00 | 2025-09-24 10:15:00 | 6746.50 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-10-15 14:00:00 | 6425.00 | 2025-10-16 09:15:00 | 6790.00 | STOP_HIT | 1.00 | -5.68% |
| SELL | retest2 | 2025-10-15 15:15:00 | 6409.00 | 2025-10-16 09:15:00 | 6790.00 | STOP_HIT | 1.00 | -5.94% |
| BUY | retest2 | 2025-10-23 12:30:00 | 6705.00 | 2025-10-23 14:15:00 | 6614.00 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-10-28 12:15:00 | 6480.00 | 2025-10-30 10:15:00 | 6583.00 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-10-28 13:30:00 | 6468.00 | 2025-10-30 10:15:00 | 6583.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-10-28 14:00:00 | 6434.00 | 2025-10-30 10:15:00 | 6583.00 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-10-29 15:00:00 | 6463.00 | 2025-10-30 10:15:00 | 6583.00 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-11-18 10:30:00 | 6678.50 | 2025-11-25 09:15:00 | 6942.50 | STOP_HIT | 1.00 | -3.95% |
| SELL | retest2 | 2025-11-18 11:45:00 | 6678.00 | 2025-11-25 09:15:00 | 6942.50 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2025-11-21 11:00:00 | 6640.00 | 2025-11-25 09:15:00 | 6942.50 | STOP_HIT | 1.00 | -4.56% |
| SELL | retest2 | 2025-11-24 09:30:00 | 6588.50 | 2025-11-25 09:15:00 | 6942.50 | STOP_HIT | 1.00 | -5.37% |
| BUY | retest2 | 2025-11-27 12:15:00 | 7042.00 | 2025-12-02 10:15:00 | 6980.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-11-28 11:45:00 | 7013.50 | 2025-12-02 10:15:00 | 6980.00 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-11-28 13:00:00 | 7005.00 | 2025-12-02 10:15:00 | 6980.00 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-12-01 12:30:00 | 6995.50 | 2025-12-02 10:15:00 | 6980.00 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-12-18 09:15:00 | 6922.50 | 2025-12-18 15:15:00 | 7076.50 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-12-23 09:15:00 | 7076.50 | 2025-12-24 15:15:00 | 7002.50 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-12-23 09:45:00 | 7066.50 | 2025-12-24 15:15:00 | 7002.50 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-12-23 11:15:00 | 7098.00 | 2025-12-24 15:15:00 | 7002.50 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-12-24 09:15:00 | 7078.50 | 2025-12-24 15:15:00 | 7002.50 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-12-24 13:45:00 | 7102.50 | 2025-12-24 15:15:00 | 7002.50 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2026-01-08 11:15:00 | 7703.00 | 2026-01-12 11:15:00 | 7334.95 | PARTIAL | 0.50 | 4.78% |
| SELL | retest2 | 2026-01-08 11:15:00 | 7703.00 | 2026-01-12 12:15:00 | 7566.00 | STOP_HIT | 0.50 | 1.78% |
| SELL | retest2 | 2026-01-08 12:45:00 | 7721.00 | 2026-01-14 09:15:00 | 7754.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2026-01-08 13:15:00 | 7709.00 | 2026-01-14 09:15:00 | 7754.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-01-08 14:00:00 | 7716.00 | 2026-01-14 09:15:00 | 7754.00 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2026-01-22 11:15:00 | 7488.00 | 2026-01-22 14:15:00 | 7616.50 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest1 | 2026-02-05 11:15:00 | 7828.00 | 2026-02-06 12:15:00 | 7755.50 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest1 | 2026-02-05 11:45:00 | 7822.00 | 2026-02-06 12:15:00 | 7755.50 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest1 | 2026-02-05 13:45:00 | 7823.50 | 2026-02-06 12:15:00 | 7755.50 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-02-06 14:45:00 | 7800.00 | 2026-02-13 09:15:00 | 7734.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2026-02-23 09:45:00 | 8099.50 | 2026-02-24 09:15:00 | 7909.00 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2026-02-26 11:15:00 | 7750.50 | 2026-03-02 09:15:00 | 7362.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 11:15:00 | 7750.50 | 2026-03-02 10:15:00 | 7634.50 | STOP_HIT | 0.50 | 1.50% |
| BUY | retest2 | 2026-04-13 10:45:00 | 7390.50 | 2026-04-24 10:15:00 | 7611.50 | STOP_HIT | 1.00 | 2.99% |
| BUY | retest2 | 2026-04-13 11:15:00 | 7405.50 | 2026-04-24 10:15:00 | 7611.50 | STOP_HIT | 1.00 | 2.78% |
| BUY | retest2 | 2026-04-15 09:15:00 | 7466.00 | 2026-04-24 10:15:00 | 7611.50 | STOP_HIT | 1.00 | 1.95% |
| BUY | retest2 | 2026-04-15 14:15:00 | 7411.50 | 2026-04-24 10:15:00 | 7611.50 | STOP_HIT | 1.00 | 2.70% |
| BUY | retest2 | 2026-04-15 15:15:00 | 7450.00 | 2026-04-24 10:15:00 | 7611.50 | STOP_HIT | 1.00 | 2.17% |
| SELL | retest2 | 2026-04-29 10:30:00 | 7553.50 | 2026-04-29 11:15:00 | 7644.50 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-05-04 09:15:00 | 7785.00 | 2026-05-05 10:15:00 | 7616.50 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2026-05-07 15:15:00 | 8777.00 | 2026-05-08 10:15:00 | 9654.70 | TARGET_HIT | 1.00 | 10.00% |
