# Avenue Supermarts Ltd. (DMART)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-03-25 15:15:00 (1794 bars)
- **Last close:** 3920.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 69 |
| ALERT1 | 44 |
| ALERT2 | 43 |
| ALERT2_SKIP | 24 |
| ALERT3 | 115 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 58 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 61 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 61 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 50
- **Target hits / Stop hits / Partials:** 0 / 59 / 2
- **Avg / median % per leg:** -0.35% / -0.80%
- **Sum % (uncompounded):** -21.26%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 24 | 4 | 16.7% | 0 | 23 | 1 | -0.48% | -11.5% |
| BUY @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.98% | 10.0% |
| BUY @ 3rd Alert (retest2) | 22 | 2 | 9.1% | 0 | 22 | 0 | -0.97% | -21.4% |
| SELL (all) | 37 | 7 | 18.9% | 0 | 36 | 1 | -0.26% | -9.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 37 | 7 | 18.9% | 0 | 36 | 1 | -0.26% | -9.8% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.98% | 10.0% |
| retest2 (combined) | 59 | 9 | 15.3% | 0 | 58 | 1 | -0.53% | -31.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 13:15:00 | 4021.80 | 4000.99 | 3999.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 15:15:00 | 4032.00 | 4010.24 | 4004.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 4047.50 | 4054.37 | 4037.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-14 09:15:00 | 4047.50 | 4054.37 | 4037.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 4047.50 | 4054.37 | 4037.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 09:45:00 | 4036.00 | 4054.37 | 4037.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 4037.50 | 4051.00 | 4037.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:00:00 | 4037.50 | 4051.00 | 4037.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 4023.30 | 4045.46 | 4036.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:45:00 | 4030.00 | 4045.46 | 4036.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 4021.90 | 4040.75 | 4035.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 4019.80 | 4040.75 | 4035.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 4052.50 | 4042.66 | 4036.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:30:00 | 4034.90 | 4042.66 | 4036.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 4070.90 | 4051.47 | 4043.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 12:30:00 | 4090.10 | 4059.97 | 4047.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 4099.60 | 4069.29 | 4055.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 4087.90 | 4141.79 | 4142.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 4087.90 | 4141.79 | 4142.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 4070.00 | 4127.43 | 4136.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 12:15:00 | 4114.10 | 4108.26 | 4121.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 12:30:00 | 4116.50 | 4108.26 | 4121.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 13:15:00 | 4099.10 | 4106.43 | 4119.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 4081.40 | 4107.39 | 4117.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 13:45:00 | 4092.90 | 4088.29 | 4102.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 4149.40 | 4099.44 | 4103.86 | SL hit (close>static) qty=1.00 sl=4120.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 11:15:00 | 4121.80 | 4108.27 | 4107.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 12:15:00 | 4131.50 | 4112.92 | 4109.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 12:15:00 | 4128.40 | 4132.48 | 4123.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 12:15:00 | 4128.40 | 4132.48 | 4123.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 12:15:00 | 4128.40 | 4132.48 | 4123.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 13:15:00 | 4124.00 | 4132.48 | 4123.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 4129.60 | 4131.90 | 4123.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 14:15:00 | 4136.30 | 4131.90 | 4123.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 14:45:00 | 4135.60 | 4133.52 | 4125.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-27 09:15:00 | 4101.00 | 4127.37 | 4123.93 | SL hit (close<static) qty=1.00 sl=4120.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 11:15:00 | 4106.10 | 4119.93 | 4120.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 14:15:00 | 4096.90 | 4110.26 | 4115.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 15:15:00 | 4050.00 | 4047.76 | 4064.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-30 09:15:00 | 4062.00 | 4047.76 | 4064.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 4030.90 | 4044.39 | 4061.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 12:00:00 | 4018.00 | 4036.52 | 4054.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:45:00 | 4021.80 | 4014.02 | 4024.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 11:45:00 | 4020.00 | 4024.60 | 4027.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 13:15:00 | 4053.20 | 4033.77 | 4031.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 13:15:00 | 4053.20 | 4033.77 | 4031.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 15:15:00 | 4055.90 | 4041.63 | 4035.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 4023.00 | 4037.90 | 4034.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 4023.00 | 4037.90 | 4034.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 4023.00 | 4037.90 | 4034.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 4015.60 | 4037.90 | 4034.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 4041.60 | 4038.64 | 4035.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 14:30:00 | 4052.10 | 4047.78 | 4040.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 12:15:00 | 4090.90 | 4149.44 | 4156.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 12:15:00 | 4090.90 | 4149.44 | 4156.18 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 4087.00 | 4066.61 | 4064.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 09:15:00 | 4179.00 | 4091.27 | 4077.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 4185.40 | 4195.15 | 4158.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 12:45:00 | 4181.70 | 4195.15 | 4158.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 4185.70 | 4195.17 | 4170.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 11:30:00 | 4246.60 | 4206.26 | 4179.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 4263.30 | 4370.40 | 4374.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 09:15:00 | 4263.30 | 4370.40 | 4374.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 4216.80 | 4245.20 | 4275.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 10:15:00 | 4230.70 | 4223.54 | 4246.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 10:45:00 | 4228.00 | 4223.54 | 4246.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 4065.90 | 4038.01 | 4057.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:45:00 | 4070.00 | 4038.01 | 4057.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 4100.10 | 4050.42 | 4061.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 4100.10 | 4050.42 | 4061.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 4098.00 | 4059.94 | 4065.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 12:30:00 | 4082.70 | 4064.41 | 4066.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 4100.90 | 4071.71 | 4069.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 4100.90 | 4071.71 | 4069.77 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 12:15:00 | 4050.30 | 4070.77 | 4071.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 09:15:00 | 4045.70 | 4061.01 | 4065.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 14:15:00 | 4053.20 | 4052.55 | 4059.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 14:15:00 | 4053.20 | 4052.55 | 4059.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 4053.20 | 4052.55 | 4059.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:45:00 | 4050.10 | 4052.55 | 4059.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 4023.90 | 4031.21 | 4041.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:30:00 | 4002.00 | 4031.80 | 4038.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 4046.30 | 4038.55 | 4037.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 4046.30 | 4038.55 | 4037.87 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 4029.50 | 4036.74 | 4037.11 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 13:15:00 | 4049.10 | 4039.58 | 4038.36 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 14:15:00 | 4017.00 | 4035.06 | 4036.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 4000.00 | 4027.96 | 4032.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-25 14:15:00 | 3999.20 | 3996.06 | 4012.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-25 15:00:00 | 3999.20 | 3996.06 | 4012.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 4011.20 | 3995.77 | 4008.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:45:00 | 4007.20 | 3995.77 | 4008.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 3988.10 | 3994.24 | 4006.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 12:15:00 | 3984.90 | 3994.24 | 4006.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 09:30:00 | 3966.50 | 3967.22 | 3986.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 09:15:00 | 4019.00 | 3987.80 | 3986.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 4019.00 | 3987.80 | 3986.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 11:15:00 | 4206.10 | 4038.71 | 4010.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 10:15:00 | 4242.80 | 4258.48 | 4199.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 11:00:00 | 4242.80 | 4258.48 | 4199.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 4186.00 | 4228.71 | 4202.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 4186.00 | 4228.71 | 4202.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 4216.60 | 4226.29 | 4203.89 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 13:15:00 | 4163.80 | 4191.89 | 4193.82 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 4215.00 | 4195.65 | 4195.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 09:15:00 | 4224.90 | 4201.50 | 4197.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 10:15:00 | 4236.70 | 4243.53 | 4227.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-06 11:00:00 | 4236.70 | 4243.53 | 4227.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 4230.50 | 4240.92 | 4227.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:00:00 | 4230.50 | 4240.92 | 4227.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 4243.90 | 4241.52 | 4229.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 09:30:00 | 4265.50 | 4244.42 | 4234.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 14:00:00 | 4255.00 | 4255.00 | 4243.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 4263.30 | 4254.97 | 4245.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 11:15:00 | 4205.00 | 4241.33 | 4241.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 4205.00 | 4241.33 | 4241.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 13:15:00 | 4187.90 | 4226.49 | 4234.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 10:15:00 | 4211.70 | 4203.03 | 4218.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 10:15:00 | 4211.70 | 4203.03 | 4218.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 4211.70 | 4203.03 | 4218.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 11:00:00 | 4211.70 | 4203.03 | 4218.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 4210.60 | 4204.55 | 4217.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 12:15:00 | 4220.20 | 4204.55 | 4217.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 4189.50 | 4201.54 | 4215.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 13:15:00 | 4185.00 | 4201.54 | 4215.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 4232.80 | 4200.33 | 4209.44 | SL hit (close>static) qty=1.00 sl=4222.80 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 4257.60 | 4217.01 | 4215.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 4276.70 | 4245.95 | 4232.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 15:15:00 | 4347.80 | 4351.40 | 4316.09 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 09:15:00 | 4474.00 | 4351.40 | 4316.09 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 09:15:00 | 4697.70 | 4629.06 | 4547.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-21 11:15:00 | 4695.90 | 4708.06 | 4648.23 | SL hit (close<ema200) qty=0.50 sl=4708.06 alert=retest1 |

### Cycle 20 — SELL (started 2025-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 09:15:00 | 4680.00 | 4708.87 | 4709.19 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 10:15:00 | 4719.60 | 4711.02 | 4710.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-29 11:15:00 | 4762.80 | 4721.37 | 4714.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-01 14:15:00 | 4729.00 | 4767.64 | 4749.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 14:15:00 | 4729.00 | 4767.64 | 4749.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 4729.00 | 4767.64 | 4749.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 4729.00 | 4767.64 | 4749.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 4748.00 | 4763.71 | 4749.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 4694.40 | 4763.71 | 4749.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 4702.20 | 4751.41 | 4745.43 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 10:15:00 | 4696.60 | 4740.45 | 4740.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-02 11:15:00 | 4679.20 | 4728.20 | 4735.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 4719.40 | 4703.78 | 4718.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 09:15:00 | 4719.40 | 4703.78 | 4718.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 4719.40 | 4703.78 | 4718.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:45:00 | 4712.60 | 4703.78 | 4718.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 4705.80 | 4704.18 | 4717.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-03 11:30:00 | 4692.70 | 4707.23 | 4717.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 12:15:00 | 4741.50 | 4714.08 | 4719.47 | SL hit (close>static) qty=1.00 sl=4725.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 14:15:00 | 4761.00 | 4725.12 | 4723.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 4841.50 | 4754.28 | 4737.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 11:15:00 | 4796.60 | 4805.54 | 4780.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-05 12:00:00 | 4796.60 | 4805.54 | 4780.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 4795.10 | 4803.45 | 4781.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:00:00 | 4795.10 | 4803.45 | 4781.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 4800.20 | 4802.04 | 4784.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:30:00 | 4785.80 | 4802.04 | 4784.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 4800.00 | 4801.63 | 4786.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 4744.30 | 4801.63 | 4786.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 4752.10 | 4791.73 | 4782.93 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 11:15:00 | 4751.80 | 4774.89 | 4776.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 15:15:00 | 4712.00 | 4744.72 | 4760.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 4768.60 | 4749.50 | 4760.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 4768.60 | 4749.50 | 4760.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 4768.60 | 4749.50 | 4760.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:00:00 | 4768.60 | 4749.50 | 4760.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 4765.10 | 4752.62 | 4761.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 10:30:00 | 4767.50 | 4752.62 | 4761.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 4754.00 | 4752.89 | 4760.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 13:00:00 | 4743.20 | 4750.95 | 4759.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 09:45:00 | 4740.30 | 4752.89 | 4757.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:15:00 | 4744.00 | 4752.89 | 4757.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 13:15:00 | 4677.00 | 4637.11 | 4636.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-15 13:15:00 | 4677.00 | 4637.11 | 4636.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 10:15:00 | 4699.90 | 4661.46 | 4648.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 4707.40 | 4714.65 | 4690.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 14:00:00 | 4707.40 | 4714.65 | 4690.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 12:15:00 | 4733.00 | 4750.92 | 4736.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 13:00:00 | 4733.00 | 4750.92 | 4736.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 13:15:00 | 4755.00 | 4751.74 | 4738.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 14:30:00 | 4763.90 | 4756.73 | 4741.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 09:30:00 | 4780.00 | 4761.35 | 4746.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 10:15:00 | 4765.50 | 4761.35 | 4746.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 12:15:00 | 4723.00 | 4749.16 | 4744.59 | SL hit (close<static) qty=1.00 sl=4728.30 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 13:15:00 | 4689.80 | 4737.29 | 4739.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 4666.30 | 4723.09 | 4732.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 4651.80 | 4645.09 | 4674.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 10:00:00 | 4651.80 | 4645.09 | 4674.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 11:15:00 | 4668.40 | 4648.86 | 4671.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:45:00 | 4669.40 | 4648.86 | 4671.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 4729.40 | 4664.96 | 4676.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:00:00 | 4729.40 | 4664.96 | 4676.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 4687.30 | 4669.43 | 4677.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:15:00 | 4675.20 | 4669.43 | 4677.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 12:15:00 | 4441.44 | 4496.02 | 4531.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 4464.10 | 4460.24 | 4489.79 | SL hit (close>ema200) qty=0.50 sl=4460.24 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 4353.40 | 4320.86 | 4317.10 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 4269.10 | 4312.11 | 4314.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 12:15:00 | 4215.00 | 4273.72 | 4295.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 4252.20 | 4223.04 | 4243.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 4252.20 | 4223.04 | 4243.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 4252.20 | 4223.04 | 4243.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 4238.50 | 4223.04 | 4243.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 4249.30 | 4228.30 | 4243.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:45:00 | 4262.80 | 4228.30 | 4243.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 4246.80 | 4233.12 | 4243.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:00:00 | 4246.80 | 4233.12 | 4243.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 4251.00 | 4236.69 | 4243.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:15:00 | 4250.00 | 4236.69 | 4243.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 4279.20 | 4252.32 | 4249.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 11:15:00 | 4314.80 | 4270.84 | 4259.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 4290.10 | 4296.49 | 4278.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 4290.10 | 4296.49 | 4278.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 4290.10 | 4296.49 | 4278.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 09:30:00 | 4288.10 | 4296.49 | 4278.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 4290.30 | 4295.44 | 4281.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:45:00 | 4292.50 | 4295.44 | 4281.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 4273.00 | 4291.60 | 4285.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:00:00 | 4273.00 | 4291.60 | 4285.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 4273.40 | 4287.96 | 4284.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:30:00 | 4267.10 | 4287.96 | 4284.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 13:15:00 | 4271.00 | 4282.61 | 4282.63 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 4291.50 | 4282.12 | 4282.12 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 10:15:00 | 4280.30 | 4281.76 | 4281.95 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 11:15:00 | 4290.10 | 4283.42 | 4282.69 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 4272.40 | 4282.26 | 4282.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 09:15:00 | 4244.80 | 4273.61 | 4278.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 4233.80 | 4232.56 | 4250.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 11:15:00 | 4249.30 | 4235.13 | 4248.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 11:15:00 | 4249.30 | 4235.13 | 4248.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:00:00 | 4249.30 | 4235.13 | 4248.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 12:15:00 | 4257.10 | 4239.52 | 4249.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 12:45:00 | 4257.80 | 4239.52 | 4249.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 4255.30 | 4242.68 | 4250.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 14:00:00 | 4255.30 | 4242.68 | 4250.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 4264.70 | 4247.08 | 4251.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 15:00:00 | 4264.70 | 4247.08 | 4251.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 4233.90 | 4223.76 | 4234.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 4242.40 | 4223.76 | 4234.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 4240.10 | 4227.02 | 4234.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:15:00 | 4237.20 | 4227.02 | 4234.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 4243.70 | 4230.36 | 4235.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:15:00 | 4250.00 | 4230.36 | 4235.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 4230.10 | 4230.31 | 4235.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 15:00:00 | 4227.20 | 4231.05 | 4234.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:15:00 | 4213.90 | 4230.64 | 4233.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 12:15:00 | 4185.00 | 4176.15 | 4176.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 12:15:00 | 4185.00 | 4176.15 | 4176.14 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 09:15:00 | 4146.00 | 4173.14 | 4175.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 10:15:00 | 4143.50 | 4167.21 | 4172.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 10:15:00 | 4063.50 | 4043.17 | 4077.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-10 11:00:00 | 4063.50 | 4043.17 | 4077.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 13:15:00 | 4056.40 | 4042.89 | 4054.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 13:45:00 | 4056.70 | 4042.89 | 4054.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 4071.00 | 4048.51 | 4055.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 15:00:00 | 4071.00 | 4048.51 | 4055.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 15:15:00 | 4075.00 | 4053.81 | 4057.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:15:00 | 4072.80 | 4053.81 | 4057.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 4070.70 | 4058.18 | 4059.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:00:00 | 4070.70 | 4058.18 | 4059.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 4070.10 | 4060.56 | 4060.05 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 15:15:00 | 4054.00 | 4059.47 | 4059.90 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 09:15:00 | 4067.50 | 4061.08 | 4060.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 10:15:00 | 4080.10 | 4064.88 | 4062.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 4054.40 | 4067.78 | 4065.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 14:15:00 | 4054.40 | 4067.78 | 4065.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 4054.40 | 4067.78 | 4065.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 4054.40 | 4067.78 | 4065.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 4060.00 | 4066.22 | 4064.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 4049.10 | 4066.22 | 4064.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 4055.50 | 4062.12 | 4063.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 11:15:00 | 4031.60 | 4056.02 | 4060.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 14:15:00 | 4056.00 | 4050.61 | 4056.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 14:15:00 | 4056.00 | 4050.61 | 4056.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 4056.00 | 4050.61 | 4056.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 15:00:00 | 4056.00 | 4050.61 | 4056.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 4053.00 | 4051.09 | 4055.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 4056.40 | 4051.09 | 4055.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 4044.40 | 4049.75 | 4054.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 10:15:00 | 4042.00 | 4049.75 | 4054.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 10:45:00 | 4041.70 | 4047.96 | 4053.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:45:00 | 4042.00 | 4046.59 | 4052.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 12:45:00 | 4042.40 | 4044.53 | 4050.96 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 4016.50 | 4010.97 | 4024.70 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 4063.40 | 4031.86 | 4029.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 4063.40 | 4031.86 | 4029.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 13:15:00 | 4088.70 | 4060.11 | 4044.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 4050.00 | 4065.11 | 4051.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 4050.00 | 4065.11 | 4051.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 4050.00 | 4065.11 | 4051.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 4050.00 | 4065.11 | 4051.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 4058.20 | 4063.73 | 4052.23 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-11-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 09:15:00 | 4009.10 | 4043.99 | 4047.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 11:15:00 | 4000.00 | 4028.52 | 4039.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 4021.00 | 3997.91 | 4007.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 4021.00 | 3997.91 | 4007.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 4021.00 | 3997.91 | 4007.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:45:00 | 4020.00 | 3997.91 | 4007.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 4015.20 | 4001.37 | 4008.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:45:00 | 4015.90 | 4001.37 | 4008.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 12:15:00 | 4015.00 | 4006.79 | 4009.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 14:15:00 | 4012.50 | 4008.67 | 4010.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 15:15:00 | 4015.00 | 4011.45 | 4011.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2025-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 15:15:00 | 4015.00 | 4011.45 | 4011.34 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 3987.30 | 4007.11 | 4009.54 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 10:15:00 | 4020.60 | 4011.82 | 4010.69 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 13:15:00 | 3995.00 | 4007.65 | 4009.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 09:15:00 | 3977.10 | 3998.07 | 4004.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 14:15:00 | 3962.20 | 3957.02 | 3971.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 15:00:00 | 3962.20 | 3957.02 | 3971.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 3914.40 | 3949.58 | 3965.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:30:00 | 3906.90 | 3939.59 | 3959.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 11:00:00 | 3899.60 | 3939.59 | 3959.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 15:15:00 | 3905.50 | 3918.21 | 3941.78 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 10:00:00 | 3906.70 | 3913.87 | 3935.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 3926.30 | 3916.36 | 3934.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 3925.00 | 3916.36 | 3934.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 3934.80 | 3920.05 | 3934.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 12:00:00 | 3934.80 | 3920.05 | 3934.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 3916.20 | 3919.28 | 3933.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:15:00 | 3910.00 | 3919.28 | 3933.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 15:15:00 | 3912.00 | 3917.50 | 3929.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:30:00 | 3911.80 | 3918.08 | 3927.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 10:15:00 | 3938.20 | 3922.10 | 3928.82 | SL hit (close>static) qty=1.00 sl=3935.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 14:15:00 | 3954.70 | 3935.36 | 3933.38 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 3903.40 | 3928.12 | 3930.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 12:15:00 | 3869.90 | 3911.71 | 3922.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 3889.50 | 3879.32 | 3897.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-09 11:15:00 | 3889.50 | 3879.32 | 3897.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 3889.50 | 3879.32 | 3897.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 3889.50 | 3879.32 | 3897.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 3895.40 | 3882.54 | 3897.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:15:00 | 3888.10 | 3884.23 | 3896.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 3911.10 | 3889.60 | 3898.11 | SL hit (close>static) qty=1.00 sl=3908.00 alert=retest2 |

### Cycle 49 — BUY (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 09:15:00 | 3892.70 | 3842.46 | 3840.63 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 11:15:00 | 3818.00 | 3846.28 | 3848.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 3794.50 | 3824.24 | 3835.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 11:15:00 | 3790.00 | 3777.12 | 3796.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 12:00:00 | 3790.00 | 3777.12 | 3796.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 3785.20 | 3778.74 | 3795.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:00:00 | 3785.20 | 3778.74 | 3795.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 3808.00 | 3784.59 | 3796.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:00:00 | 3808.00 | 3784.59 | 3796.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 3816.50 | 3790.97 | 3798.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 3817.80 | 3790.97 | 3798.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 3863.90 | 3811.27 | 3806.97 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 12:15:00 | 3805.00 | 3818.30 | 3819.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 13:15:00 | 3784.30 | 3811.50 | 3816.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 14:15:00 | 3788.00 | 3786.94 | 3795.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 15:00:00 | 3788.00 | 3786.94 | 3795.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 3794.60 | 3788.95 | 3794.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:30:00 | 3801.00 | 3788.95 | 3794.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 3775.90 | 3786.34 | 3792.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 14:45:00 | 3769.40 | 3777.06 | 3786.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 09:15:00 | 3764.40 | 3777.10 | 3780.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-07 10:15:00 | 3792.20 | 3703.93 | 3696.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 10:15:00 | 3792.20 | 3703.93 | 3696.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 12:15:00 | 3835.90 | 3746.94 | 3718.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 13:15:00 | 3792.60 | 3802.02 | 3771.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 14:00:00 | 3792.60 | 3802.02 | 3771.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 3782.70 | 3794.25 | 3776.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:00:00 | 3782.70 | 3794.25 | 3776.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 3757.50 | 3786.90 | 3775.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:00:00 | 3757.50 | 3786.90 | 3775.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 3766.20 | 3782.76 | 3774.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:45:00 | 3764.00 | 3782.76 | 3774.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 14:15:00 | 3801.30 | 3785.58 | 3777.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 09:15:00 | 3899.00 | 3789.86 | 3779.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 12:15:00 | 3816.10 | 3805.99 | 3791.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 13:15:00 | 3809.60 | 3803.93 | 3791.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-13 10:15:00 | 3814.60 | 3813.28 | 3801.13 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 3822.20 | 3815.07 | 3803.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-13 11:45:00 | 3836.10 | 3819.05 | 3805.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 13:15:00 | 3789.60 | 3809.15 | 3803.48 | SL hit (close<static) qty=1.00 sl=3792.50 alert=retest2 |

### Cycle 54 — SELL (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 11:15:00 | 3768.80 | 3811.60 | 3816.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 13:15:00 | 3748.50 | 3791.94 | 3806.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 14:15:00 | 3767.00 | 3763.41 | 3781.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-19 14:45:00 | 3766.20 | 3763.41 | 3781.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 3715.00 | 3753.65 | 3773.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 10:15:00 | 3709.20 | 3753.65 | 3773.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 10:45:00 | 3697.90 | 3743.88 | 3767.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:00:00 | 3701.50 | 3694.40 | 3696.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 10:15:00 | 3727.30 | 3700.98 | 3698.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 10:15:00 | 3727.30 | 3700.98 | 3698.94 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 3670.00 | 3695.79 | 3697.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 15:15:00 | 3665.60 | 3689.75 | 3694.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 3685.00 | 3674.17 | 3681.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 15:15:00 | 3685.00 | 3674.17 | 3681.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 3685.00 | 3674.17 | 3681.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 3694.70 | 3674.17 | 3681.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 3697.50 | 3678.84 | 3683.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:00:00 | 3697.50 | 3678.84 | 3683.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 3704.00 | 3683.87 | 3685.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 3704.00 | 3683.87 | 3685.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 3715.40 | 3690.18 | 3687.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 3735.00 | 3707.67 | 3697.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 3694.60 | 3709.43 | 3700.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 3694.60 | 3709.43 | 3700.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 3694.60 | 3709.43 | 3700.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:30:00 | 3679.20 | 3709.43 | 3700.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 3670.40 | 3701.62 | 3697.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 3672.50 | 3701.62 | 3697.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 3679.00 | 3697.10 | 3695.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:15:00 | 3644.90 | 3697.10 | 3695.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 12:15:00 | 3663.60 | 3690.40 | 3692.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 3626.00 | 3672.65 | 3679.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 13:15:00 | 3650.30 | 3645.07 | 3659.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 13:45:00 | 3646.50 | 3645.07 | 3659.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 3674.20 | 3650.90 | 3660.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 15:00:00 | 3674.20 | 3650.90 | 3660.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 3665.50 | 3653.82 | 3660.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 3732.00 | 3653.82 | 3660.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 3720.00 | 3667.05 | 3666.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 3778.30 | 3700.32 | 3682.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 3873.10 | 3886.23 | 3838.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 09:15:00 | 3972.80 | 3986.04 | 3963.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 3972.80 | 3986.04 | 3963.22 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 3890.10 | 3950.66 | 3954.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 10:15:00 | 3880.30 | 3906.96 | 3926.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 3916.00 | 3889.53 | 3906.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 3916.00 | 3889.53 | 3906.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 3916.00 | 3889.53 | 3906.22 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 14:15:00 | 3920.20 | 3905.74 | 3904.55 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 3881.00 | 3901.63 | 3903.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 3861.70 | 3893.65 | 3899.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 3881.90 | 3863.43 | 3878.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 10:15:00 | 3881.90 | 3863.43 | 3878.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 3881.90 | 3863.43 | 3878.21 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 3940.90 | 3866.05 | 3857.01 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 12:15:00 | 3838.40 | 3861.81 | 3863.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 3811.80 | 3852.59 | 3858.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 10:15:00 | 3871.90 | 3856.46 | 3859.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 10:15:00 | 3871.90 | 3856.46 | 3859.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 3871.90 | 3856.46 | 3859.77 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 3850.00 | 3805.43 | 3800.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 09:15:00 | 3872.60 | 3818.86 | 3806.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 11:15:00 | 3916.70 | 3941.93 | 3917.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 11:15:00 | 3916.70 | 3941.93 | 3917.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 3916.70 | 3941.93 | 3917.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 12:00:00 | 3916.70 | 3941.93 | 3917.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 3927.00 | 3938.94 | 3918.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 13:15:00 | 3938.70 | 3938.94 | 3918.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 09:15:00 | 3902.60 | 3932.75 | 3922.58 | SL hit (close<static) qty=1.00 sl=3908.10 alert=retest2 |

### Cycle 66 — SELL (started 2026-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 13:15:00 | 3841.40 | 3916.28 | 3924.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 15:15:00 | 3830.00 | 3886.66 | 3909.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 3815.00 | 3805.04 | 3850.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 3830.80 | 3805.04 | 3850.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 3814.00 | 3809.54 | 3841.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:45:00 | 3839.90 | 3809.54 | 3841.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 3867.00 | 3808.77 | 3824.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:00:00 | 3867.00 | 3808.77 | 3824.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 3891.20 | 3825.25 | 3830.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:45:00 | 3895.20 | 3825.25 | 3830.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 3854.40 | 3838.23 | 3836.31 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 3802.10 | 3835.91 | 3837.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 13:15:00 | 3791.00 | 3817.89 | 3826.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 3696.80 | 3691.71 | 3738.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 11:15:00 | 3760.40 | 3709.98 | 3738.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 3760.40 | 3709.98 | 3738.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 3760.40 | 3709.98 | 3738.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 3784.70 | 3724.92 | 3742.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 3807.20 | 3724.92 | 3742.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 3759.80 | 3744.27 | 3748.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 3850.40 | 3744.27 | 3748.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 3869.40 | 3769.29 | 3759.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 3910.80 | 3797.59 | 3773.23 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 12:30:00 | 4090.10 | 2025-05-20 13:15:00 | 4087.90 | STOP_HIT | 1.00 | -0.05% |
| BUY | retest2 | 2025-05-16 09:15:00 | 4099.60 | 2025-05-20 13:15:00 | 4087.90 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-05-22 09:15:00 | 4081.40 | 2025-05-23 09:15:00 | 4149.40 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-05-22 13:45:00 | 4092.90 | 2025-05-23 09:15:00 | 4149.40 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-05-26 14:15:00 | 4136.30 | 2025-05-27 09:15:00 | 4101.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-05-26 14:45:00 | 4135.60 | 2025-05-27 09:15:00 | 4101.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-05-30 12:00:00 | 4018.00 | 2025-06-03 13:15:00 | 4053.20 | STOP_HIT | 1.00 | -0.88% |
| SELL | retest2 | 2025-06-03 09:45:00 | 4021.80 | 2025-06-03 13:15:00 | 4053.20 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-06-03 11:45:00 | 4020.00 | 2025-06-03 13:15:00 | 4053.20 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-06-04 14:30:00 | 4052.10 | 2025-06-10 12:15:00 | 4090.90 | STOP_HIT | 1.00 | 0.96% |
| BUY | retest2 | 2025-06-20 11:30:00 | 4246.60 | 2025-07-03 09:15:00 | 4263.30 | STOP_HIT | 1.00 | 0.39% |
| SELL | retest2 | 2025-07-16 12:30:00 | 4082.70 | 2025-07-16 13:15:00 | 4100.90 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-07-23 09:30:00 | 4002.00 | 2025-07-24 10:15:00 | 4046.30 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-07-28 12:15:00 | 3984.90 | 2025-07-30 09:15:00 | 4019.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-07-29 09:30:00 | 3966.50 | 2025-07-30 09:15:00 | 4019.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-08-07 09:30:00 | 4265.50 | 2025-08-08 11:15:00 | 4205.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-08-07 14:00:00 | 4255.00 | 2025-08-08 11:15:00 | 4205.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-08-08 09:15:00 | 4263.30 | 2025-08-08 11:15:00 | 4205.00 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-08-11 13:15:00 | 4185.00 | 2025-08-12 09:15:00 | 4232.80 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest1 | 2025-08-18 09:15:00 | 4474.00 | 2025-08-20 09:15:00 | 4697.70 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-08-18 09:15:00 | 4474.00 | 2025-08-21 11:15:00 | 4695.90 | STOP_HIT | 0.50 | 4.96% |
| BUY | retest2 | 2025-08-22 13:15:00 | 4687.40 | 2025-08-29 09:15:00 | 4680.00 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-09-03 11:30:00 | 4692.70 | 2025-09-03 12:15:00 | 4741.50 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-09-09 13:00:00 | 4743.20 | 2025-09-15 13:15:00 | 4677.00 | STOP_HIT | 1.00 | 1.40% |
| SELL | retest2 | 2025-09-10 09:45:00 | 4740.30 | 2025-09-15 13:15:00 | 4677.00 | STOP_HIT | 1.00 | 1.34% |
| SELL | retest2 | 2025-09-10 10:15:00 | 4744.00 | 2025-09-15 13:15:00 | 4677.00 | STOP_HIT | 1.00 | 1.41% |
| BUY | retest2 | 2025-09-19 14:30:00 | 4763.90 | 2025-09-22 12:15:00 | 4723.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-09-22 09:30:00 | 4780.00 | 2025-09-22 12:15:00 | 4723.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-09-22 10:15:00 | 4765.50 | 2025-09-22 12:15:00 | 4723.00 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-09-24 14:15:00 | 4675.20 | 2025-09-30 12:15:00 | 4441.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 14:15:00 | 4675.20 | 2025-10-01 13:15:00 | 4464.10 | STOP_HIT | 0.50 | 4.52% |
| SELL | retest2 | 2025-10-29 15:00:00 | 4227.20 | 2025-11-04 12:15:00 | 4185.00 | STOP_HIT | 1.00 | 1.00% |
| SELL | retest2 | 2025-10-30 09:15:00 | 4213.90 | 2025-11-04 12:15:00 | 4185.00 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2025-11-17 10:15:00 | 4042.00 | 2025-11-20 09:15:00 | 4063.40 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-11-17 10:45:00 | 4041.70 | 2025-11-20 09:15:00 | 4063.40 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-11-17 11:45:00 | 4042.00 | 2025-11-20 09:15:00 | 4063.40 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-11-17 12:45:00 | 4042.40 | 2025-11-20 09:15:00 | 4063.40 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-11-26 14:15:00 | 4012.50 | 2025-11-26 15:15:00 | 4015.00 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-12-03 10:30:00 | 3906.90 | 2025-12-05 10:15:00 | 3938.20 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-12-03 11:00:00 | 3899.60 | 2025-12-05 10:15:00 | 3938.20 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-12-03 15:15:00 | 3905.50 | 2025-12-05 10:15:00 | 3938.20 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-12-04 10:00:00 | 3906.70 | 2025-12-05 14:15:00 | 3954.70 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-12-04 13:15:00 | 3910.00 | 2025-12-05 14:15:00 | 3954.70 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-12-04 15:15:00 | 3912.00 | 2025-12-05 14:15:00 | 3954.70 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-12-05 09:30:00 | 3911.80 | 2025-12-05 14:15:00 | 3954.70 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-12-09 14:15:00 | 3888.10 | 2025-12-09 14:15:00 | 3911.10 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-12-10 10:15:00 | 3876.90 | 2025-12-16 09:15:00 | 3892.70 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-12-30 14:45:00 | 3769.40 | 2026-01-07 10:15:00 | 3792.20 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2026-01-01 09:15:00 | 3764.40 | 2026-01-07 10:15:00 | 3792.20 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2026-01-12 09:15:00 | 3899.00 | 2026-01-13 13:15:00 | 3789.60 | STOP_HIT | 1.00 | -2.81% |
| BUY | retest2 | 2026-01-12 12:15:00 | 3816.10 | 2026-01-16 09:15:00 | 3792.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2026-01-12 13:15:00 | 3809.60 | 2026-01-16 09:15:00 | 3792.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2026-01-13 10:15:00 | 3814.60 | 2026-01-16 11:15:00 | 3768.80 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-01-13 11:45:00 | 3836.10 | 2026-01-16 11:15:00 | 3768.80 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2026-01-14 09:15:00 | 3857.10 | 2026-01-16 11:15:00 | 3768.80 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2026-01-14 10:30:00 | 3841.70 | 2026-01-16 11:15:00 | 3768.80 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2026-01-20 10:15:00 | 3709.20 | 2026-01-23 10:15:00 | 3727.30 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2026-01-20 10:45:00 | 3697.90 | 2026-01-23 10:15:00 | 3727.30 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-01-23 10:00:00 | 3701.50 | 2026-01-23 10:15:00 | 3727.30 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2026-03-11 13:15:00 | 3938.70 | 2026-03-12 09:15:00 | 3902.60 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-03-12 11:30:00 | 3941.00 | 2026-03-13 12:15:00 | 3872.60 | STOP_HIT | 1.00 | -1.74% |
