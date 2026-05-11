# Hitachi Energy India Ltd. (POWERINDIA)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 33960.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 214 |
| ALERT1 | 154 |
| ALERT2 | 152 |
| ALERT2_SKIP | 81 |
| ALERT3 | 398 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 159 |
| PARTIAL | 10 |
| TARGET_HIT | 17 |
| STOP_HIT | 150 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 177 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 62 / 115
- **Target hits / Stop hits / Partials:** 17 / 150 / 10
- **Avg / median % per leg:** 0.37% / -0.91%
- **Sum % (uncompounded):** 66.34%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 72 | 30 | 41.7% | 16 | 56 | 0 | 1.69% | 122.0% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.74% | -1.7% |
| BUY @ 3rd Alert (retest2) | 71 | 30 | 42.3% | 16 | 55 | 0 | 1.74% | 123.7% |
| SELL (all) | 105 | 32 | 30.5% | 1 | 94 | 10 | -0.53% | -55.6% |
| SELL @ 2nd Alert (retest1) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.79% | -12.6% |
| SELL @ 3rd Alert (retest2) | 98 | 32 | 32.7% | 1 | 87 | 10 | -0.44% | -43.1% |
| retest1 (combined) | 8 | 0 | 0.0% | 0 | 8 | 0 | -1.79% | -14.3% |
| retest2 (combined) | 169 | 62 | 36.7% | 17 | 142 | 10 | 0.48% | 80.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-24 14:15:00 | 3821.95 | 3975.71 | 3980.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-24 15:15:00 | 3784.00 | 3937.37 | 3962.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-26 11:15:00 | 3805.25 | 3803.27 | 3853.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-05-26 12:00:00 | 3805.25 | 3803.27 | 3853.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 09:15:00 | 3867.15 | 3809.16 | 3836.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-29 10:00:00 | 3867.15 | 3809.16 | 3836.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-29 10:15:00 | 3918.05 | 3830.94 | 3843.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-29 10:30:00 | 3919.05 | 3830.94 | 3843.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2023-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-29 12:15:00 | 3897.95 | 3856.36 | 3853.92 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2023-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 13:15:00 | 3849.65 | 3866.83 | 3867.71 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-06-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 12:15:00 | 3875.00 | 3868.22 | 3867.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-01 13:15:00 | 3900.35 | 3874.65 | 3870.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-05 10:15:00 | 3911.95 | 3914.84 | 3900.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-05 10:15:00 | 3911.95 | 3914.84 | 3900.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 10:15:00 | 3911.95 | 3914.84 | 3900.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-05 10:45:00 | 3905.05 | 3914.84 | 3900.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 13:15:00 | 3906.05 | 3922.22 | 3908.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-05 14:00:00 | 3906.05 | 3922.22 | 3908.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 14:15:00 | 3875.90 | 3912.96 | 3905.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-05 15:00:00 | 3875.90 | 3912.96 | 3905.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-05 15:15:00 | 3900.00 | 3910.37 | 3904.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-06 09:45:00 | 3912.80 | 3914.92 | 3907.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-08 15:15:00 | 3906.25 | 3947.19 | 3948.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2023-06-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 15:15:00 | 3906.25 | 3947.19 | 3948.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 10:15:00 | 3903.95 | 3932.92 | 3941.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 11:15:00 | 3916.10 | 3884.62 | 3904.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 11:15:00 | 3916.10 | 3884.62 | 3904.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 11:15:00 | 3916.10 | 3884.62 | 3904.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 11:45:00 | 3923.90 | 3884.62 | 3904.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 12:15:00 | 3926.80 | 3893.05 | 3906.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 12:45:00 | 3930.75 | 3893.05 | 3906.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2023-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 09:15:00 | 3999.95 | 3928.85 | 3920.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-13 14:15:00 | 4085.00 | 4004.19 | 3964.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-14 13:15:00 | 4132.55 | 4152.20 | 4069.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-14 14:00:00 | 4132.55 | 4152.20 | 4069.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 14:15:00 | 4092.20 | 4140.20 | 4071.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-14 15:00:00 | 4092.20 | 4140.20 | 4071.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-14 15:15:00 | 4098.00 | 4131.76 | 4074.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-15 09:15:00 | 4142.40 | 4131.76 | 4074.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-15 14:15:00 | 4022.55 | 4084.45 | 4073.97 | SL hit (close<static) qty=1.00 sl=4056.50 alert=retest2 |

### Cycle 7 — SELL (started 2023-06-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-15 15:15:00 | 3992.00 | 4065.96 | 4066.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-16 09:15:00 | 3972.00 | 4047.17 | 4057.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-16 10:15:00 | 4060.70 | 4049.87 | 4058.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-16 10:15:00 | 4060.70 | 4049.87 | 4058.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 10:15:00 | 4060.70 | 4049.87 | 4058.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-16 11:00:00 | 4060.70 | 4049.87 | 4058.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-16 11:15:00 | 4052.00 | 4050.30 | 4057.62 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2023-06-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-16 15:15:00 | 4080.00 | 4063.06 | 4061.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-19 09:15:00 | 4140.00 | 4078.45 | 4068.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-19 12:15:00 | 4096.35 | 4100.12 | 4082.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-19 13:00:00 | 4096.35 | 4100.12 | 4082.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 13:15:00 | 4112.00 | 4102.50 | 4085.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 13:30:00 | 4092.00 | 4102.50 | 4085.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 09:15:00 | 4105.65 | 4125.76 | 4113.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-21 10:00:00 | 4105.65 | 4125.76 | 4113.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 10:15:00 | 4103.70 | 4121.35 | 4112.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-21 10:30:00 | 4106.80 | 4121.35 | 4112.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-21 11:15:00 | 4118.90 | 4120.86 | 4113.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-21 12:15:00 | 4150.35 | 4120.86 | 4113.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-22 13:30:00 | 4124.40 | 4140.19 | 4133.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-22 14:30:00 | 4128.05 | 4137.97 | 4132.72 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-22 15:00:00 | 4129.10 | 4137.97 | 4132.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-22 15:15:00 | 4140.00 | 4138.38 | 4133.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 09:15:00 | 4085.90 | 4138.38 | 4133.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 09:15:00 | 4106.60 | 4132.02 | 4130.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-23 09:30:00 | 4055.35 | 4132.02 | 4130.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-23 10:15:00 | 4129.05 | 4131.43 | 4130.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-06-23 11:30:00 | 4140.00 | 4134.54 | 4132.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-23 14:15:00 | 4090.05 | 4136.19 | 4134.75 | SL hit (close<static) qty=1.00 sl=4104.25 alert=retest2 |

### Cycle 9 — SELL (started 2023-06-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-23 15:15:00 | 4100.00 | 4128.95 | 4131.59 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2023-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-26 11:15:00 | 4174.95 | 4140.10 | 4136.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-26 13:15:00 | 4187.05 | 4154.27 | 4143.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-27 09:15:00 | 4154.95 | 4166.77 | 4153.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-27 09:15:00 | 4154.95 | 4166.77 | 4153.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 4154.95 | 4166.77 | 4153.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 09:30:00 | 4166.00 | 4166.77 | 4153.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 4114.00 | 4156.22 | 4149.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 11:00:00 | 4114.00 | 4156.22 | 4149.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 11:15:00 | 4105.90 | 4146.15 | 4145.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-27 11:45:00 | 4095.00 | 4146.15 | 4145.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2023-06-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-27 12:15:00 | 4095.80 | 4136.08 | 4140.98 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-06-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-28 11:15:00 | 4192.55 | 4144.00 | 4141.09 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2023-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-28 14:15:00 | 4115.35 | 4137.91 | 4139.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-30 10:15:00 | 4107.00 | 4128.71 | 4134.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-30 12:15:00 | 4138.00 | 4128.54 | 4133.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-30 12:15:00 | 4138.00 | 4128.54 | 4133.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 12:15:00 | 4138.00 | 4128.54 | 4133.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-30 13:00:00 | 4138.00 | 4128.54 | 4133.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 13:15:00 | 4121.00 | 4127.04 | 4132.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-30 13:30:00 | 4141.90 | 4127.04 | 4132.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 14:15:00 | 4150.00 | 4131.63 | 4133.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-30 15:00:00 | 4150.00 | 4131.63 | 4133.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 15:15:00 | 4142.00 | 4133.70 | 4134.46 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2023-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-03 09:15:00 | 4142.90 | 4135.54 | 4135.23 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-04 14:15:00 | 4110.65 | 4139.58 | 4140.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-04 15:15:00 | 4105.50 | 4132.76 | 4137.75 | Break + close below crossover candle low |

### Cycle 16 — BUY (started 2023-07-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-05 12:15:00 | 4287.45 | 4157.36 | 4146.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 09:15:00 | 4337.00 | 4213.14 | 4177.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 10:15:00 | 4270.00 | 4287.69 | 4247.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-07 10:15:00 | 4270.00 | 4287.69 | 4247.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 10:15:00 | 4270.00 | 4287.69 | 4247.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 10:45:00 | 4266.95 | 4287.69 | 4247.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 12:15:00 | 4258.95 | 4281.49 | 4251.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 12:30:00 | 4255.05 | 4281.49 | 4251.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 14:15:00 | 4260.40 | 4285.24 | 4258.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 15:00:00 | 4260.40 | 4285.24 | 4258.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 15:15:00 | 4241.65 | 4276.52 | 4257.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-10 09:15:00 | 4317.35 | 4276.52 | 4257.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-10 15:15:00 | 4211.00 | 4261.06 | 4260.36 | SL hit (close<static) qty=1.00 sl=4231.50 alert=retest2 |

### Cycle 17 — SELL (started 2023-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-11 09:15:00 | 4247.45 | 4258.34 | 4259.18 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 10:15:00 | 4288.00 | 4264.27 | 4261.80 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2023-07-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-11 11:15:00 | 4230.55 | 4257.53 | 4258.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-11 13:15:00 | 4193.00 | 4239.41 | 4250.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-12 09:15:00 | 4227.40 | 4221.54 | 4237.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-12 10:00:00 | 4227.40 | 4221.54 | 4237.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 10:15:00 | 4246.85 | 4226.60 | 4238.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 10:30:00 | 4263.05 | 4226.60 | 4238.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-12 11:15:00 | 4260.10 | 4233.30 | 4240.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-12 11:45:00 | 4268.60 | 4233.30 | 4240.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — BUY (started 2023-07-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-12 13:15:00 | 4258.95 | 4245.35 | 4245.30 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-07-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-12 14:15:00 | 4243.90 | 4245.06 | 4245.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-12 15:15:00 | 4199.95 | 4236.04 | 4241.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 09:15:00 | 4214.00 | 4196.69 | 4215.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-14 09:15:00 | 4214.00 | 4196.69 | 4215.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 09:15:00 | 4214.00 | 4196.69 | 4215.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 09:45:00 | 4203.50 | 4196.69 | 4215.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 10:15:00 | 4228.15 | 4202.98 | 4216.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 10:45:00 | 4232.90 | 4202.98 | 4216.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 11:15:00 | 4225.20 | 4207.43 | 4217.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 11:30:00 | 4243.95 | 4207.43 | 4217.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 12:15:00 | 4220.00 | 4209.94 | 4217.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 12:30:00 | 4230.00 | 4209.94 | 4217.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 13:15:00 | 4226.40 | 4213.23 | 4218.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 13:30:00 | 4227.35 | 4213.23 | 4218.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 14:15:00 | 4217.10 | 4214.01 | 4218.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-14 15:00:00 | 4217.10 | 4214.01 | 4218.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-14 15:15:00 | 4225.05 | 4216.22 | 4218.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 09:15:00 | 4246.00 | 4216.22 | 4218.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 4217.00 | 4216.37 | 4218.61 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 10:15:00 | 4255.25 | 4224.15 | 4221.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-18 09:15:00 | 4289.75 | 4250.37 | 4236.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-18 12:15:00 | 4265.05 | 4267.05 | 4249.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-18 12:15:00 | 4265.05 | 4267.05 | 4249.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 12:15:00 | 4265.05 | 4267.05 | 4249.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 13:00:00 | 4265.05 | 4267.05 | 4249.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 13:15:00 | 4242.20 | 4262.08 | 4248.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 14:00:00 | 4242.20 | 4262.08 | 4248.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-18 14:15:00 | 4238.00 | 4257.26 | 4247.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-18 15:15:00 | 4237.00 | 4257.26 | 4247.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 10:15:00 | 4262.70 | 4255.80 | 4249.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-19 10:30:00 | 4255.65 | 4255.80 | 4249.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 11:15:00 | 4249.25 | 4254.49 | 4249.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-19 11:30:00 | 4245.00 | 4254.49 | 4249.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-19 12:15:00 | 4231.75 | 4249.94 | 4247.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-19 13:00:00 | 4231.75 | 4249.94 | 4247.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2023-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-19 13:15:00 | 4199.40 | 4239.83 | 4243.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-20 09:15:00 | 4171.05 | 4215.76 | 4230.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-20 15:15:00 | 4180.00 | 4179.12 | 4202.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-20 15:15:00 | 4180.00 | 4179.12 | 4202.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-20 15:15:00 | 4180.00 | 4179.12 | 4202.18 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-24 10:15:00 | 4221.50 | 4203.30 | 4202.19 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 12:15:00 | 4185.00 | 4199.35 | 4200.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-24 14:15:00 | 4114.95 | 4180.98 | 4191.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-25 15:15:00 | 4195.00 | 4146.87 | 4162.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-25 15:15:00 | 4195.00 | 4146.87 | 4162.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-25 15:15:00 | 4195.00 | 4146.87 | 4162.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 10:30:00 | 4011.00 | 4071.46 | 4106.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-27 12:15:00 | 4023.05 | 4067.00 | 4100.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-28 09:15:00 | 4030.50 | 4057.41 | 4084.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-07-28 11:15:00 | 4025.00 | 4054.52 | 4078.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 13:15:00 | 3973.55 | 3962.57 | 3983.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-01 13:45:00 | 3973.00 | 3962.57 | 3983.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-01 15:15:00 | 3990.00 | 3967.65 | 3982.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-02 09:15:00 | 4000.95 | 3967.65 | 3982.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-02 09:15:00 | 4000.00 | 3974.12 | 3983.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-02 10:30:00 | 3953.00 | 3967.25 | 3979.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-02 14:15:00 | 4040.00 | 3979.50 | 3980.41 | SL hit (close>static) qty=1.00 sl=4019.95 alert=retest2 |

### Cycle 26 — BUY (started 2023-08-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-02 15:15:00 | 4015.05 | 3986.61 | 3983.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-03 09:15:00 | 4048.00 | 3998.88 | 3989.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-04 11:15:00 | 4180.25 | 4186.69 | 4115.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-04 11:45:00 | 4185.00 | 4186.69 | 4115.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 09:15:00 | 4188.60 | 4190.18 | 4166.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 10:15:00 | 4165.00 | 4190.18 | 4166.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 10:15:00 | 4160.20 | 4184.18 | 4166.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-08 11:15:00 | 4164.05 | 4184.18 | 4166.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-08 11:15:00 | 4213.40 | 4190.03 | 4170.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-08-08 13:00:00 | 4255.00 | 4203.02 | 4178.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-14 12:15:00 | 4330.10 | 4365.67 | 4369.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2023-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-14 12:15:00 | 4330.10 | 4365.67 | 4369.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-14 14:15:00 | 4291.75 | 4346.75 | 4359.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-17 10:15:00 | 4279.35 | 4277.17 | 4305.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-17 11:00:00 | 4279.35 | 4277.17 | 4305.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-17 11:15:00 | 4280.70 | 4277.88 | 4303.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-18 15:15:00 | 4240.00 | 4295.95 | 4302.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-21 13:15:00 | 4340.10 | 4299.41 | 4298.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2023-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 13:15:00 | 4340.10 | 4299.41 | 4298.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-21 14:15:00 | 4370.00 | 4313.53 | 4304.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-23 15:15:00 | 4585.05 | 4594.07 | 4509.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-24 09:15:00 | 4590.80 | 4594.07 | 4509.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 14:15:00 | 4549.20 | 4569.69 | 4533.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-24 15:00:00 | 4549.20 | 4569.69 | 4533.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-24 15:15:00 | 4650.00 | 4585.76 | 4543.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-08-25 09:15:00 | 4645.00 | 4585.76 | 4543.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-25 09:15:00 | 4600.00 | 4588.60 | 4549.03 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 15:15:00 | 4535.00 | 4571.83 | 4576.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-30 09:15:00 | 4505.10 | 4558.48 | 4570.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-01 09:15:00 | 4432.80 | 4419.91 | 4461.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-09-01 09:30:00 | 4433.85 | 4419.91 | 4461.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-04 09:15:00 | 4342.15 | 4381.92 | 4419.95 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2023-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-06 09:15:00 | 4430.00 | 4414.66 | 4414.57 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-06 10:15:00 | 4353.60 | 4402.45 | 4409.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-06 11:15:00 | 4294.15 | 4380.79 | 4398.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-06 15:15:00 | 4355.00 | 4351.18 | 4375.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-06 15:15:00 | 4355.00 | 4351.18 | 4375.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-06 15:15:00 | 4355.00 | 4351.18 | 4375.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-07 09:15:00 | 4416.80 | 4351.18 | 4375.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 09:15:00 | 4383.85 | 4357.71 | 4376.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-07 09:45:00 | 4417.85 | 4357.71 | 4376.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-07 10:15:00 | 4383.45 | 4362.86 | 4377.21 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2023-09-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-07 13:15:00 | 4420.05 | 4386.45 | 4385.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-07 14:15:00 | 4449.85 | 4399.13 | 4391.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-11 13:15:00 | 4578.80 | 4589.70 | 4535.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-11 14:00:00 | 4578.80 | 4589.70 | 4535.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 09:15:00 | 4494.05 | 4564.56 | 4536.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 09:45:00 | 4473.45 | 4564.56 | 4536.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-12 10:15:00 | 4419.65 | 4535.58 | 4526.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-12 10:30:00 | 4429.90 | 4535.58 | 4526.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2023-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 11:15:00 | 4397.05 | 4507.87 | 4514.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-12 13:15:00 | 4375.00 | 4463.24 | 4491.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-14 09:15:00 | 4417.45 | 4366.44 | 4405.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-14 09:15:00 | 4417.45 | 4366.44 | 4405.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 09:15:00 | 4417.45 | 4366.44 | 4405.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 09:30:00 | 4430.00 | 4366.44 | 4405.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-14 10:15:00 | 4390.00 | 4371.15 | 4403.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-14 10:30:00 | 4395.15 | 4371.15 | 4403.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 09:15:00 | 4454.65 | 4390.56 | 4399.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-09-15 09:30:00 | 4446.00 | 4390.56 | 4399.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-15 10:15:00 | 4440.00 | 4400.45 | 4403.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-09-15 12:00:00 | 4400.15 | 4400.39 | 4403.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2023-09-18 14:15:00 | 4180.14 | 4267.41 | 4320.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2023-09-21 09:15:00 | 4228.90 | 4205.49 | 4249.78 | SL hit (close>ema200) qty=0.50 sl=4205.49 alert=retest2 |

### Cycle 34 — BUY (started 2023-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 15:15:00 | 4010.20 | 3967.91 | 3966.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 09:15:00 | 4053.55 | 3985.04 | 3974.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-18 10:15:00 | 4556.35 | 4560.75 | 4499.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-18 10:30:00 | 4562.70 | 4560.75 | 4499.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 09:15:00 | 4577.75 | 4594.49 | 4562.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-20 09:30:00 | 4570.00 | 4594.49 | 4562.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 10:15:00 | 4568.90 | 4589.37 | 4562.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-20 10:45:00 | 4572.20 | 4589.37 | 4562.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 11:15:00 | 4566.60 | 4584.82 | 4563.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-20 11:30:00 | 4556.00 | 4584.82 | 4563.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 12:15:00 | 4555.15 | 4578.88 | 4562.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-20 13:00:00 | 4555.15 | 4578.88 | 4562.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 13:15:00 | 4559.15 | 4574.94 | 4562.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-20 14:00:00 | 4559.15 | 4574.94 | 4562.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 14:15:00 | 4551.45 | 4570.24 | 4561.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-20 14:30:00 | 4553.70 | 4570.24 | 4561.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-20 15:15:00 | 4554.60 | 4567.11 | 4560.55 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-23 09:15:00 | 4460.55 | 4545.80 | 4551.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-23 11:15:00 | 4384.40 | 4500.14 | 4528.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-25 15:15:00 | 4300.00 | 4297.61 | 4368.42 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-10-26 09:15:00 | 4174.15 | 4297.61 | 4368.42 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 4287.45 | 4193.69 | 4256.65 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-10-27 09:15:00 | 4287.45 | 4193.69 | 4256.65 | SL hit (close>ema400) qty=1.00 sl=4256.65 alert=retest1 |

### Cycle 36 — BUY (started 2023-11-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 10:15:00 | 4300.05 | 4192.43 | 4190.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 11:15:00 | 4327.00 | 4219.34 | 4202.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-06 10:15:00 | 4408.40 | 4427.95 | 4370.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-06 11:00:00 | 4408.40 | 4427.95 | 4370.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 15:15:00 | 4408.75 | 4422.55 | 4390.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 09:15:00 | 4348.00 | 4422.55 | 4390.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 09:15:00 | 4412.25 | 4420.49 | 4392.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 09:30:00 | 4351.70 | 4420.49 | 4392.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-07 11:15:00 | 4404.90 | 4418.86 | 4396.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-07 12:00:00 | 4404.90 | 4418.86 | 4396.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 09:15:00 | 4565.80 | 4567.01 | 4517.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 09:45:00 | 4532.55 | 4567.01 | 4517.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 10:15:00 | 4502.55 | 4554.12 | 4516.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 10:45:00 | 4500.20 | 4554.12 | 4516.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 11:15:00 | 4500.05 | 4543.31 | 4515.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 12:00:00 | 4500.05 | 4543.31 | 4515.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 12:15:00 | 4499.00 | 4534.44 | 4513.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 13:00:00 | 4499.00 | 4534.44 | 4513.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 13:15:00 | 4483.20 | 4524.20 | 4510.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 14:00:00 | 4483.20 | 4524.20 | 4510.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-09 14:15:00 | 4475.25 | 4514.41 | 4507.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-09 15:00:00 | 4475.25 | 4514.41 | 4507.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2023-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-10 09:15:00 | 4455.00 | 4493.82 | 4498.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-10 11:15:00 | 4401.00 | 4473.08 | 4488.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-12 18:15:00 | 4416.75 | 4415.98 | 4449.60 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2023-11-13 09:15:00 | 4382.80 | 4415.98 | 4449.60 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-13 11:15:00 | 4452.00 | 4407.35 | 4435.96 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2023-11-13 11:15:00 | 4452.00 | 4407.35 | 4435.96 | SL hit (close>ema400) qty=1.00 sl=4435.96 alert=retest1 |

### Cycle 38 — BUY (started 2023-11-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-16 10:15:00 | 4557.15 | 4432.94 | 4420.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-16 11:15:00 | 4598.95 | 4466.14 | 4436.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-20 14:15:00 | 4620.90 | 4626.40 | 4580.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-20 15:00:00 | 4620.90 | 4626.40 | 4580.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 14:15:00 | 4633.00 | 4653.45 | 4619.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-21 15:15:00 | 4644.00 | 4653.45 | 4619.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-21 15:15:00 | 4644.00 | 4651.56 | 4621.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 09:15:00 | 4614.00 | 4651.56 | 4621.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 09:15:00 | 4600.55 | 4641.36 | 4619.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-22 10:15:00 | 4588.65 | 4641.36 | 4619.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-22 10:15:00 | 4596.35 | 4632.36 | 4617.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-22 11:15:00 | 4631.15 | 4632.36 | 4617.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-29 09:15:00 | 4756.35 | 4791.21 | 4791.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2023-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-29 09:15:00 | 4756.35 | 4791.21 | 4791.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-29 12:15:00 | 4715.00 | 4763.37 | 4777.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-30 10:15:00 | 4740.00 | 4737.60 | 4757.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-30 11:00:00 | 4740.00 | 4737.60 | 4757.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 09:15:00 | 4716.75 | 4702.17 | 4727.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-01 09:30:00 | 4728.35 | 4702.17 | 4727.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 10:15:00 | 4738.80 | 4709.49 | 4728.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-01 10:45:00 | 4736.45 | 4709.49 | 4728.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-01 11:15:00 | 4824.20 | 4732.43 | 4737.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-01 11:30:00 | 4823.45 | 4732.43 | 4737.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2023-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-01 12:15:00 | 4800.00 | 4745.95 | 4742.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-04 09:15:00 | 4841.70 | 4767.47 | 4754.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-04 15:15:00 | 4800.00 | 4801.24 | 4780.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-04 15:15:00 | 4800.00 | 4801.24 | 4780.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-04 15:15:00 | 4800.00 | 4801.24 | 4780.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 09:15:00 | 4865.90 | 4801.24 | 4780.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-06 10:15:00 | 4850.95 | 4819.50 | 4805.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-06 12:15:00 | 4757.85 | 4803.34 | 4801.59 | SL hit (close<static) qty=1.00 sl=4765.00 alert=retest2 |

### Cycle 41 — SELL (started 2023-12-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 13:15:00 | 4782.50 | 4799.17 | 4799.86 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2023-12-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-06 14:15:00 | 4823.00 | 4803.94 | 4801.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-07 09:15:00 | 4853.10 | 4815.38 | 4807.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-07 11:15:00 | 4818.00 | 4819.52 | 4811.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-07 11:45:00 | 4832.95 | 4819.52 | 4811.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 13:15:00 | 4836.20 | 4857.61 | 4841.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 13:30:00 | 4835.75 | 4857.61 | 4841.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 14:15:00 | 4834.60 | 4853.01 | 4840.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-08 15:15:00 | 4821.00 | 4853.01 | 4840.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 15:15:00 | 4821.00 | 4846.61 | 4838.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-11 09:15:00 | 4874.20 | 4846.61 | 4838.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-11 11:15:00 | 4803.10 | 4833.24 | 4834.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2023-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-11 11:15:00 | 4803.10 | 4833.24 | 4834.17 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2023-12-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 12:15:00 | 4859.10 | 4838.41 | 4836.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-12 11:15:00 | 4900.00 | 4861.45 | 4849.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 14:15:00 | 4855.70 | 4871.46 | 4858.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 14:15:00 | 4855.70 | 4871.46 | 4858.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 14:15:00 | 4855.70 | 4871.46 | 4858.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 14:45:00 | 4851.65 | 4871.46 | 4858.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 15:15:00 | 4900.00 | 4877.17 | 4861.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-13 09:30:00 | 4917.20 | 4878.60 | 4863.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-13 10:30:00 | 4900.40 | 4883.49 | 4867.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-19 14:15:00 | 5066.85 | 5114.13 | 5115.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2023-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-19 14:15:00 | 5066.85 | 5114.13 | 5115.04 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2023-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-20 10:15:00 | 5172.40 | 5118.55 | 5116.00 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2023-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 12:15:00 | 5085.10 | 5114.27 | 5114.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 13:15:00 | 5024.20 | 5096.25 | 5106.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 09:15:00 | 5094.35 | 5070.84 | 5090.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-21 09:15:00 | 5094.35 | 5070.84 | 5090.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 09:15:00 | 5094.35 | 5070.84 | 5090.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 10:00:00 | 5094.35 | 5070.84 | 5090.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 10:15:00 | 5100.00 | 5076.67 | 5090.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 11:00:00 | 5100.00 | 5076.67 | 5090.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 11:15:00 | 5050.00 | 5071.34 | 5087.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-21 12:15:00 | 5006.75 | 5071.34 | 5087.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-21 13:30:00 | 5027.70 | 5051.63 | 5074.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-12-22 09:15:00 | 5046.35 | 5056.80 | 5073.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-22 11:15:00 | 5104.95 | 5067.08 | 5073.91 | SL hit (close>static) qty=1.00 sl=5102.00 alert=retest2 |

### Cycle 48 — BUY (started 2023-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 13:15:00 | 5114.35 | 5080.21 | 5078.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 14:15:00 | 5130.40 | 5090.25 | 5083.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-26 14:15:00 | 5155.00 | 5184.95 | 5146.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-26 14:15:00 | 5155.00 | 5184.95 | 5146.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 14:15:00 | 5155.00 | 5184.95 | 5146.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-26 15:00:00 | 5155.00 | 5184.95 | 5146.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-26 15:15:00 | 5198.00 | 5187.56 | 5151.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-27 09:15:00 | 5255.00 | 5187.56 | 5151.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-09 10:15:00 | 5780.50 | 5660.22 | 5616.17 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2024-01-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-17 13:15:00 | 5884.25 | 5922.34 | 5925.25 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2024-01-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-18 10:15:00 | 5981.60 | 5927.75 | 5925.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-19 09:15:00 | 6074.90 | 5969.52 | 5947.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-20 10:15:00 | 6075.00 | 6105.85 | 6051.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-20 11:00:00 | 6075.00 | 6105.85 | 6051.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 11:15:00 | 6083.40 | 6101.36 | 6054.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 11:45:00 | 6059.70 | 6101.36 | 6054.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 12:15:00 | 6129.60 | 6107.01 | 6061.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 12:30:00 | 6054.95 | 6107.01 | 6061.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 14:15:00 | 6031.50 | 6098.69 | 6065.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-20 15:00:00 | 6031.50 | 6098.69 | 6065.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-20 15:15:00 | 6049.00 | 6088.75 | 6064.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-23 09:15:00 | 5967.45 | 6088.75 | 6064.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2024-01-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-23 09:15:00 | 5882.85 | 6047.57 | 6047.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-23 11:15:00 | 5824.30 | 5973.53 | 6012.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-25 09:15:00 | 5920.00 | 5747.00 | 5814.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-25 09:15:00 | 5920.00 | 5747.00 | 5814.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 09:15:00 | 5920.00 | 5747.00 | 5814.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 10:00:00 | 5920.00 | 5747.00 | 5814.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-25 10:15:00 | 5936.20 | 5784.84 | 5825.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-25 10:45:00 | 5949.10 | 5784.84 | 5825.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — BUY (started 2024-01-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-25 14:15:00 | 5895.00 | 5856.05 | 5851.72 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-29 12:15:00 | 5827.25 | 5846.89 | 5849.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-29 14:15:00 | 5780.00 | 5828.45 | 5840.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-31 09:15:00 | 5883.45 | 5735.24 | 5770.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-31 09:15:00 | 5883.45 | 5735.24 | 5770.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 09:15:00 | 5883.45 | 5735.24 | 5770.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-31 10:00:00 | 5883.45 | 5735.24 | 5770.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-31 10:15:00 | 5845.00 | 5757.19 | 5777.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-01-31 11:45:00 | 5798.90 | 5766.76 | 5779.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-31 13:15:00 | 5909.90 | 5810.41 | 5798.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2024-01-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-31 13:15:00 | 5909.90 | 5810.41 | 5798.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-31 14:15:00 | 5928.50 | 5834.03 | 5810.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-01 09:15:00 | 5826.65 | 5850.38 | 5822.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-01 09:15:00 | 5826.65 | 5850.38 | 5822.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 09:15:00 | 5826.65 | 5850.38 | 5822.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 10:15:00 | 5820.85 | 5850.38 | 5822.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 10:15:00 | 5823.30 | 5844.96 | 5822.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-01 11:00:00 | 5823.30 | 5844.96 | 5822.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-01 11:15:00 | 5797.00 | 5835.37 | 5820.43 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2024-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-01 13:15:00 | 5740.10 | 5808.20 | 5810.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-01 14:15:00 | 5711.45 | 5788.85 | 5801.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-02 09:15:00 | 5809.10 | 5785.08 | 5796.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-02 09:15:00 | 5809.10 | 5785.08 | 5796.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-02 09:15:00 | 5809.10 | 5785.08 | 5796.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-02 10:30:00 | 5774.00 | 5780.75 | 5793.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-02 12:45:00 | 5767.70 | 5769.59 | 5786.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-05 09:30:00 | 5767.55 | 5754.26 | 5771.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-05 11:15:00 | 6021.15 | 5817.55 | 5797.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2024-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-05 11:15:00 | 6021.15 | 5817.55 | 5797.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-05 14:15:00 | 6118.85 | 5946.04 | 5867.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-06 14:15:00 | 6130.00 | 6178.21 | 6057.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-06 14:30:00 | 6169.70 | 6178.21 | 6057.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 15:15:00 | 6250.05 | 6271.21 | 6222.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 09:15:00 | 6167.00 | 6271.21 | 6222.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 09:15:00 | 6133.75 | 6243.72 | 6214.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-09 10:00:00 | 6133.75 | 6243.72 | 6214.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-09 10:15:00 | 6170.05 | 6228.98 | 6210.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 12:30:00 | 6230.25 | 6224.44 | 6210.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-09 15:15:00 | 6197.20 | 6212.35 | 6207.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-02-12 09:30:00 | 6249.15 | 6224.74 | 6213.69 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-12 12:15:00 | 6150.05 | 6195.96 | 6202.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2024-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-12 12:15:00 | 6150.05 | 6195.96 | 6202.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-12 13:15:00 | 6055.60 | 6167.88 | 6188.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-14 09:15:00 | 5981.70 | 5979.22 | 6044.15 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-02-14 14:15:00 | 5899.50 | 5968.00 | 6018.02 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 10:15:00 | 5912.65 | 5928.40 | 5979.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-15 10:45:00 | 5930.70 | 5928.40 | 5979.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 13:15:00 | 5905.00 | 5918.10 | 5962.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-15 14:45:00 | 5843.90 | 5906.23 | 5952.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-02-16 09:15:00 | 6033.40 | 5928.24 | 5954.36 | SL hit (close>ema400) qty=1.00 sl=5954.36 alert=retest1 |

### Cycle 58 — BUY (started 2024-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-16 12:15:00 | 6030.00 | 5972.19 | 5970.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-19 12:15:00 | 6059.95 | 6012.07 | 5993.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-20 13:15:00 | 6140.40 | 6175.12 | 6110.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-02-20 14:15:00 | 6098.70 | 6159.84 | 6109.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 14:15:00 | 6098.70 | 6159.84 | 6109.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-20 15:00:00 | 6098.70 | 6159.84 | 6109.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-20 15:15:00 | 6147.00 | 6157.27 | 6112.88 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-21 14:15:00 | 5895.00 | 6075.12 | 6093.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-21 15:15:00 | 5875.00 | 6035.09 | 6073.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-23 10:15:00 | 5883.70 | 5870.94 | 5940.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-23 11:00:00 | 5883.70 | 5870.94 | 5940.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 11:15:00 | 5922.20 | 5881.20 | 5938.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-23 13:15:00 | 5865.25 | 5896.75 | 5940.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-26 11:45:00 | 5872.15 | 5889.64 | 5916.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-26 13:30:00 | 5874.35 | 5888.69 | 5911.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-02-26 14:30:00 | 5879.80 | 5888.77 | 5909.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-26 15:15:00 | 5929.95 | 5897.01 | 5911.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-27 10:00:00 | 5930.00 | 5903.61 | 5912.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-02-27 10:15:00 | 6000.00 | 5922.88 | 5920.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — BUY (started 2024-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-27 10:15:00 | 6000.00 | 5922.88 | 5920.82 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2024-02-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-27 13:15:00 | 5836.65 | 5912.49 | 5917.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-28 12:15:00 | 5813.90 | 5869.70 | 5891.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-28 13:15:00 | 5875.35 | 5870.83 | 5889.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-28 14:00:00 | 5875.35 | 5870.83 | 5889.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 14:15:00 | 5909.40 | 5878.54 | 5891.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-28 15:00:00 | 5909.40 | 5878.54 | 5891.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-28 15:15:00 | 5910.00 | 5884.83 | 5893.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-29 09:15:00 | 5864.90 | 5884.83 | 5893.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-29 09:15:00 | 5886.75 | 5885.22 | 5892.75 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-02-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-29 14:15:00 | 6013.70 | 5906.53 | 5899.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-01 09:15:00 | 6059.70 | 5950.52 | 5921.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-04 14:15:00 | 6086.95 | 6129.85 | 6079.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-04 14:15:00 | 6086.95 | 6129.85 | 6079.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 14:15:00 | 6086.95 | 6129.85 | 6079.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-04 14:30:00 | 6066.30 | 6129.85 | 6079.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-04 15:15:00 | 6160.00 | 6135.88 | 6086.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 09:15:00 | 6184.55 | 6135.88 | 6086.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 10:15:00 | 6169.10 | 6135.74 | 6090.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-05 11:00:00 | 6160.15 | 6140.63 | 6097.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-06 12:15:00 | 6180.00 | 6225.44 | 6178.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 12:15:00 | 6233.15 | 6226.98 | 6183.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-06 14:00:00 | 6271.90 | 6235.96 | 6191.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-03-11 10:15:00 | 6786.01 | 6557.43 | 6422.08 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2024-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 10:15:00 | 6310.30 | 6437.03 | 6451.54 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-03-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-14 11:15:00 | 6577.20 | 6451.38 | 6444.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-14 13:15:00 | 6607.55 | 6504.79 | 6471.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-15 13:15:00 | 6607.40 | 6620.72 | 6563.83 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-15 14:00:00 | 6607.40 | 6620.72 | 6563.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-15 14:15:00 | 6643.30 | 6625.24 | 6571.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-15 14:45:00 | 6522.00 | 6625.24 | 6571.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 09:15:00 | 6528.00 | 6608.15 | 6572.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-18 09:30:00 | 6464.15 | 6608.15 | 6572.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 10:15:00 | 6576.20 | 6601.76 | 6573.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-18 11:45:00 | 6630.25 | 6606.73 | 6578.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-18 12:45:00 | 6627.30 | 6601.62 | 6578.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-18 14:15:00 | 6507.55 | 6574.55 | 6569.52 | SL hit (close<static) qty=1.00 sl=6525.70 alert=retest2 |

### Cycle 65 — SELL (started 2024-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-19 09:15:00 | 6501.85 | 6555.88 | 6561.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-20 09:15:00 | 6416.05 | 6496.42 | 6525.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-20 12:15:00 | 6570.05 | 6483.69 | 6509.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-20 12:15:00 | 6570.05 | 6483.69 | 6509.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 12:15:00 | 6570.05 | 6483.69 | 6509.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 13:00:00 | 6570.05 | 6483.69 | 6509.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 13:15:00 | 6630.00 | 6512.95 | 6520.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-20 14:00:00 | 6630.00 | 6512.95 | 6520.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2024-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-20 14:15:00 | 6658.00 | 6541.96 | 6533.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-21 09:15:00 | 6985.00 | 6646.26 | 6582.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-22 15:15:00 | 7125.00 | 7144.82 | 6992.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-26 09:15:00 | 7019.50 | 7144.82 | 6992.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-26 09:15:00 | 7175.10 | 7150.88 | 7008.75 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2024-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-28 14:15:00 | 6977.00 | 7017.55 | 7019.80 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2024-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-01 09:15:00 | 7145.00 | 7036.23 | 7027.48 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-04-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-02 11:15:00 | 6940.75 | 7018.77 | 7024.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-02 12:15:00 | 6917.00 | 6998.42 | 7014.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-03 13:15:00 | 6920.60 | 6914.02 | 6948.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-03 14:15:00 | 6935.00 | 6918.21 | 6947.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 14:15:00 | 6935.00 | 6918.21 | 6947.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-03 15:00:00 | 6935.00 | 6918.21 | 6947.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 15:15:00 | 6930.00 | 6920.57 | 6946.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-04 09:15:00 | 7033.00 | 6920.57 | 6946.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 09:15:00 | 6955.00 | 6927.46 | 6946.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-04 09:30:00 | 6953.35 | 6927.46 | 6946.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 10:15:00 | 6922.25 | 6926.42 | 6944.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-04 10:45:00 | 6920.00 | 6926.42 | 6944.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 11:15:00 | 6890.35 | 6919.20 | 6939.68 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2024-04-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-05 10:15:00 | 7104.65 | 6964.93 | 6950.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-08 09:15:00 | 7454.80 | 7151.61 | 7058.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-10 09:15:00 | 7541.00 | 7654.30 | 7507.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-10 09:45:00 | 7550.00 | 7654.30 | 7507.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 11:15:00 | 7511.25 | 7604.35 | 7508.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-10 11:30:00 | 7530.65 | 7604.35 | 7508.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-10 12:15:00 | 7575.00 | 7598.48 | 7514.62 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2024-04-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 11:15:00 | 7420.65 | 7492.12 | 7492.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-12 14:15:00 | 7349.00 | 7455.35 | 7474.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 09:15:00 | 7405.60 | 7310.34 | 7364.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-16 09:15:00 | 7405.60 | 7310.34 | 7364.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 09:15:00 | 7405.60 | 7310.34 | 7364.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 10:00:00 | 7405.60 | 7310.34 | 7364.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 7425.10 | 7333.29 | 7370.44 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2024-04-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-16 13:15:00 | 7519.20 | 7397.07 | 7392.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-18 09:15:00 | 7795.80 | 7519.15 | 7452.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-22 09:15:00 | 8199.90 | 8209.15 | 8018.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-04-22 09:30:00 | 8200.00 | 8209.15 | 8018.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 09:15:00 | 7987.60 | 8129.21 | 8070.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-23 10:00:00 | 7987.60 | 8129.21 | 8070.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-23 10:15:00 | 8000.00 | 8103.37 | 8063.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-23 11:30:00 | 8033.55 | 8096.27 | 8064.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-04-25 11:15:00 | 8836.91 | 8498.21 | 8300.33 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2024-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-06 15:15:00 | 9235.15 | 9467.33 | 9493.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 10:15:00 | 8915.00 | 9317.29 | 9418.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-08 09:15:00 | 9082.00 | 9046.80 | 9211.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-08 10:15:00 | 9353.30 | 9108.10 | 9224.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 10:15:00 | 9353.30 | 9108.10 | 9224.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-08 11:00:00 | 9353.30 | 9108.10 | 9224.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-08 11:15:00 | 9120.25 | 9110.53 | 9215.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-08 13:00:00 | 9019.15 | 9092.25 | 9197.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-09 10:00:00 | 9051.20 | 9043.22 | 9136.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-13 12:15:00 | 9309.00 | 9021.68 | 9009.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2024-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 12:15:00 | 9309.00 | 9021.68 | 9009.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 13:15:00 | 9330.00 | 9083.34 | 9038.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 14:15:00 | 11214.00 | 11242.93 | 10642.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 15:00:00 | 11214.00 | 11242.93 | 10642.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 10999.65 | 11198.13 | 11019.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:45:00 | 11004.00 | 11198.13 | 11019.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 10920.00 | 11142.50 | 11010.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:15:00 | 11403.55 | 11000.59 | 10983.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-22 10:15:00 | 10611.95 | 10941.98 | 10960.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 10:15:00 | 10611.95 | 10941.98 | 10960.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 12:15:00 | 10439.90 | 10786.85 | 10883.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 11:15:00 | 10745.90 | 10617.06 | 10730.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 11:15:00 | 10745.90 | 10617.06 | 10730.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 10745.90 | 10617.06 | 10730.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 12:00:00 | 10745.90 | 10617.06 | 10730.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 12:15:00 | 10870.00 | 10667.65 | 10743.24 | EMA400 retest candle locked (from downside) |

### Cycle 76 — BUY (started 2024-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 14:15:00 | 11030.00 | 10806.90 | 10797.55 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2024-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 15:15:00 | 10800.00 | 10891.79 | 10903.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 09:15:00 | 10451.30 | 10803.69 | 10862.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 10856.85 | 10557.38 | 10664.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 10856.85 | 10557.38 | 10664.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 10856.85 | 10557.38 | 10664.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 09:45:00 | 10916.50 | 10557.38 | 10664.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 10765.15 | 10598.93 | 10673.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 11:45:00 | 10741.10 | 10618.99 | 10675.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 13:00:00 | 10700.00 | 10635.19 | 10678.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 10:15:00 | 10721.55 | 10695.18 | 10694.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2024-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-30 10:15:00 | 10721.55 | 10695.18 | 10694.54 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2024-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 11:15:00 | 10680.00 | 10692.15 | 10693.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 14:15:00 | 10645.60 | 10676.07 | 10685.10 | Break + close below crossover candle low |

### Cycle 80 — BUY (started 2024-05-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-31 09:15:00 | 10805.60 | 10698.61 | 10693.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-03 09:15:00 | 11056.70 | 10810.18 | 10757.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-03 12:15:00 | 10877.60 | 10908.84 | 10823.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-03 13:00:00 | 10877.60 | 10908.84 | 10823.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 11178.95 | 11031.47 | 10914.18 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 9747.50 | 10774.68 | 10808.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-05 09:15:00 | 9350.50 | 9946.90 | 10311.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 13:15:00 | 9900.05 | 9826.24 | 10123.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 14:00:00 | 9900.05 | 9826.24 | 10123.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 10203.50 | 9901.69 | 10130.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 14:30:00 | 10300.15 | 9901.69 | 10130.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 15:15:00 | 10150.00 | 9951.36 | 10132.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:15:00 | 10379.90 | 9951.36 | 10132.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 10355.15 | 10032.11 | 10152.88 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2024-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 13:15:00 | 10422.10 | 10248.57 | 10230.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 14:15:00 | 10470.00 | 10292.86 | 10251.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 15:15:00 | 10469.95 | 10474.07 | 10393.93 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-10 09:15:00 | 10609.95 | 10474.07 | 10393.93 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 13:15:00 | 10485.40 | 10569.28 | 10483.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 14:00:00 | 10485.40 | 10569.28 | 10483.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 14:15:00 | 10489.35 | 10553.29 | 10484.42 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-10 15:15:00 | 10425.00 | 10527.63 | 10479.02 | SL hit (close<ema400) qty=1.00 sl=10479.02 alert=retest1 |

### Cycle 83 — SELL (started 2024-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-11 14:15:00 | 10360.05 | 10445.23 | 10453.04 | EMA200 below EMA400 |

### Cycle 84 — BUY (started 2024-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-12 09:15:00 | 10597.20 | 10463.59 | 10459.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-12 10:15:00 | 10717.75 | 10514.42 | 10482.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 12:15:00 | 10504.90 | 10539.52 | 10501.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-12 12:15:00 | 10504.90 | 10539.52 | 10501.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 12:15:00 | 10504.90 | 10539.52 | 10501.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 12:45:00 | 10505.00 | 10539.52 | 10501.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 13:15:00 | 10439.15 | 10519.45 | 10495.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-12 14:00:00 | 10439.15 | 10519.45 | 10495.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 10484.00 | 10512.36 | 10494.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 09:15:00 | 10600.00 | 10509.89 | 10495.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 14:15:00 | 10990.00 | 11159.47 | 11166.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2024-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 14:15:00 | 10990.00 | 11159.47 | 11166.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-20 15:15:00 | 10960.00 | 11119.57 | 11148.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 09:15:00 | 11150.35 | 11125.73 | 11148.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 09:15:00 | 11150.35 | 11125.73 | 11148.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 09:15:00 | 11150.35 | 11125.73 | 11148.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 11:45:00 | 10958.65 | 11092.56 | 11129.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 13:30:00 | 10950.95 | 11041.66 | 11098.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-24 09:15:00 | 11427.15 | 11128.30 | 11123.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-24 09:15:00 | 11427.15 | 11128.30 | 11123.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 10:15:00 | 11555.00 | 11213.64 | 11163.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-25 13:15:00 | 11572.65 | 11744.48 | 11552.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 13:15:00 | 11572.65 | 11744.48 | 11552.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 13:15:00 | 11572.65 | 11744.48 | 11552.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 14:00:00 | 11572.65 | 11744.48 | 11552.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 14:15:00 | 11397.10 | 11675.01 | 11538.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-25 15:00:00 | 11397.10 | 11675.01 | 11538.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 15:15:00 | 11340.00 | 11608.01 | 11520.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-26 09:15:00 | 11647.40 | 11608.01 | 11520.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-26 15:15:00 | 12812.14 | 12132.16 | 11848.62 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2024-07-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 13:15:00 | 13730.60 | 13780.58 | 13781.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-09 14:15:00 | 13455.60 | 13715.58 | 13752.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-11 09:15:00 | 13245.00 | 13139.60 | 13347.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-11 09:15:00 | 13245.00 | 13139.60 | 13347.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 13245.00 | 13139.60 | 13347.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:45:00 | 13332.55 | 13139.60 | 13347.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 12413.90 | 12226.34 | 12465.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:30:00 | 12488.10 | 12226.34 | 12465.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 12051.20 | 12147.91 | 12310.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 14:15:00 | 11799.55 | 12013.39 | 12188.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 10:30:00 | 11799.80 | 11935.50 | 12090.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 11:15:00 | 11795.25 | 11935.50 | 12090.67 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 13:15:00 | 11209.57 | 11667.06 | 11922.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 13:15:00 | 11209.81 | 11667.06 | 11922.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-19 13:15:00 | 11205.49 | 11667.06 | 11922.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-07-23 12:15:00 | 11413.00 | 11327.66 | 11479.73 | SL hit (close>ema200) qty=0.50 sl=11327.66 alert=retest2 |

### Cycle 88 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 11923.45 | 11526.47 | 11516.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-24 15:15:00 | 12200.00 | 11872.93 | 11710.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 11:15:00 | 11690.90 | 11890.45 | 11764.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 11:15:00 | 11690.90 | 11890.45 | 11764.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 11:15:00 | 11690.90 | 11890.45 | 11764.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 12:00:00 | 11690.90 | 11890.45 | 11764.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 11798.30 | 11872.02 | 11767.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 12:30:00 | 11619.05 | 11872.02 | 11767.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 11845.00 | 11866.61 | 11774.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-25 13:45:00 | 11772.40 | 11866.61 | 11774.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 15:15:00 | 11699.10 | 11830.46 | 11773.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:15:00 | 11582.95 | 11830.46 | 11773.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 11677.65 | 11799.90 | 11765.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 09:45:00 | 11725.15 | 11799.90 | 11765.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 11630.55 | 11766.03 | 11752.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:45:00 | 11649.45 | 11766.03 | 11752.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — SELL (started 2024-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-26 11:15:00 | 11508.50 | 11714.52 | 11730.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-26 13:15:00 | 11488.95 | 11643.63 | 11694.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-29 09:15:00 | 11707.40 | 11619.21 | 11667.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-29 09:15:00 | 11707.40 | 11619.21 | 11667.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 11707.40 | 11619.21 | 11667.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 10:15:00 | 11722.60 | 11619.21 | 11667.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 10:15:00 | 11752.55 | 11645.87 | 11674.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 12:00:00 | 11682.35 | 11653.17 | 11675.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-29 12:30:00 | 11598.60 | 11628.90 | 11662.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-29 14:15:00 | 11900.95 | 11700.46 | 11690.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2024-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 14:15:00 | 11900.95 | 11700.46 | 11690.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 15:15:00 | 11920.00 | 11744.37 | 11711.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 14:15:00 | 11869.65 | 11887.00 | 11810.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 14:15:00 | 11869.65 | 11887.00 | 11810.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 11869.65 | 11887.00 | 11810.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 15:00:00 | 11869.65 | 11887.00 | 11810.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 11863.15 | 11876.40 | 11818.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 09:30:00 | 11821.50 | 11876.40 | 11818.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 10:15:00 | 11793.65 | 11859.85 | 11816.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 10:45:00 | 11812.40 | 11859.85 | 11816.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 11:15:00 | 11845.90 | 11857.06 | 11819.24 | EMA400 retest candle locked (from upside) |

### Cycle 91 — SELL (started 2024-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 14:15:00 | 11600.35 | 11786.32 | 11794.75 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 10:15:00 | 11905.95 | 11801.69 | 11797.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-01 11:15:00 | 12204.00 | 11882.15 | 11834.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-02 11:15:00 | 12112.85 | 12166.46 | 12040.03 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-02 12:00:00 | 12112.85 | 12166.46 | 12040.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 12:15:00 | 12008.85 | 12134.94 | 12037.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 13:00:00 | 12008.85 | 12134.94 | 12037.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 13:15:00 | 12025.00 | 12112.95 | 12036.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 13:30:00 | 12008.00 | 12112.95 | 12036.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 14:15:00 | 12080.00 | 12106.36 | 12040.08 | EMA400 retest candle locked (from upside) |

### Cycle 93 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 11545.45 | 11932.67 | 11973.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 11:15:00 | 11489.00 | 11843.93 | 11929.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-08 14:15:00 | 10932.10 | 10927.74 | 11085.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-08 15:00:00 | 10932.10 | 10927.74 | 11085.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 15:15:00 | 11063.00 | 10954.80 | 11083.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:15:00 | 11128.95 | 10954.80 | 11083.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 11001.40 | 10964.12 | 11075.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 14:00:00 | 10938.45 | 11016.39 | 11072.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-09 15:00:00 | 10942.15 | 11001.54 | 11060.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 12:15:00 | 11215.00 | 11101.16 | 11089.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2024-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 12:15:00 | 11215.00 | 11101.16 | 11089.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 13:15:00 | 11439.60 | 11168.85 | 11121.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 14:15:00 | 11370.00 | 11457.04 | 11340.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 14:30:00 | 11375.00 | 11457.04 | 11340.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 15:15:00 | 11300.05 | 11425.64 | 11336.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 10:00:00 | 11250.10 | 11390.53 | 11328.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 11239.90 | 11360.41 | 11320.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 10:30:00 | 11310.00 | 11360.41 | 11320.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2024-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 13:15:00 | 11208.00 | 11290.09 | 11294.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 15:15:00 | 11145.00 | 11242.06 | 11270.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 09:15:00 | 11263.55 | 11246.36 | 11270.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 09:15:00 | 11263.55 | 11246.36 | 11270.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 09:15:00 | 11263.55 | 11246.36 | 11270.30 | EMA400 retest candle locked (from downside) |

### Cycle 96 — BUY (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 13:15:00 | 11679.20 | 11359.21 | 11316.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 14:15:00 | 11800.05 | 11447.38 | 11360.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-21 15:15:00 | 12345.90 | 12399.60 | 12255.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 09:15:00 | 12164.70 | 12399.60 | 12255.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 09:15:00 | 12151.05 | 12349.89 | 12246.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 09:30:00 | 12134.00 | 12349.89 | 12246.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 12147.60 | 12309.43 | 12237.21 | EMA400 retest candle locked (from upside) |

### Cycle 97 — SELL (started 2024-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-22 14:15:00 | 12080.00 | 12192.80 | 12199.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-22 15:15:00 | 12000.15 | 12154.27 | 12181.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-23 13:15:00 | 12200.00 | 12144.48 | 12163.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 13:15:00 | 12200.00 | 12144.48 | 12163.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 12200.00 | 12144.48 | 12163.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 13:30:00 | 12223.00 | 12144.48 | 12163.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 12202.35 | 12156.05 | 12167.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-23 15:00:00 | 12202.35 | 12156.05 | 12167.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 12210.00 | 12166.84 | 12171.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 12149.20 | 12166.84 | 12171.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 12061.70 | 12068.12 | 12105.19 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2024-08-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-27 15:15:00 | 12175.00 | 12115.99 | 12113.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 09:15:00 | 12258.40 | 12144.47 | 12126.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 09:15:00 | 12009.00 | 12156.64 | 12150.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 09:15:00 | 12009.00 | 12156.64 | 12150.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 09:15:00 | 12009.00 | 12156.64 | 12150.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 10:00:00 | 12009.00 | 12156.64 | 12150.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 99 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 12000.00 | 12125.31 | 12137.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 11:15:00 | 11935.40 | 12087.33 | 12118.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 13:15:00 | 11900.00 | 11828.10 | 11926.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-30 14:00:00 | 11900.00 | 11828.10 | 11926.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 14:15:00 | 12097.70 | 11882.02 | 11942.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 15:00:00 | 12097.70 | 11882.02 | 11942.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 15:15:00 | 12000.05 | 11905.63 | 11947.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-02 09:15:00 | 12052.65 | 11905.63 | 11947.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 09:15:00 | 11962.45 | 11916.99 | 11948.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 10:15:00 | 11851.25 | 11916.99 | 11948.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 11:15:00 | 11902.25 | 11923.25 | 11948.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 10:15:00 | 11901.10 | 11878.59 | 11906.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 12:15:00 | 11765.40 | 11729.68 | 11729.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2024-09-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 12:15:00 | 11765.40 | 11729.68 | 11729.56 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2024-09-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-06 13:15:00 | 11704.25 | 11724.59 | 11727.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 11675.15 | 11714.71 | 11722.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 15:15:00 | 11470.00 | 11469.53 | 11564.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 09:15:00 | 11512.45 | 11469.53 | 11564.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 11558.10 | 11487.24 | 11563.66 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2024-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 13:15:00 | 11823.50 | 11629.12 | 11611.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-10 14:15:00 | 11919.10 | 11687.12 | 11639.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-12 10:15:00 | 12015.05 | 12023.25 | 11893.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-12 10:30:00 | 12012.10 | 12023.25 | 11893.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 12924.95 | 13040.23 | 12930.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:00:00 | 12924.95 | 13040.23 | 12930.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 12924.90 | 13017.16 | 12929.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:45:00 | 12880.60 | 13017.16 | 12929.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 12876.00 | 12988.93 | 12924.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 15:00:00 | 12876.00 | 12988.93 | 12924.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 15:15:00 | 12950.00 | 12981.14 | 12927.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:15:00 | 12824.05 | 12981.14 | 12927.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 09:15:00 | 12751.85 | 12935.28 | 12911.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 10:00:00 | 12751.85 | 12935.28 | 12911.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 10:15:00 | 12709.65 | 12890.16 | 12892.93 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-09-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 12:15:00 | 12854.80 | 12800.87 | 12798.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 13:15:00 | 12966.65 | 12834.02 | 12813.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 14:15:00 | 12742.65 | 12815.75 | 12806.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-23 14:15:00 | 12742.65 | 12815.75 | 12806.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 14:15:00 | 12742.65 | 12815.75 | 12806.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 15:00:00 | 12742.65 | 12815.75 | 12806.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2024-09-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 15:15:00 | 12695.00 | 12791.60 | 12796.75 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2024-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 11:15:00 | 12845.00 | 12805.87 | 12802.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 12:15:00 | 12884.90 | 12821.67 | 12809.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 14:15:00 | 12800.00 | 12817.78 | 12810.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 14:15:00 | 12800.00 | 12817.78 | 12810.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 12800.00 | 12817.78 | 12810.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 15:15:00 | 12800.00 | 12817.78 | 12810.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 12800.00 | 12814.23 | 12809.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 09:15:00 | 13163.40 | 12814.23 | 12809.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-30 14:15:00 | 14479.74 | 13903.76 | 13650.63 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2024-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 10:15:00 | 13615.00 | 14071.78 | 14086.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-08 09:15:00 | 13469.35 | 13747.16 | 13902.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 12:15:00 | 13790.50 | 13720.58 | 13849.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 12:15:00 | 13790.50 | 13720.58 | 13849.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 13790.50 | 13720.58 | 13849.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:00:00 | 13790.50 | 13720.58 | 13849.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 13995.35 | 13775.53 | 13862.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:00:00 | 13995.35 | 13775.53 | 13862.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 14120.85 | 13844.59 | 13886.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 14120.85 | 13844.59 | 13886.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 14048.95 | 13885.47 | 13900.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 14271.15 | 13885.47 | 13900.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2024-10-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 09:15:00 | 14238.50 | 13956.07 | 13931.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 10:15:00 | 14621.30 | 14089.12 | 13994.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 15:15:00 | 15852.00 | 15906.05 | 15474.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-14 09:15:00 | 15943.80 | 15906.05 | 15474.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 15:15:00 | 16002.50 | 16067.77 | 15976.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 09:15:00 | 15873.00 | 16067.77 | 15976.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 15755.10 | 16005.23 | 15956.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 15755.10 | 16005.23 | 15956.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 15641.10 | 15932.41 | 15927.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:45:00 | 15643.95 | 15932.41 | 15927.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 11:15:00 | 15614.00 | 15868.73 | 15899.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 11:15:00 | 15560.05 | 15708.34 | 15797.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 12:15:00 | 13836.95 | 13835.32 | 14145.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 12:30:00 | 13835.65 | 13835.32 | 14145.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 13559.65 | 13484.80 | 13717.02 | EMA400 retest candle locked (from downside) |

### Cycle 110 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 14171.80 | 13661.95 | 13659.00 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 10:15:00 | 13192.00 | 13590.37 | 13634.34 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-10-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 11:15:00 | 13850.00 | 13652.96 | 13632.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-31 12:15:00 | 13891.10 | 13700.59 | 13656.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 13708.75 | 13794.74 | 13734.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 13708.75 | 13794.74 | 13734.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 13708.75 | 13794.74 | 13734.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 11:45:00 | 13953.50 | 13860.48 | 13774.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 09:15:00 | 14245.75 | 13959.86 | 13860.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-05 12:45:00 | 13978.15 | 13931.75 | 13877.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-08 11:15:00 | 13918.60 | 14263.92 | 14291.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 11:15:00 | 13918.60 | 14263.92 | 14291.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 09:15:00 | 13319.05 | 13949.82 | 14118.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 11819.90 | 11566.65 | 11860.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-19 09:15:00 | 11819.90 | 11566.65 | 11860.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 11819.90 | 11566.65 | 11860.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-21 10:30:00 | 11300.05 | 11540.60 | 11729.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 09:45:00 | 11291.05 | 11395.83 | 11556.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-22 10:45:00 | 11261.30 | 11370.15 | 11529.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 12005.05 | 11549.89 | 11548.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2024-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 09:15:00 | 12005.05 | 11549.89 | 11548.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 10:15:00 | 12045.70 | 11649.05 | 11593.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 14:15:00 | 11609.85 | 11823.40 | 11714.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 14:15:00 | 11609.85 | 11823.40 | 11714.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 14:15:00 | 11609.85 | 11823.40 | 11714.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-25 15:00:00 | 11609.85 | 11823.40 | 11714.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 15:15:00 | 11899.50 | 11838.62 | 11731.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 09:15:00 | 12645.70 | 11838.62 | 11731.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-29 11:15:00 | 11981.45 | 12130.25 | 12139.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2024-11-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-29 11:15:00 | 11981.45 | 12130.25 | 12139.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 15:15:00 | 11940.00 | 12043.81 | 12091.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 10:15:00 | 12090.10 | 12052.07 | 12086.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 10:15:00 | 12090.10 | 12052.07 | 12086.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 12090.10 | 12052.07 | 12086.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 10:45:00 | 12078.20 | 12052.07 | 12086.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 11:15:00 | 12091.65 | 12059.98 | 12087.16 | EMA400 retest candle locked (from downside) |

### Cycle 116 — BUY (started 2024-12-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 13:15:00 | 12240.40 | 12104.44 | 12103.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 15:15:00 | 12252.00 | 12153.72 | 12127.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 15:15:00 | 12180.00 | 12229.15 | 12190.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 15:15:00 | 12180.00 | 12229.15 | 12190.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 15:15:00 | 12180.00 | 12229.15 | 12190.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:45:00 | 12185.70 | 12227.51 | 12193.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 12192.50 | 12220.51 | 12193.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-04 13:45:00 | 12261.10 | 12201.12 | 12187.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-04 15:15:00 | 12079.00 | 12171.71 | 12176.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2024-12-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-04 15:15:00 | 12079.00 | 12171.71 | 12176.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-05 09:15:00 | 11948.70 | 12127.11 | 12155.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-06 12:15:00 | 11941.75 | 11915.63 | 11990.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-06 13:00:00 | 11941.75 | 11915.63 | 11990.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 12107.80 | 11959.60 | 11997.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 15:00:00 | 12107.80 | 11959.60 | 11997.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 15:15:00 | 12100.00 | 11987.68 | 12006.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:15:00 | 12189.00 | 11987.68 | 12006.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 09:15:00 | 12159.65 | 12022.08 | 12020.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 10:15:00 | 12363.25 | 12090.31 | 12051.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-13 09:15:00 | 12749.60 | 12915.21 | 12740.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 09:15:00 | 12749.60 | 12915.21 | 12740.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 12749.60 | 12915.21 | 12740.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:45:00 | 12753.50 | 12915.21 | 12740.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 12765.00 | 12885.17 | 12743.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 13:00:00 | 12847.40 | 12859.99 | 12755.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 11:15:00 | 12819.50 | 13040.78 | 13044.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2024-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-18 11:15:00 | 12819.50 | 13040.78 | 13044.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 13:15:00 | 12788.00 | 12953.49 | 13001.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-19 11:15:00 | 13028.45 | 12885.63 | 12939.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 11:15:00 | 13028.45 | 12885.63 | 12939.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 13028.45 | 12885.63 | 12939.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 12:00:00 | 13028.45 | 12885.63 | 12939.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 13006.10 | 12909.72 | 12945.76 | EMA400 retest candle locked (from downside) |

### Cycle 120 — BUY (started 2024-12-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 14:15:00 | 13109.80 | 12980.89 | 12973.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-20 09:15:00 | 13242.30 | 13052.23 | 13008.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 15:15:00 | 13200.00 | 13205.23 | 13118.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-23 09:15:00 | 13299.10 | 13205.23 | 13118.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 13342.10 | 13232.60 | 13138.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 11:30:00 | 13500.00 | 13311.54 | 13192.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 12:30:00 | 13515.00 | 13358.47 | 13224.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-30 09:15:00 | 14850.00 | 14621.12 | 14336.83 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 14740.05 | 15003.34 | 15036.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 10:15:00 | 14609.55 | 14783.09 | 14898.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 12:15:00 | 14833.05 | 14776.69 | 14874.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 12:15:00 | 14833.05 | 14776.69 | 14874.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 14833.05 | 14776.69 | 14874.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 12:30:00 | 14842.50 | 14776.69 | 14874.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 14832.95 | 14787.94 | 14870.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:00:00 | 14832.95 | 14787.94 | 14870.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 14686.85 | 14767.72 | 14853.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-07 15:15:00 | 14580.00 | 14767.72 | 14853.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 12:15:00 | 13851.00 | 14159.36 | 14398.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 09:15:00 | 13122.00 | 13512.19 | 13824.46 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 122 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 12277.50 | 10620.98 | 10567.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 13518.60 | 12178.32 | 11512.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 12:15:00 | 12101.30 | 12551.70 | 12195.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 12:15:00 | 12101.30 | 12551.70 | 12195.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 12101.30 | 12551.70 | 12195.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 12101.30 | 12551.70 | 12195.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 12240.00 | 12489.36 | 12199.49 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 11300.00 | 11972.80 | 12031.90 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2025-02-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 09:15:00 | 12255.05 | 11919.09 | 11889.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-06 10:15:00 | 12539.70 | 12129.07 | 12013.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 13:15:00 | 12150.00 | 12198.97 | 12081.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 14:00:00 | 12150.00 | 12198.97 | 12081.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 12257.40 | 12283.76 | 12200.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 14:30:00 | 12259.60 | 12283.76 | 12200.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 12230.95 | 12277.72 | 12212.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 10:00:00 | 12230.95 | 12277.72 | 12212.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 11:15:00 | 12168.50 | 12246.65 | 12209.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 12:00:00 | 12168.50 | 12246.65 | 12209.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 12:15:00 | 12105.55 | 12218.43 | 12199.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-10 13:00:00 | 12105.55 | 12218.43 | 12199.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2025-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 13:15:00 | 12023.30 | 12179.40 | 12183.60 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2025-02-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-10 14:15:00 | 12234.65 | 12190.45 | 12188.24 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2025-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-11 09:15:00 | 11838.20 | 12121.53 | 12157.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 13:15:00 | 11775.55 | 11951.75 | 12056.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 14:15:00 | 11703.70 | 11648.90 | 11801.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 15:00:00 | 11703.70 | 11648.90 | 11801.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-13 09:15:00 | 11872.70 | 11707.44 | 11802.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 11415.45 | 11791.45 | 11819.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 10:15:00 | 10844.68 | 11226.74 | 11458.03 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 10649.55 | 10569.76 | 10821.03 | SL hit (close>ema200) qty=0.50 sl=10569.76 alert=retest2 |

### Cycle 128 — BUY (started 2025-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 13:15:00 | 11098.90 | 10777.62 | 10765.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 14:15:00 | 11339.05 | 10889.90 | 10817.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-24 14:15:00 | 11600.00 | 11641.93 | 11411.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-24 14:45:00 | 11608.85 | 11641.93 | 11411.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 15:15:00 | 11691.70 | 11699.66 | 11567.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 09:15:00 | 11746.20 | 11699.66 | 11567.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 11507.20 | 11661.17 | 11561.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 09:30:00 | 11470.70 | 11661.17 | 11561.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 10:15:00 | 11484.75 | 11625.89 | 11554.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 11:00:00 | 11484.75 | 11625.89 | 11554.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 15:15:00 | 11514.00 | 11556.87 | 11541.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-28 09:15:00 | 11407.05 | 11556.87 | 11541.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2025-02-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-28 09:15:00 | 11250.90 | 11495.67 | 11515.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-03 09:15:00 | 11203.60 | 11300.52 | 11387.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 11462.55 | 11313.65 | 11370.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 12:15:00 | 11462.55 | 11313.65 | 11370.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 12:15:00 | 11462.55 | 11313.65 | 11370.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:00:00 | 11462.55 | 11313.65 | 11370.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 13:15:00 | 11678.25 | 11386.57 | 11398.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 13:45:00 | 11617.10 | 11386.57 | 11398.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2025-03-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 14:15:00 | 11750.00 | 11459.26 | 11430.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 09:15:00 | 12063.70 | 11630.98 | 11516.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 10:15:00 | 13430.00 | 13432.86 | 13062.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 10:45:00 | 13398.20 | 13432.86 | 13062.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 13090.00 | 13294.36 | 13109.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 15:00:00 | 13090.00 | 13294.36 | 13109.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 12900.00 | 13215.49 | 13090.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 12655.05 | 13215.49 | 13090.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 12882.40 | 13148.87 | 13071.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 12724.25 | 13148.87 | 13071.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 12884.60 | 13096.02 | 13054.85 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2025-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 12:15:00 | 12811.00 | 13004.45 | 13018.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 12716.35 | 12916.52 | 12973.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 13:15:00 | 12180.00 | 12165.63 | 12400.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 14:00:00 | 12180.00 | 12165.63 | 12400.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 12349.85 | 12207.09 | 12345.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-13 10:30:00 | 12350.00 | 12207.09 | 12345.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 11:15:00 | 12106.55 | 12186.98 | 12323.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 13:15:00 | 12063.75 | 12168.49 | 12303.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 14:00:00 | 12028.05 | 12140.41 | 12278.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 09:15:00 | 12666.95 | 12207.73 | 12271.02 | SL hit (close>static) qty=1.00 sl=12429.95 alert=retest2 |

### Cycle 132 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 11:15:00 | 12738.00 | 12389.19 | 12347.19 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-18 10:15:00 | 12047.20 | 12368.31 | 12379.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-19 09:15:00 | 11811.05 | 12073.05 | 12210.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-20 09:15:00 | 11999.80 | 11904.82 | 12034.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 09:15:00 | 11999.80 | 11904.82 | 12034.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 11999.80 | 11904.82 | 12034.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 09:30:00 | 11987.85 | 11904.82 | 12034.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 11922.80 | 11908.42 | 12024.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 12:00:00 | 11860.00 | 11898.73 | 12009.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 13:00:00 | 11839.15 | 11886.82 | 11994.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-20 13:45:00 | 11845.85 | 11880.55 | 11981.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-21 10:15:00 | 12086.80 | 11945.23 | 11980.32 | SL hit (close>static) qty=1.00 sl=12046.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-03-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-21 11:15:00 | 12310.00 | 12018.18 | 12010.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-24 09:15:00 | 12525.95 | 12161.65 | 12084.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 14:15:00 | 12213.00 | 12264.22 | 12174.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 15:00:00 | 12213.00 | 12264.22 | 12174.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 15:15:00 | 12315.00 | 12274.38 | 12187.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 09:30:00 | 12379.10 | 12261.50 | 12189.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 15:15:00 | 12113.00 | 12213.71 | 12195.76 | SL hit (close<static) qty=1.00 sl=12150.20 alert=retest2 |

### Cycle 135 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 12166.75 | 12190.29 | 12192.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 09:15:00 | 12000.00 | 12152.99 | 12174.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 12:15:00 | 12289.90 | 12137.15 | 12158.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 12:15:00 | 12289.90 | 12137.15 | 12158.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 12289.90 | 12137.15 | 12158.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:00:00 | 12289.90 | 12137.15 | 12158.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 13:15:00 | 12337.40 | 12177.20 | 12174.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 14:15:00 | 13115.20 | 12364.80 | 12259.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 14:15:00 | 12658.75 | 12672.47 | 12503.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-28 15:00:00 | 12658.75 | 12672.47 | 12503.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 12385.90 | 12603.56 | 12501.03 | EMA400 retest candle locked (from upside) |

### Cycle 137 — SELL (started 2025-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 14:15:00 | 12267.00 | 12419.13 | 12438.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 11:15:00 | 12225.05 | 12327.17 | 12358.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 11700.05 | 11640.89 | 11878.49 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-08 10:45:00 | 11531.75 | 11636.72 | 11855.00 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-08 11:30:00 | 11576.80 | 11626.37 | 11830.45 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-08 13:30:00 | 11577.40 | 11624.71 | 11794.64 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-09 09:15:00 | 11473.10 | 11643.07 | 11773.69 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 11712.75 | 11488.81 | 11594.21 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 11712.75 | 11488.81 | 11594.21 | SL hit (close>ema400) qty=1.00 sl=11594.21 alert=retest1 |

### Cycle 138 — BUY (started 2025-04-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 12:15:00 | 12018.35 | 11669.77 | 11656.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 09:15:00 | 12508.00 | 11957.47 | 11805.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 13:15:00 | 13860.00 | 13910.61 | 13758.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-24 14:00:00 | 13860.00 | 13910.61 | 13758.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 14:15:00 | 13943.00 | 13917.09 | 13775.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 14:45:00 | 13798.00 | 13917.09 | 13775.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 13581.00 | 13839.14 | 13764.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-25 10:00:00 | 13581.00 | 13839.14 | 13764.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 10:15:00 | 13457.00 | 13762.71 | 13736.10 | EMA400 retest candle locked (from upside) |

### Cycle 139 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 13440.00 | 13698.17 | 13709.19 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 10:15:00 | 13872.00 | 13695.01 | 13689.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 14:15:00 | 13978.00 | 13811.46 | 13752.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 14:15:00 | 14506.00 | 14510.99 | 14282.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-30 15:00:00 | 14506.00 | 14510.99 | 14282.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 14486.00 | 14752.54 | 14644.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 15:00:00 | 14486.00 | 14752.54 | 14644.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 14580.00 | 14718.03 | 14638.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 09:30:00 | 14725.00 | 14672.03 | 14624.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 13:15:00 | 14500.00 | 14603.27 | 14603.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2025-05-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 13:15:00 | 14500.00 | 14603.27 | 14603.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 14423.00 | 14567.22 | 14587.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 12:15:00 | 14497.00 | 14438.43 | 14502.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 12:45:00 | 14486.00 | 14438.43 | 14502.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 14456.00 | 14441.95 | 14498.17 | EMA400 retest candle locked (from downside) |

### Cycle 142 — BUY (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 15:15:00 | 14750.00 | 14539.08 | 14534.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 09:15:00 | 14916.00 | 14614.47 | 14569.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 14:15:00 | 14700.00 | 14796.72 | 14698.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 14:15:00 | 14700.00 | 14796.72 | 14698.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 14:15:00 | 14700.00 | 14796.72 | 14698.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 15:00:00 | 14700.00 | 14796.72 | 14698.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 14660.00 | 14769.38 | 14694.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 14241.00 | 14769.38 | 14694.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 14386.00 | 14692.70 | 14666.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:30:00 | 14350.00 | 14692.70 | 14666.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 12:15:00 | 14804.00 | 14706.79 | 14678.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 13:30:00 | 14823.00 | 14772.03 | 14710.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-14 09:15:00 | 16305.30 | 15983.62 | 15656.18 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2025-05-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 14:15:00 | 15580.00 | 15941.39 | 15958.93 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 11:15:00 | 15990.00 | 15897.64 | 15890.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 16185.00 | 15999.78 | 15946.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 11:15:00 | 15980.00 | 16000.34 | 15956.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 11:15:00 | 15980.00 | 16000.34 | 15956.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 15980.00 | 16000.34 | 15956.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:00:00 | 15980.00 | 16000.34 | 15956.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 15888.00 | 15977.87 | 15950.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 13:00:00 | 15888.00 | 15977.87 | 15950.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 15814.00 | 15945.10 | 15937.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 15814.00 | 15945.10 | 15937.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 15807.00 | 15917.48 | 15925.99 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 15:15:00 | 15996.00 | 15933.18 | 15932.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 09:15:00 | 16196.00 | 15985.74 | 15956.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 13:15:00 | 17124.00 | 17235.70 | 16943.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-23 14:15:00 | 17042.00 | 17235.70 | 16943.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 16727.00 | 17099.23 | 16951.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:00:00 | 16727.00 | 17099.23 | 16951.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 10:15:00 | 16637.00 | 17006.78 | 16922.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 11:00:00 | 16637.00 | 17006.78 | 16922.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 13:15:00 | 16588.00 | 16826.45 | 16853.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-26 14:15:00 | 16294.00 | 16719.96 | 16802.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-27 09:15:00 | 16831.00 | 16674.97 | 16763.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 16831.00 | 16674.97 | 16763.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 16831.00 | 16674.97 | 16763.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:45:00 | 17077.00 | 16674.97 | 16763.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 16838.00 | 16707.58 | 16770.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:15:00 | 16949.00 | 16707.58 | 16770.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 11:15:00 | 16973.00 | 16760.66 | 16789.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-27 11:30:00 | 17055.00 | 16760.66 | 16789.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — BUY (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 13:15:00 | 17097.00 | 16850.22 | 16826.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 14:15:00 | 17311.00 | 16942.38 | 16870.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-05 13:15:00 | 19780.00 | 19847.44 | 19656.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 13:15:00 | 19780.00 | 19847.44 | 19656.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 19780.00 | 19847.44 | 19656.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 13:30:00 | 19664.00 | 19847.44 | 19656.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 19421.00 | 19762.15 | 19634.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 15:00:00 | 19421.00 | 19762.15 | 19634.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 15:15:00 | 19451.00 | 19699.92 | 19618.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 09:15:00 | 19138.00 | 19699.92 | 19618.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 09:15:00 | 18551.00 | 19470.14 | 19521.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 10:15:00 | 18430.00 | 19262.11 | 19421.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 18011.00 | 17935.53 | 18187.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 18011.00 | 17935.53 | 18187.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 18011.00 | 17935.53 | 18187.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:30:00 | 18159.00 | 17935.53 | 18187.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 17446.00 | 17208.99 | 17457.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:45:00 | 17440.00 | 17208.99 | 17457.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 17534.00 | 17273.99 | 17464.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 12:00:00 | 17534.00 | 17273.99 | 17464.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 12:15:00 | 17545.00 | 17328.19 | 17471.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 13:15:00 | 17590.00 | 17328.19 | 17471.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — BUY (started 2025-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 09:15:00 | 17732.00 | 17572.75 | 17557.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 09:15:00 | 18073.00 | 17694.50 | 17628.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 18471.00 | 18593.70 | 18379.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 13:00:00 | 18471.00 | 18593.70 | 18379.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 18465.00 | 18567.96 | 18387.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 18408.00 | 18567.96 | 18387.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 18390.00 | 18532.37 | 18387.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 15:00:00 | 18390.00 | 18532.37 | 18387.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 18388.00 | 18503.50 | 18387.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 18201.00 | 18503.50 | 18387.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 18702.00 | 18543.20 | 18416.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 10:45:00 | 18754.00 | 18590.36 | 18449.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 11:30:00 | 18812.00 | 18519.89 | 18430.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 15:15:00 | 18890.00 | 18531.70 | 18455.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 09:30:00 | 18819.00 | 18664.49 | 18532.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 12:15:00 | 18662.00 | 18671.15 | 18569.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 13:30:00 | 18813.00 | 18697.92 | 18590.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-30 09:15:00 | 20629.40 | 20052.72 | 19794.99 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 151 — SELL (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 12:15:00 | 19480.00 | 19822.36 | 19856.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 13:15:00 | 19100.00 | 19677.88 | 19787.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 09:15:00 | 20080.00 | 19659.24 | 19742.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 09:15:00 | 20080.00 | 19659.24 | 19742.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 20080.00 | 19659.24 | 19742.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 10:00:00 | 20080.00 | 19659.24 | 19742.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 19995.00 | 19726.39 | 19765.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 12:00:00 | 19890.00 | 19759.11 | 19776.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 14:15:00 | 19975.00 | 19813.75 | 19797.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 152 — BUY (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 14:15:00 | 19975.00 | 19813.75 | 19797.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 15:15:00 | 19980.00 | 19847.00 | 19814.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 09:15:00 | 19775.00 | 19832.60 | 19810.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 19775.00 | 19832.60 | 19810.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 19775.00 | 19832.60 | 19810.66 | EMA400 retest candle locked (from upside) |

### Cycle 153 — SELL (started 2025-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 12:15:00 | 19700.00 | 19779.41 | 19789.90 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 09:15:00 | 19970.00 | 19805.50 | 19795.60 | EMA200 above EMA400 |

### Cycle 155 — SELL (started 2025-07-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-07 12:15:00 | 19600.00 | 19801.33 | 19817.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 14:15:00 | 19560.00 | 19720.85 | 19775.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 19490.00 | 19333.32 | 19487.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 19490.00 | 19333.32 | 19487.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 19490.00 | 19333.32 | 19487.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:30:00 | 19480.00 | 19333.32 | 19487.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 19475.00 | 19361.66 | 19486.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:45:00 | 19520.00 | 19361.66 | 19486.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 19620.00 | 19413.32 | 19498.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:00:00 | 19620.00 | 19413.32 | 19498.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 12:15:00 | 19655.00 | 19461.66 | 19512.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 12:45:00 | 19675.00 | 19461.66 | 19512.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 156 — BUY (started 2025-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 14:15:00 | 19855.00 | 19575.26 | 19557.45 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 15:15:00 | 19460.00 | 19555.85 | 19568.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 19050.00 | 19454.68 | 19521.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-16 09:15:00 | 18490.00 | 18483.68 | 18646.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 12:15:00 | 18545.00 | 18499.68 | 18613.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 18545.00 | 18499.68 | 18613.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:45:00 | 18595.00 | 18499.68 | 18613.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 18575.00 | 18515.00 | 18592.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 09:15:00 | 18710.00 | 18515.00 | 18592.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 19040.00 | 18620.00 | 18632.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:00:00 | 19040.00 | 18620.00 | 18632.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 158 — BUY (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 10:15:00 | 19170.00 | 18730.00 | 18681.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 11:15:00 | 19270.00 | 18838.00 | 18735.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 12:15:00 | 19100.00 | 19198.87 | 19033.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 13:00:00 | 19100.00 | 19198.87 | 19033.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 19050.00 | 19169.10 | 19035.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:00:00 | 19050.00 | 19169.10 | 19035.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 19070.00 | 19149.28 | 19038.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 09:30:00 | 19135.00 | 19148.54 | 19056.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 10:15:00 | 19290.00 | 19705.30 | 19743.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 19290.00 | 19705.30 | 19743.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 12:15:00 | 19130.00 | 19542.99 | 19659.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 19245.00 | 19112.52 | 19301.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 09:15:00 | 19245.00 | 19112.52 | 19301.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 19245.00 | 19112.52 | 19301.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 09:45:00 | 19405.00 | 19112.52 | 19301.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 19385.00 | 19167.02 | 19309.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 11:00:00 | 19385.00 | 19167.02 | 19309.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 11:15:00 | 19330.00 | 19199.62 | 19310.92 | EMA400 retest candle locked (from downside) |

### Cycle 160 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 19895.00 | 19448.56 | 19406.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 20585.00 | 19732.08 | 19546.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 09:15:00 | 19825.00 | 20412.62 | 20087.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 19825.00 | 20412.62 | 20087.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 19825.00 | 20412.62 | 20087.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:30:00 | 19660.00 | 20412.62 | 20087.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 19850.00 | 20300.09 | 20065.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:15:00 | 19840.00 | 20300.09 | 20065.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 19990.00 | 20238.07 | 20059.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:00:00 | 20030.00 | 20196.46 | 20056.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:30:00 | 20010.00 | 20167.17 | 20055.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 14:30:00 | 20185.00 | 20147.73 | 20057.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 12:15:00 | 20705.00 | 20883.55 | 20885.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 161 — SELL (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 12:15:00 | 20705.00 | 20883.55 | 20885.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 14:15:00 | 20420.00 | 20742.27 | 20817.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 09:15:00 | 20875.00 | 20718.85 | 20790.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 20875.00 | 20718.85 | 20790.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 20875.00 | 20718.85 | 20790.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:30:00 | 20960.00 | 20718.85 | 20790.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 20940.00 | 20763.08 | 20804.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 11:00:00 | 20940.00 | 20763.08 | 20804.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 11:15:00 | 20715.00 | 20753.47 | 20796.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 13:15:00 | 20575.00 | 20747.77 | 20789.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 20305.00 | 20724.62 | 20768.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 14:30:00 | 20465.00 | 20606.28 | 20682.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 12:00:00 | 20585.00 | 20601.94 | 20653.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 12:15:00 | 20770.00 | 20635.55 | 20663.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 13:00:00 | 20770.00 | 20635.55 | 20663.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 13:15:00 | 20715.00 | 20651.44 | 20668.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-11 14:15:00 | 20880.00 | 20697.15 | 20687.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 162 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 20880.00 | 20697.15 | 20687.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 21585.00 | 20895.18 | 20780.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 13:15:00 | 20855.00 | 21040.87 | 20906.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 13:15:00 | 20855.00 | 21040.87 | 20906.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 20855.00 | 21040.87 | 20906.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:00:00 | 20855.00 | 21040.87 | 20906.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 20740.00 | 20980.69 | 20891.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:45:00 | 20750.00 | 20980.69 | 20891.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 20660.00 | 20916.55 | 20870.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 09:15:00 | 20980.00 | 20916.55 | 20870.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 12:15:00 | 20485.00 | 20889.67 | 20920.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2025-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 12:15:00 | 20485.00 | 20889.67 | 20920.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 14:15:00 | 20400.00 | 20728.59 | 20838.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-20 09:15:00 | 20300.00 | 19919.95 | 20114.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-20 09:15:00 | 20300.00 | 19919.95 | 20114.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 09:15:00 | 20300.00 | 19919.95 | 20114.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-20 10:00:00 | 20300.00 | 19919.95 | 20114.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 10:15:00 | 20170.00 | 19969.96 | 20119.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 12:30:00 | 20065.00 | 20033.17 | 20124.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 13:15:00 | 20095.00 | 20033.17 | 20124.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-20 13:45:00 | 20045.00 | 20019.54 | 20110.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-21 09:45:00 | 20045.00 | 20029.00 | 20091.04 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 10:15:00 | 20245.00 | 20072.20 | 20105.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 11:00:00 | 20245.00 | 20072.20 | 20105.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 20100.00 | 20080.61 | 20103.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:45:00 | 20100.00 | 20080.61 | 20103.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 20065.00 | 20081.63 | 20098.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:15:00 | 20210.00 | 20081.63 | 20098.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 20090.00 | 20083.31 | 20097.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 10:15:00 | 19770.00 | 20083.31 | 20097.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 15:15:00 | 19505.00 | 19474.73 | 19584.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:45:00 | 19840.00 | 19499.63 | 19577.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 13:15:00 | 19950.00 | 19668.22 | 19639.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — BUY (started 2025-08-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-28 13:15:00 | 19950.00 | 19668.22 | 19639.98 | EMA200 above EMA400 |

### Cycle 165 — SELL (started 2025-08-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 15:15:00 | 19500.00 | 19622.06 | 19623.12 | EMA200 below EMA400 |

### Cycle 166 — BUY (started 2025-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 09:15:00 | 19815.00 | 19660.65 | 19640.57 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 12:15:00 | 19415.00 | 19630.85 | 19634.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 14:15:00 | 19085.00 | 19481.54 | 19563.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 09:15:00 | 19051.00 | 18980.48 | 19106.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 09:15:00 | 19051.00 | 18980.48 | 19106.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 19051.00 | 18980.48 | 19106.86 | EMA400 retest candle locked (from downside) |

### Cycle 168 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 19322.00 | 19177.53 | 19163.59 | EMA200 above EMA400 |

### Cycle 169 — SELL (started 2025-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 14:15:00 | 18839.00 | 19105.67 | 19137.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 14:15:00 | 18814.00 | 18955.15 | 19038.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 12:15:00 | 18901.00 | 18886.16 | 18967.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-08 12:45:00 | 18952.00 | 18886.16 | 18967.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 18849.00 | 18865.85 | 18931.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:30:00 | 18709.00 | 18816.75 | 18896.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 12:00:00 | 18729.00 | 18816.75 | 18896.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 13:15:00 | 18695.00 | 18802.00 | 18882.29 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 19394.00 | 18944.85 | 18922.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 170 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 19394.00 | 18944.85 | 18922.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 11:15:00 | 19449.00 | 19115.30 | 19008.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 13:15:00 | 19648.00 | 19700.25 | 19455.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 14:00:00 | 19648.00 | 19700.25 | 19455.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 19455.00 | 19651.20 | 19455.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 19455.00 | 19651.20 | 19455.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 19500.00 | 19620.96 | 19459.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 19722.00 | 19620.96 | 19459.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-18 10:15:00 | 19641.00 | 19948.52 | 19971.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 171 — SELL (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 10:15:00 | 19641.00 | 19948.52 | 19971.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 19449.00 | 19792.69 | 19892.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 09:15:00 | 19356.00 | 19254.94 | 19462.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-22 09:45:00 | 19420.00 | 19254.94 | 19462.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 19434.00 | 19290.75 | 19459.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:45:00 | 19463.00 | 19290.75 | 19459.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 11:15:00 | 19566.00 | 19345.80 | 19469.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 11:30:00 | 19541.00 | 19345.80 | 19469.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 12:15:00 | 19479.00 | 19372.44 | 19470.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 12:30:00 | 19628.00 | 19372.44 | 19470.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 19190.00 | 19335.95 | 19444.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:30:00 | 19550.00 | 19335.95 | 19444.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 19344.00 | 19276.33 | 19384.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-23 09:30:00 | 19400.00 | 19276.33 | 19384.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 18948.00 | 19210.66 | 19344.78 | EMA400 retest candle locked (from downside) |

### Cycle 172 — BUY (started 2025-09-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 15:15:00 | 19247.00 | 19134.42 | 19128.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-26 09:15:00 | 19530.00 | 19213.53 | 19165.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-26 14:15:00 | 19069.00 | 19323.99 | 19255.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-26 14:15:00 | 19069.00 | 19323.99 | 19255.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 14:15:00 | 19069.00 | 19323.99 | 19255.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 15:00:00 | 19069.00 | 19323.99 | 19255.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 15:15:00 | 19210.00 | 19301.20 | 19251.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-29 09:15:00 | 19637.00 | 19301.20 | 19251.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 12:15:00 | 18978.00 | 19233.57 | 19239.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — SELL (started 2025-09-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 12:15:00 | 18978.00 | 19233.57 | 19239.13 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 13:15:00 | 19333.00 | 19253.45 | 19247.66 | EMA200 above EMA400 |

### Cycle 175 — SELL (started 2025-09-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-30 09:15:00 | 18682.00 | 19134.33 | 19194.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 12:15:00 | 18144.00 | 18820.33 | 19026.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 15:15:00 | 18225.00 | 18148.41 | 18425.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-03 09:15:00 | 18429.00 | 18148.41 | 18425.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 18286.00 | 18175.93 | 18412.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 12:00:00 | 18113.00 | 18172.64 | 18370.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 10:15:00 | 18150.00 | 18236.87 | 18282.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-07 11:15:00 | 18159.00 | 18263.30 | 18290.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 09:15:00 | 17207.35 | 17468.55 | 17573.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 09:15:00 | 17242.50 | 17468.55 | 17573.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-15 09:15:00 | 17251.05 | 17468.55 | 17573.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-15 12:15:00 | 17460.00 | 17428.92 | 17525.59 | SL hit (close>ema200) qty=0.50 sl=17428.92 alert=retest2 |

### Cycle 176 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 17731.00 | 17523.95 | 17523.93 | EMA200 above EMA400 |

### Cycle 177 — SELL (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 14:15:00 | 17450.00 | 17541.25 | 17547.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 15:15:00 | 17410.00 | 17515.00 | 17535.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 10:15:00 | 17570.00 | 17519.60 | 17533.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 10:15:00 | 17570.00 | 17519.60 | 17533.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 17570.00 | 17519.60 | 17533.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:45:00 | 17571.00 | 17519.60 | 17533.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 17402.00 | 17496.08 | 17521.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 12:00:00 | 17317.00 | 17456.62 | 17485.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 10:15:00 | 17175.00 | 17003.51 | 16984.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 178 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 17175.00 | 17003.51 | 16984.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 12:15:00 | 17321.00 | 17102.93 | 17035.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 10:15:00 | 17872.00 | 17875.84 | 17617.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 11:00:00 | 17872.00 | 17875.84 | 17617.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 17762.00 | 17809.00 | 17698.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:15:00 | 17731.00 | 17809.00 | 17698.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 17365.00 | 17720.20 | 17668.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-03 12:00:00 | 17365.00 | 17720.20 | 17668.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 17478.00 | 17671.76 | 17651.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 14:00:00 | 17600.00 | 17657.41 | 17646.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-04 09:15:00 | 19360.00 | 18234.46 | 17917.85 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 179 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 21352.00 | 21488.51 | 21503.59 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 21674.00 | 21520.93 | 21509.08 | EMA200 above EMA400 |

### Cycle 181 — SELL (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-17 15:15:00 | 21466.00 | 21509.42 | 21509.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 09:15:00 | 21149.00 | 21437.34 | 21476.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-18 12:15:00 | 21420.00 | 21395.92 | 21444.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 12:15:00 | 21420.00 | 21395.92 | 21444.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 12:15:00 | 21420.00 | 21395.92 | 21444.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 13:00:00 | 21420.00 | 21395.92 | 21444.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 21567.00 | 21430.13 | 21455.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 14:00:00 | 21567.00 | 21430.13 | 21455.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 21393.00 | 21422.71 | 21449.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 14:30:00 | 21493.00 | 21422.71 | 21449.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 21378.00 | 21413.76 | 21443.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 21301.00 | 21413.76 | 21443.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 11:00:00 | 21330.00 | 21400.57 | 21432.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 11:15:00 | 21575.00 | 21435.46 | 21445.40 | SL hit (close>static) qty=1.00 sl=21444.00 alert=retest2 |

### Cycle 182 — BUY (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 12:15:00 | 21543.00 | 21456.96 | 21454.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 09:15:00 | 22172.00 | 21657.23 | 21553.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 21956.00 | 22142.48 | 21914.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-21 10:00:00 | 21956.00 | 22142.48 | 21914.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 21813.00 | 22076.58 | 21905.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:00:00 | 21813.00 | 22076.58 | 21905.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 21768.00 | 22014.86 | 21892.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 12:00:00 | 21768.00 | 22014.86 | 21892.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 183 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 21565.00 | 21825.89 | 21828.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 11:15:00 | 21424.00 | 21644.02 | 21733.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 21665.00 | 21590.19 | 21680.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 15:00:00 | 21665.00 | 21590.19 | 21680.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 21529.00 | 21577.96 | 21667.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 09:15:00 | 21462.00 | 21577.96 | 21667.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-25 14:15:00 | 21778.00 | 21632.02 | 21653.41 | SL hit (close>static) qty=1.00 sl=21724.00 alert=retest2 |

### Cycle 184 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 21955.00 | 21720.29 | 21691.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 11:15:00 | 22169.00 | 21843.75 | 21754.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 21982.00 | 22005.85 | 21878.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 10:00:00 | 21982.00 | 22005.85 | 21878.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 21918.00 | 21988.28 | 21882.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 09:45:00 | 22020.00 | 21923.00 | 21889.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:15:00 | 22040.00 | 21978.47 | 21927.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 14:30:00 | 22005.00 | 21990.54 | 21941.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 12:15:00 | 21840.00 | 22244.96 | 22246.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 185 — SELL (started 2025-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 12:15:00 | 21840.00 | 22244.96 | 22246.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 13:15:00 | 21695.00 | 22134.97 | 22195.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 14:15:00 | 19125.00 | 19052.67 | 19716.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-08 15:00:00 | 19125.00 | 19052.67 | 19716.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 19645.00 | 19258.64 | 19607.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 19645.00 | 19258.64 | 19607.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 19425.00 | 19291.91 | 19590.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 13:15:00 | 19325.00 | 19291.91 | 19590.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:30:00 | 19400.00 | 19367.62 | 19575.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 19405.00 | 19392.10 | 19567.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 11:15:00 | 19555.00 | 19300.48 | 19278.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 186 — BUY (started 2025-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 11:15:00 | 19555.00 | 19300.48 | 19278.22 | EMA200 above EMA400 |

### Cycle 187 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 19245.00 | 19353.89 | 19367.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 15:15:00 | 19225.00 | 19321.09 | 19350.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 19410.00 | 19338.87 | 19355.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 19410.00 | 19338.87 | 19355.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 19410.00 | 19338.87 | 19355.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:30:00 | 19530.00 | 19338.87 | 19355.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 19395.00 | 19350.10 | 19359.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 11:15:00 | 19290.00 | 19350.10 | 19359.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 12:15:00 | 18325.50 | 18748.57 | 19016.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 13:15:00 | 18325.00 | 18272.75 | 18563.04 | SL hit (close>ema200) qty=0.50 sl=18272.75 alert=retest2 |

### Cycle 188 — BUY (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 13:15:00 | 18630.00 | 18611.05 | 18609.31 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 14:15:00 | 18415.00 | 18571.84 | 18591.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 18335.00 | 18495.38 | 18551.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 13:15:00 | 18350.00 | 18320.93 | 18398.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 14:00:00 | 18350.00 | 18320.93 | 18398.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 18380.00 | 18332.74 | 18397.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 18380.00 | 18332.74 | 18397.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 18295.00 | 18325.19 | 18387.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:45:00 | 18215.00 | 18304.15 | 18372.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 14:30:00 | 18210.00 | 18133.93 | 18245.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 09:45:00 | 18145.00 | 18160.52 | 18238.92 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:30:00 | 18220.00 | 18207.33 | 18248.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 18340.00 | 18233.86 | 18256.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-31 13:15:00 | 18420.00 | 18271.09 | 18271.30 | SL hit (close>static) qty=1.00 sl=18390.00 alert=retest2 |

### Cycle 190 — BUY (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 09:15:00 | 18326.00 | 18280.96 | 18275.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 14:15:00 | 18501.00 | 18343.67 | 18308.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 10:15:00 | 18842.00 | 18903.30 | 18773.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-06 11:00:00 | 18842.00 | 18903.30 | 18773.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 11:15:00 | 18888.00 | 18900.24 | 18783.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 19021.00 | 18848.05 | 18792.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 14:15:00 | 18460.00 | 19312.19 | 19239.74 | SL hit (close<static) qty=1.00 sl=18772.00 alert=retest2 |

### Cycle 191 — SELL (started 2026-01-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 15:15:00 | 18388.00 | 19127.35 | 19162.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 17810.00 | 18863.88 | 19039.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 17446.00 | 17444.05 | 17887.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:45:00 | 17539.00 | 17444.05 | 17887.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 16903.00 | 16533.68 | 16799.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:45:00 | 16861.00 | 16533.68 | 16799.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 16795.00 | 16585.95 | 16798.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 16724.00 | 16796.87 | 16838.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 10:15:00 | 16675.00 | 16790.29 | 16831.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 12:30:00 | 16655.00 | 16772.90 | 16815.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 14:45:00 | 16715.00 | 16706.70 | 16775.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 16665.00 | 16687.69 | 16754.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 10:30:00 | 16538.00 | 16688.55 | 16748.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 14:00:00 | 16555.00 | 16698.48 | 16741.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 11:15:00 | 16858.00 | 16759.94 | 16755.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 192 — BUY (started 2026-01-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 11:15:00 | 16858.00 | 16759.94 | 16755.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 12:15:00 | 16926.00 | 16793.15 | 16770.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 09:15:00 | 16637.00 | 16796.82 | 16784.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 09:15:00 | 16637.00 | 16796.82 | 16784.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 16637.00 | 16796.82 | 16784.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:45:00 | 16633.00 | 16796.82 | 16784.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 193 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 16609.00 | 16759.26 | 16768.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 16484.00 | 16704.20 | 16743.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 16829.00 | 16616.19 | 16671.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 16829.00 | 16616.19 | 16671.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 16829.00 | 16616.19 | 16671.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:00:00 | 16829.00 | 16616.19 | 16671.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 16596.00 | 16612.15 | 16664.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 11:15:00 | 16585.00 | 16612.15 | 16664.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 14:15:00 | 16587.00 | 16593.37 | 16640.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 15:15:00 | 16835.00 | 16657.96 | 16663.06 | SL hit (close>static) qty=1.00 sl=16831.00 alert=retest2 |

### Cycle 194 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 17116.00 | 16749.57 | 16704.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 17457.00 | 16968.72 | 16816.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 18549.00 | 18649.06 | 18288.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-01 12:00:00 | 18549.00 | 18649.06 | 18288.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 18456.00 | 18610.45 | 18303.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 18388.00 | 18610.45 | 18303.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 18012.00 | 18462.37 | 18287.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 18012.00 | 18462.37 | 18287.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 18128.00 | 18395.49 | 18273.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 18466.00 | 18395.49 | 18273.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-03 09:15:00 | 20312.60 | 18903.03 | 18609.08 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 195 — SELL (started 2026-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-05 13:15:00 | 19051.00 | 19085.02 | 19085.40 | EMA200 below EMA400 |

### Cycle 196 — BUY (started 2026-02-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 14:15:00 | 19189.00 | 19105.82 | 19094.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 09:15:00 | 21560.00 | 19603.72 | 19323.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 15:15:00 | 22391.00 | 22391.54 | 21883.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-11 09:15:00 | 22488.00 | 22391.54 | 21883.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 22252.00 | 22673.99 | 22527.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 22366.00 | 22673.99 | 22527.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 22333.00 | 22605.79 | 22509.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:45:00 | 22480.00 | 22600.43 | 22515.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 15:15:00 | 22773.00 | 23013.55 | 23014.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 197 — SELL (started 2026-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 15:15:00 | 22773.00 | 23013.55 | 23014.36 | EMA200 below EMA400 |

### Cycle 198 — BUY (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 09:15:00 | 23314.00 | 23073.64 | 23041.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-20 10:15:00 | 23620.00 | 23182.91 | 23094.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 15:15:00 | 25508.00 | 25508.59 | 25259.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-02 09:15:00 | 25660.00 | 25508.59 | 25259.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 25780.00 | 25562.87 | 25307.22 | EMA400 retest candle locked (from upside) |

### Cycle 199 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 24510.00 | 25182.06 | 25245.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 24335.00 | 25012.65 | 25163.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 25135.00 | 24764.90 | 24951.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 25135.00 | 24764.90 | 24951.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 25135.00 | 24764.90 | 24951.20 | EMA400 retest candle locked (from downside) |

### Cycle 200 — BUY (started 2026-03-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 14:15:00 | 25240.00 | 25080.28 | 25060.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 15:15:00 | 25400.00 | 25144.22 | 25091.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 24945.00 | 25544.33 | 25402.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 24945.00 | 25544.33 | 25402.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 24945.00 | 25544.33 | 25402.22 | EMA400 retest candle locked (from upside) |

### Cycle 201 — SELL (started 2026-03-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 11:15:00 | 24825.00 | 25246.97 | 25282.21 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2026-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 09:15:00 | 25890.00 | 25283.95 | 25266.65 | EMA200 above EMA400 |

### Cycle 203 — SELL (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 10:15:00 | 25045.00 | 25291.18 | 25303.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 11:15:00 | 24940.00 | 25220.94 | 25270.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 10:15:00 | 25130.00 | 24965.03 | 25095.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 10:15:00 | 25130.00 | 24965.03 | 25095.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 25130.00 | 24965.03 | 25095.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 11:00:00 | 25130.00 | 24965.03 | 25095.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 11:15:00 | 24915.00 | 24955.02 | 25078.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 12:00:00 | 24915.00 | 24955.02 | 25078.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 24645.00 | 24833.90 | 24966.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 10:15:00 | 24575.00 | 24833.90 | 24966.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 14:15:00 | 24770.00 | 24454.32 | 24427.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 204 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 24770.00 | 24454.32 | 24427.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 25010.00 | 24598.36 | 24499.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 24565.00 | 24826.65 | 24704.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 24565.00 | 24826.65 | 24704.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 24565.00 | 24826.65 | 24704.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 10:30:00 | 24810.00 | 24803.32 | 24705.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 12:15:00 | 24765.00 | 24785.65 | 24706.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 13:30:00 | 24795.00 | 24772.02 | 24713.26 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-19 14:30:00 | 24765.00 | 24739.61 | 24703.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 15:15:00 | 24750.00 | 24741.69 | 24708.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-20 09:15:00 | 25630.00 | 24741.69 | 24708.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-23 09:15:00 | 24245.00 | 24958.83 | 24920.84 | SL hit (close<static) qty=1.00 sl=24335.00 alert=retest2 |

### Cycle 205 — SELL (started 2026-03-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 10:15:00 | 24055.00 | 24778.06 | 24842.13 | EMA200 below EMA400 |

### Cycle 206 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 25140.00 | 24732.35 | 24716.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 25680.00 | 25050.79 | 24882.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 25350.00 | 25479.57 | 25236.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 25350.00 | 25479.57 | 25236.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 25350.00 | 25479.57 | 25236.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:30:00 | 25195.00 | 25479.57 | 25236.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 25150.00 | 25413.66 | 25228.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 25150.00 | 25413.66 | 25228.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 24960.00 | 25322.93 | 25204.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 12:00:00 | 24960.00 | 25322.93 | 25204.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 24920.00 | 25242.34 | 25178.22 | EMA400 retest candle locked (from upside) |

### Cycle 207 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 24710.00 | 25079.50 | 25111.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 24350.00 | 24880.88 | 25012.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 24955.00 | 24550.45 | 24720.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 24955.00 | 24550.45 | 24720.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 24955.00 | 24550.45 | 24720.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 25125.00 | 24550.45 | 24720.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 208 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 25520.00 | 24943.27 | 24873.77 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 24250.00 | 24867.53 | 24869.46 | EMA200 below EMA400 |

### Cycle 210 — BUY (started 2026-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 14:15:00 | 25365.00 | 24922.03 | 24874.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 09:15:00 | 25780.00 | 25163.70 | 24996.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-06 10:15:00 | 25125.00 | 25155.96 | 25008.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-06 11:00:00 | 25125.00 | 25155.96 | 25008.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 13:15:00 | 25050.00 | 25179.29 | 25060.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 13:45:00 | 24975.00 | 25179.29 | 25060.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 24980.00 | 25139.43 | 25052.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 14:45:00 | 24980.00 | 25139.43 | 25052.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 15:15:00 | 25080.00 | 25127.55 | 25055.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 09:15:00 | 24850.00 | 25127.55 | 25055.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 24985.00 | 25099.04 | 25049.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:30:00 | 25070.00 | 25054.23 | 25033.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 11:45:00 | 25155.00 | 25051.38 | 25033.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 12:15:00 | 25065.00 | 25051.38 | 25033.84 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 13:00:00 | 25090.00 | 25059.11 | 25038.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 13:15:00 | 25055.00 | 25058.29 | 25040.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 13:45:00 | 25055.00 | 25058.29 | 25040.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 25045.00 | 25055.63 | 25040.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 15:00:00 | 25045.00 | 25055.63 | 25040.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-07 15:15:00 | 24930.00 | 25030.50 | 25030.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 211 — SELL (started 2026-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 15:15:00 | 24930.00 | 25030.50 | 25030.74 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 25440.00 | 25112.40 | 25067.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 12:15:00 | 25995.00 | 25462.63 | 25254.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 14:15:00 | 28305.00 | 28398.31 | 27777.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-13 15:00:00 | 28305.00 | 28398.31 | 27777.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 27935.00 | 28375.95 | 28149.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:45:00 | 27925.00 | 28375.95 | 28149.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 27995.00 | 28299.76 | 28135.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 14:45:00 | 28215.00 | 28129.50 | 28089.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-23 09:15:00 | 31036.50 | 30422.90 | 30036.50 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 213 — SELL (started 2026-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-06 15:15:00 | 33425.00 | 33583.77 | 33587.28 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 34415.00 | 33750.01 | 33662.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 10:15:00 | 34555.00 | 33911.01 | 33743.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 34095.00 | 34400.52 | 34116.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 34095.00 | 34400.52 | 34116.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 34095.00 | 34400.52 | 34116.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:00:00 | 34095.00 | 34400.52 | 34116.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 10:15:00 | 33595.00 | 34239.42 | 34068.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:00:00 | 33595.00 | 34239.42 | 34068.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 33570.00 | 34105.54 | 34023.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 12:00:00 | 33570.00 | 34105.54 | 34023.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 33960.00 | 33990.81 | 33982.19 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-06-06 09:45:00 | 3912.80 | 2023-06-08 15:15:00 | 3906.25 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2023-06-15 09:15:00 | 4142.40 | 2023-06-15 14:15:00 | 4022.55 | STOP_HIT | 1.00 | -2.89% |
| BUY | retest2 | 2023-06-21 12:15:00 | 4150.35 | 2023-06-23 14:15:00 | 4090.05 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2023-06-22 13:30:00 | 4124.40 | 2023-06-23 15:15:00 | 4100.00 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2023-06-22 14:30:00 | 4128.05 | 2023-06-23 15:15:00 | 4100.00 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2023-06-22 15:00:00 | 4129.10 | 2023-06-23 15:15:00 | 4100.00 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2023-06-23 11:30:00 | 4140.00 | 2023-06-23 15:15:00 | 4100.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2023-07-10 09:15:00 | 4317.35 | 2023-07-10 15:15:00 | 4211.00 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2023-07-27 10:30:00 | 4011.00 | 2023-08-02 14:15:00 | 4040.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2023-07-27 12:15:00 | 4023.05 | 2023-08-02 15:15:00 | 4015.05 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2023-07-28 09:15:00 | 4030.50 | 2023-08-02 15:15:00 | 4015.05 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2023-07-28 11:15:00 | 4025.00 | 2023-08-02 15:15:00 | 4015.05 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2023-08-02 10:30:00 | 3953.00 | 2023-08-02 15:15:00 | 4015.05 | STOP_HIT | 1.00 | -1.57% |
| BUY | retest2 | 2023-08-08 13:00:00 | 4255.00 | 2023-08-14 12:15:00 | 4330.10 | STOP_HIT | 1.00 | 1.76% |
| SELL | retest2 | 2023-08-18 15:15:00 | 4240.00 | 2023-08-21 13:15:00 | 4340.10 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2023-09-15 12:00:00 | 4400.15 | 2023-09-18 14:15:00 | 4180.14 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2023-09-15 12:00:00 | 4400.15 | 2023-09-21 09:15:00 | 4228.90 | STOP_HIT | 0.50 | 3.89% |
| SELL | retest1 | 2023-10-26 09:15:00 | 4174.15 | 2023-10-27 09:15:00 | 4287.45 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2023-10-27 15:00:00 | 4250.05 | 2023-11-02 10:15:00 | 4300.05 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest1 | 2023-11-13 09:15:00 | 4382.80 | 2023-11-13 11:15:00 | 4452.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2023-11-13 14:30:00 | 4380.65 | 2023-11-16 09:15:00 | 4487.90 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2023-11-22 11:15:00 | 4631.15 | 2023-11-29 09:15:00 | 4756.35 | STOP_HIT | 1.00 | 2.70% |
| BUY | retest2 | 2023-12-05 09:15:00 | 4865.90 | 2023-12-06 12:15:00 | 4757.85 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2023-12-06 10:15:00 | 4850.95 | 2023-12-06 12:15:00 | 4757.85 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2023-12-11 09:15:00 | 4874.20 | 2023-12-11 11:15:00 | 4803.10 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2023-12-13 09:30:00 | 4917.20 | 2023-12-19 14:15:00 | 5066.85 | STOP_HIT | 1.00 | 3.04% |
| BUY | retest2 | 2023-12-13 10:30:00 | 4900.40 | 2023-12-19 14:15:00 | 5066.85 | STOP_HIT | 1.00 | 3.40% |
| SELL | retest2 | 2023-12-21 12:15:00 | 5006.75 | 2023-12-22 11:15:00 | 5104.95 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2023-12-21 13:30:00 | 5027.70 | 2023-12-22 11:15:00 | 5104.95 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2023-12-22 09:15:00 | 5046.35 | 2023-12-22 11:15:00 | 5104.95 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2023-12-27 09:15:00 | 5255.00 | 2024-01-09 10:15:00 | 5780.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-01-31 11:45:00 | 5798.90 | 2024-01-31 13:15:00 | 5909.90 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2024-02-02 10:30:00 | 5774.00 | 2024-02-05 11:15:00 | 6021.15 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest2 | 2024-02-02 12:45:00 | 5767.70 | 2024-02-05 11:15:00 | 6021.15 | STOP_HIT | 1.00 | -4.39% |
| SELL | retest2 | 2024-02-05 09:30:00 | 5767.55 | 2024-02-05 11:15:00 | 6021.15 | STOP_HIT | 1.00 | -4.40% |
| BUY | retest2 | 2024-02-09 12:30:00 | 6230.25 | 2024-02-12 12:15:00 | 6150.05 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-02-09 15:15:00 | 6197.20 | 2024-02-12 12:15:00 | 6150.05 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-02-12 09:30:00 | 6249.15 | 2024-02-12 12:15:00 | 6150.05 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest1 | 2024-02-14 14:15:00 | 5899.50 | 2024-02-16 09:15:00 | 6033.40 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2024-02-15 14:45:00 | 5843.90 | 2024-02-16 09:15:00 | 6033.40 | STOP_HIT | 1.00 | -3.24% |
| SELL | retest2 | 2024-02-23 13:15:00 | 5865.25 | 2024-02-27 10:15:00 | 6000.00 | STOP_HIT | 1.00 | -2.30% |
| SELL | retest2 | 2024-02-26 11:45:00 | 5872.15 | 2024-02-27 10:15:00 | 6000.00 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2024-02-26 13:30:00 | 5874.35 | 2024-02-27 10:15:00 | 6000.00 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2024-02-26 14:30:00 | 5879.80 | 2024-02-27 10:15:00 | 6000.00 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-03-05 09:15:00 | 6184.55 | 2024-03-11 10:15:00 | 6786.01 | TARGET_HIT | 1.00 | 9.73% |
| BUY | retest2 | 2024-03-05 10:15:00 | 6169.10 | 2024-03-11 10:15:00 | 6776.16 | TARGET_HIT | 1.00 | 9.84% |
| BUY | retest2 | 2024-03-05 11:00:00 | 6160.15 | 2024-03-11 10:15:00 | 6798.00 | TARGET_HIT | 1.00 | 10.35% |
| BUY | retest2 | 2024-03-06 12:15:00 | 6180.00 | 2024-03-13 10:15:00 | 6310.30 | STOP_HIT | 1.00 | 2.11% |
| BUY | retest2 | 2024-03-06 14:00:00 | 6271.90 | 2024-03-13 10:15:00 | 6310.30 | STOP_HIT | 1.00 | 0.61% |
| BUY | retest2 | 2024-03-18 11:45:00 | 6630.25 | 2024-03-18 14:15:00 | 6507.55 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2024-03-18 12:45:00 | 6627.30 | 2024-03-18 14:15:00 | 6507.55 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2024-04-23 11:30:00 | 8033.55 | 2024-04-25 11:15:00 | 8836.91 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-05-08 13:00:00 | 9019.15 | 2024-05-13 12:15:00 | 9309.00 | STOP_HIT | 1.00 | -3.21% |
| SELL | retest2 | 2024-05-09 10:00:00 | 9051.20 | 2024-05-13 12:15:00 | 9309.00 | STOP_HIT | 1.00 | -2.85% |
| BUY | retest2 | 2024-05-22 09:15:00 | 11403.55 | 2024-05-22 10:15:00 | 10611.95 | STOP_HIT | 1.00 | -6.94% |
| SELL | retest2 | 2024-05-29 11:45:00 | 10741.10 | 2024-05-30 10:15:00 | 10721.55 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2024-05-29 13:00:00 | 10700.00 | 2024-05-30 10:15:00 | 10721.55 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest1 | 2024-06-10 09:15:00 | 10609.95 | 2024-06-10 15:15:00 | 10425.00 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-06-13 09:15:00 | 10600.00 | 2024-06-20 14:15:00 | 10990.00 | STOP_HIT | 1.00 | 3.68% |
| SELL | retest2 | 2024-06-21 11:45:00 | 10958.65 | 2024-06-24 09:15:00 | 11427.15 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest2 | 2024-06-21 13:30:00 | 10950.95 | 2024-06-24 09:15:00 | 11427.15 | STOP_HIT | 1.00 | -4.35% |
| BUY | retest2 | 2024-06-26 09:15:00 | 11647.40 | 2024-06-26 15:15:00 | 12812.14 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-18 14:15:00 | 11799.55 | 2024-07-19 13:15:00 | 11209.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-19 10:30:00 | 11799.80 | 2024-07-19 13:15:00 | 11209.81 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-19 11:15:00 | 11795.25 | 2024-07-19 13:15:00 | 11205.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-18 14:15:00 | 11799.55 | 2024-07-23 12:15:00 | 11413.00 | STOP_HIT | 0.50 | 3.28% |
| SELL | retest2 | 2024-07-19 10:30:00 | 11799.80 | 2024-07-23 12:15:00 | 11413.00 | STOP_HIT | 0.50 | 3.28% |
| SELL | retest2 | 2024-07-19 11:15:00 | 11795.25 | 2024-07-23 12:15:00 | 11413.00 | STOP_HIT | 0.50 | 3.24% |
| SELL | retest2 | 2024-07-24 10:00:00 | 11832.25 | 2024-07-24 10:15:00 | 11923.45 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-07-29 12:00:00 | 11682.35 | 2024-07-29 14:15:00 | 11900.95 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2024-07-29 12:30:00 | 11598.60 | 2024-07-29 14:15:00 | 11900.95 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2024-08-09 14:00:00 | 10938.45 | 2024-08-12 12:15:00 | 11215.00 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2024-08-09 15:00:00 | 10942.15 | 2024-08-12 12:15:00 | 11215.00 | STOP_HIT | 1.00 | -2.49% |
| SELL | retest2 | 2024-09-02 10:15:00 | 11851.25 | 2024-09-06 12:15:00 | 11765.40 | STOP_HIT | 1.00 | 0.72% |
| SELL | retest2 | 2024-09-02 11:15:00 | 11902.25 | 2024-09-06 12:15:00 | 11765.40 | STOP_HIT | 1.00 | 1.15% |
| SELL | retest2 | 2024-09-03 10:15:00 | 11901.10 | 2024-09-06 12:15:00 | 11765.40 | STOP_HIT | 1.00 | 1.14% |
| BUY | retest2 | 2024-09-25 09:15:00 | 13163.40 | 2024-09-30 14:15:00 | 14479.74 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-04 11:45:00 | 13953.50 | 2024-11-08 11:15:00 | 13918.60 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2024-11-05 09:15:00 | 14245.75 | 2024-11-08 11:15:00 | 13918.60 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2024-11-05 12:45:00 | 13978.15 | 2024-11-08 11:15:00 | 13918.60 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2024-11-21 10:30:00 | 11300.05 | 2024-11-25 09:15:00 | 12005.05 | STOP_HIT | 1.00 | -6.24% |
| SELL | retest2 | 2024-11-22 09:45:00 | 11291.05 | 2024-11-25 09:15:00 | 12005.05 | STOP_HIT | 1.00 | -6.32% |
| SELL | retest2 | 2024-11-22 10:45:00 | 11261.30 | 2024-11-25 09:15:00 | 12005.05 | STOP_HIT | 1.00 | -6.60% |
| BUY | retest2 | 2024-11-26 09:15:00 | 12645.70 | 2024-11-29 11:15:00 | 11981.45 | STOP_HIT | 1.00 | -5.25% |
| BUY | retest2 | 2024-12-04 13:45:00 | 12261.10 | 2024-12-04 15:15:00 | 12079.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2024-12-13 13:00:00 | 12847.40 | 2024-12-18 11:15:00 | 12819.50 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2024-12-23 11:30:00 | 13500.00 | 2024-12-30 09:15:00 | 14850.00 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-23 12:30:00 | 13515.00 | 2024-12-30 09:15:00 | 14866.50 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-07 15:15:00 | 14580.00 | 2025-01-09 12:15:00 | 13851.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-07 15:15:00 | 14580.00 | 2025-01-13 09:15:00 | 13122.00 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-14 09:15:00 | 11415.45 | 2025-02-17 10:15:00 | 10844.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 09:15:00 | 11415.45 | 2025-02-19 09:15:00 | 10649.55 | STOP_HIT | 0.50 | 6.71% |
| SELL | retest2 | 2025-03-13 13:15:00 | 12063.75 | 2025-03-17 09:15:00 | 12666.95 | STOP_HIT | 1.00 | -5.00% |
| SELL | retest2 | 2025-03-13 14:00:00 | 12028.05 | 2025-03-17 09:15:00 | 12666.95 | STOP_HIT | 1.00 | -5.31% |
| SELL | retest2 | 2025-03-20 12:00:00 | 11860.00 | 2025-03-21 10:15:00 | 12086.80 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-03-20 13:00:00 | 11839.15 | 2025-03-21 10:15:00 | 12086.80 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2025-03-20 13:45:00 | 11845.85 | 2025-03-21 10:15:00 | 12086.80 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-03-25 09:30:00 | 12379.10 | 2025-03-25 15:15:00 | 12113.00 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest1 | 2025-04-08 10:45:00 | 11531.75 | 2025-04-11 09:15:00 | 11712.75 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest1 | 2025-04-08 11:30:00 | 11576.80 | 2025-04-11 09:15:00 | 11712.75 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest1 | 2025-04-08 13:30:00 | 11577.40 | 2025-04-11 09:15:00 | 11712.75 | STOP_HIT | 1.00 | -1.17% |
| SELL | retest1 | 2025-04-09 09:15:00 | 11473.10 | 2025-04-11 09:15:00 | 11712.75 | STOP_HIT | 1.00 | -2.09% |
| BUY | retest2 | 2025-05-06 09:30:00 | 14725.00 | 2025-05-06 13:15:00 | 14500.00 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-05-09 13:30:00 | 14823.00 | 2025-05-14 09:15:00 | 16305.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-20 10:45:00 | 18754.00 | 2025-06-30 09:15:00 | 20629.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-20 11:30:00 | 18812.00 | 2025-06-30 09:15:00 | 20693.20 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-20 15:15:00 | 18890.00 | 2025-06-30 09:15:00 | 20694.30 | TARGET_HIT | 1.00 | 9.55% |
| BUY | retest2 | 2025-06-23 09:30:00 | 18819.00 | 2025-07-01 12:15:00 | 19480.00 | STOP_HIT | 1.00 | 3.51% |
| BUY | retest2 | 2025-06-23 13:30:00 | 18813.00 | 2025-07-01 12:15:00 | 19480.00 | STOP_HIT | 1.00 | 3.55% |
| SELL | retest2 | 2025-07-02 12:00:00 | 19890.00 | 2025-07-02 14:15:00 | 19975.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-07-21 09:30:00 | 19135.00 | 2025-07-25 10:15:00 | 19290.00 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2025-07-31 13:00:00 | 20030.00 | 2025-08-06 12:15:00 | 20705.00 | STOP_HIT | 1.00 | 3.37% |
| BUY | retest2 | 2025-07-31 13:30:00 | 20010.00 | 2025-08-06 12:15:00 | 20705.00 | STOP_HIT | 1.00 | 3.47% |
| BUY | retest2 | 2025-07-31 14:30:00 | 20185.00 | 2025-08-06 12:15:00 | 20705.00 | STOP_HIT | 1.00 | 2.58% |
| SELL | retest2 | 2025-08-07 13:15:00 | 20575.00 | 2025-08-11 14:15:00 | 20880.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-08-08 09:15:00 | 20305.00 | 2025-08-11 14:15:00 | 20880.00 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-08-08 14:30:00 | 20465.00 | 2025-08-11 14:15:00 | 20880.00 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-08-11 12:00:00 | 20585.00 | 2025-08-11 14:15:00 | 20880.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-08-13 09:15:00 | 20980.00 | 2025-08-14 12:15:00 | 20485.00 | STOP_HIT | 1.00 | -2.36% |
| SELL | retest2 | 2025-08-20 12:30:00 | 20065.00 | 2025-08-28 13:15:00 | 19950.00 | STOP_HIT | 1.00 | 0.57% |
| SELL | retest2 | 2025-08-20 13:15:00 | 20095.00 | 2025-08-28 13:15:00 | 19950.00 | STOP_HIT | 1.00 | 0.72% |
| SELL | retest2 | 2025-08-20 13:45:00 | 20045.00 | 2025-08-28 13:15:00 | 19950.00 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2025-08-21 09:45:00 | 20045.00 | 2025-08-28 13:15:00 | 19950.00 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2025-08-22 10:15:00 | 19770.00 | 2025-08-28 13:15:00 | 19950.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-08-26 15:15:00 | 19505.00 | 2025-08-28 13:15:00 | 19950.00 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-08-28 09:45:00 | 19840.00 | 2025-08-28 13:15:00 | 19950.00 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-09-09 11:30:00 | 18709.00 | 2025-09-10 09:15:00 | 19394.00 | STOP_HIT | 1.00 | -3.66% |
| SELL | retest2 | 2025-09-09 12:00:00 | 18729.00 | 2025-09-10 09:15:00 | 19394.00 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-09-09 13:15:00 | 18695.00 | 2025-09-10 09:15:00 | 19394.00 | STOP_HIT | 1.00 | -3.74% |
| BUY | retest2 | 2025-09-12 09:15:00 | 19722.00 | 2025-09-18 10:15:00 | 19641.00 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2025-09-29 09:15:00 | 19637.00 | 2025-09-29 12:15:00 | 18978.00 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2025-10-03 12:00:00 | 18113.00 | 2025-10-15 09:15:00 | 17207.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-07 10:15:00 | 18150.00 | 2025-10-15 09:15:00 | 17242.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-07 11:15:00 | 18159.00 | 2025-10-15 09:15:00 | 17251.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-03 12:00:00 | 18113.00 | 2025-10-15 12:15:00 | 17460.00 | STOP_HIT | 0.50 | 3.61% |
| SELL | retest2 | 2025-10-07 10:15:00 | 18150.00 | 2025-10-15 12:15:00 | 17460.00 | STOP_HIT | 0.50 | 3.80% |
| SELL | retest2 | 2025-10-07 11:15:00 | 18159.00 | 2025-10-15 12:15:00 | 17460.00 | STOP_HIT | 0.50 | 3.85% |
| SELL | retest2 | 2025-10-23 12:00:00 | 17317.00 | 2025-10-29 10:15:00 | 17175.00 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2025-11-03 14:00:00 | 17600.00 | 2025-11-04 09:15:00 | 19360.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-11-19 09:15:00 | 21301.00 | 2025-11-19 11:15:00 | 21575.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-11-19 11:00:00 | 21330.00 | 2025-11-19 11:15:00 | 21575.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-11-25 09:15:00 | 21462.00 | 2025-11-25 14:15:00 | 21778.00 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-11-28 09:45:00 | 22020.00 | 2025-12-03 12:15:00 | 21840.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-11-28 13:15:00 | 22040.00 | 2025-12-03 12:15:00 | 21840.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-11-28 14:30:00 | 22005.00 | 2025-12-03 12:15:00 | 21840.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-12-09 13:15:00 | 19325.00 | 2025-12-12 11:15:00 | 19555.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-12-09 14:30:00 | 19400.00 | 2025-12-12 11:15:00 | 19555.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-12-10 09:15:00 | 19405.00 | 2025-12-12 11:15:00 | 19555.00 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-12-17 11:15:00 | 19290.00 | 2025-12-18 12:15:00 | 18325.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-17 11:15:00 | 19290.00 | 2025-12-19 13:15:00 | 18325.00 | STOP_HIT | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-30 09:45:00 | 18215.00 | 2025-12-31 13:15:00 | 18420.00 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-12-30 14:30:00 | 18210.00 | 2025-12-31 13:15:00 | 18420.00 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-12-31 09:45:00 | 18145.00 | 2025-12-31 13:15:00 | 18420.00 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-12-31 11:30:00 | 18220.00 | 2025-12-31 13:15:00 | 18420.00 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-12-31 15:15:00 | 18265.00 | 2026-01-01 09:15:00 | 18326.00 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2026-01-07 09:15:00 | 19021.00 | 2026-01-08 14:15:00 | 18460.00 | STOP_HIT | 1.00 | -2.95% |
| SELL | retest2 | 2026-01-20 09:15:00 | 16724.00 | 2026-01-22 11:15:00 | 16858.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-01-20 10:15:00 | 16675.00 | 2026-01-22 11:15:00 | 16858.00 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2026-01-20 12:30:00 | 16655.00 | 2026-01-22 11:15:00 | 16858.00 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2026-01-20 14:45:00 | 16715.00 | 2026-01-22 11:15:00 | 16858.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-01-21 10:30:00 | 16538.00 | 2026-01-22 11:15:00 | 16858.00 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2026-01-21 14:00:00 | 16555.00 | 2026-01-22 11:15:00 | 16858.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-01-27 11:15:00 | 16585.00 | 2026-01-27 15:15:00 | 16835.00 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2026-01-27 14:15:00 | 16587.00 | 2026-01-27 15:15:00 | 16835.00 | STOP_HIT | 1.00 | -1.50% |
| BUY | retest2 | 2026-02-02 09:15:00 | 18466.00 | 2026-02-03 09:15:00 | 20312.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-13 11:45:00 | 22480.00 | 2026-02-19 15:15:00 | 22773.00 | STOP_HIT | 1.00 | 1.30% |
| SELL | retest2 | 2026-03-13 10:15:00 | 24575.00 | 2026-03-17 14:15:00 | 24770.00 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2026-03-19 10:30:00 | 24810.00 | 2026-03-23 09:15:00 | 24245.00 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2026-03-19 12:15:00 | 24765.00 | 2026-03-23 09:15:00 | 24245.00 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2026-03-19 13:30:00 | 24795.00 | 2026-03-23 09:15:00 | 24245.00 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2026-03-19 14:30:00 | 24765.00 | 2026-03-23 09:15:00 | 24245.00 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2026-03-20 09:15:00 | 25630.00 | 2026-03-23 09:15:00 | 24245.00 | STOP_HIT | 1.00 | -5.40% |
| BUY | retest2 | 2026-04-07 10:30:00 | 25070.00 | 2026-04-07 15:15:00 | 24930.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2026-04-07 11:45:00 | 25155.00 | 2026-04-07 15:15:00 | 24930.00 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2026-04-07 12:15:00 | 25065.00 | 2026-04-07 15:15:00 | 24930.00 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2026-04-07 13:00:00 | 25090.00 | 2026-04-07 15:15:00 | 24930.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2026-04-16 14:45:00 | 28215.00 | 2026-04-23 09:15:00 | 31036.50 | TARGET_HIT | 1.00 | 10.00% |
