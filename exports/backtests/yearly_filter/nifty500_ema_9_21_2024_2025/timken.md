# Timken India Ltd. (TIMKEN)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3709 bars)
- **Last close:** 3600.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 140 |
| ALERT1 | 91 |
| ALERT2 | 88 |
| ALERT2_SKIP | 43 |
| ALERT3 | 250 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 16 |
| ENTRY2 | 126 |
| PARTIAL | 5 |
| TARGET_HIT | 10 |
| STOP_HIT | 132 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 147 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 62 / 85
- **Target hits / Stop hits / Partials:** 10 / 132 / 5
- **Avg / median % per leg:** 0.28% / -0.85%
- **Sum % (uncompounded):** 41.48%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 86 | 39 | 45.3% | 8 | 78 | 0 | 0.73% | 62.5% |
| BUY @ 2nd Alert (retest1) | 9 | 3 | 33.3% | 0 | 9 | 0 | -0.61% | -5.5% |
| BUY @ 3rd Alert (retest2) | 77 | 36 | 46.8% | 8 | 69 | 0 | 0.88% | 67.9% |
| SELL (all) | 61 | 23 | 37.7% | 2 | 54 | 5 | -0.34% | -21.0% |
| SELL @ 2nd Alert (retest1) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.97% | -13.8% |
| SELL @ 3rd Alert (retest2) | 54 | 23 | 42.6% | 2 | 47 | 5 | -0.13% | -7.2% |
| retest1 (combined) | 16 | 3 | 18.8% | 0 | 16 | 0 | -1.20% | -19.2% |
| retest2 (combined) | 131 | 59 | 45.0% | 10 | 116 | 5 | 0.46% | 60.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 13:15:00 | 3925.00 | 4050.50 | 4059.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 11:15:00 | 3871.70 | 3937.68 | 3976.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 09:15:00 | 4063.95 | 3926.41 | 3950.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 4063.95 | 3926.41 | 3950.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 4063.95 | 3926.41 | 3950.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 09:30:00 | 4155.10 | 3926.41 | 3950.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 10:15:00 | 4191.80 | 3979.49 | 3972.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-24 12:15:00 | 4316.00 | 4079.61 | 4021.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 15:15:00 | 4117.00 | 4119.57 | 4057.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 15:15:00 | 4117.00 | 4119.57 | 4057.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 15:15:00 | 4117.00 | 4119.57 | 4057.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 09:15:00 | 4147.45 | 4119.57 | 4057.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 09:15:00 | 4050.00 | 4105.66 | 4057.14 | SL hit (close<static) qty=1.00 sl=4051.30 alert=retest2 |

### Cycle 3 — SELL (started 2024-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 13:15:00 | 3972.00 | 4027.53 | 4030.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-27 14:15:00 | 3951.10 | 4012.24 | 4023.47 | Break + close below crossover candle low |

### Cycle 4 — BUY (started 2024-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 09:15:00 | 4182.90 | 4036.41 | 4031.89 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 12:15:00 | 4044.85 | 4060.67 | 4060.94 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-05-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-29 13:15:00 | 4067.00 | 4061.94 | 4061.49 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 14:15:00 | 3945.65 | 4038.68 | 4050.96 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 11:15:00 | 4097.80 | 4004.06 | 4000.89 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 10:15:00 | 3822.85 | 3987.22 | 3999.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 11:15:00 | 3754.00 | 3940.57 | 3977.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 13:15:00 | 3880.00 | 3870.79 | 3909.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 14:00:00 | 3880.00 | 3870.79 | 3909.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 14:15:00 | 3880.75 | 3872.78 | 3906.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-05 15:00:00 | 3880.75 | 3872.78 | 3906.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 4113.65 | 3918.91 | 3921.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 4089.90 | 3918.91 | 3921.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 4137.65 | 3962.66 | 3941.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 11:15:00 | 4184.95 | 4007.12 | 3963.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 11:15:00 | 4090.90 | 4121.50 | 4087.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 12:00:00 | 4090.90 | 4121.50 | 4087.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 12:15:00 | 4082.25 | 4113.65 | 4086.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-10 13:00:00 | 4082.25 | 4113.65 | 4086.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 13:15:00 | 4076.45 | 4106.21 | 4085.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 11:45:00 | 4092.50 | 4093.54 | 4085.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 13:30:00 | 4101.25 | 4105.90 | 4099.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-14 09:15:00 | 4501.75 | 4321.95 | 4225.48 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 10:15:00 | 4493.80 | 4579.38 | 4583.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 14:15:00 | 4425.65 | 4526.70 | 4554.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 11:15:00 | 4514.85 | 4509.42 | 4535.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 11:15:00 | 4514.85 | 4509.42 | 4535.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 4514.85 | 4509.42 | 4535.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:30:00 | 4521.50 | 4509.42 | 4535.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 4514.00 | 4510.34 | 4533.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 12:30:00 | 4515.15 | 4510.34 | 4533.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 4642.10 | 4513.66 | 4525.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 09:30:00 | 4607.05 | 4513.66 | 4525.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — BUY (started 2024-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 10:15:00 | 4637.25 | 4538.38 | 4535.29 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 15:15:00 | 4500.00 | 4535.31 | 4537.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 09:15:00 | 4446.00 | 4517.45 | 4528.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 14:15:00 | 4315.95 | 4310.82 | 4348.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-01 14:15:00 | 4315.95 | 4310.82 | 4348.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 14:15:00 | 4315.95 | 4310.82 | 4348.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 15:00:00 | 4315.95 | 4310.82 | 4348.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 4311.85 | 4313.76 | 4343.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-02 09:30:00 | 4332.35 | 4313.76 | 4343.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 15:15:00 | 4310.00 | 4294.15 | 4318.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 09:15:00 | 4390.40 | 4294.15 | 4318.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 09:15:00 | 4431.35 | 4321.59 | 4328.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-03 10:15:00 | 4417.75 | 4321.59 | 4328.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2024-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 10:15:00 | 4427.95 | 4342.86 | 4337.79 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 10:15:00 | 4339.80 | 4368.43 | 4369.98 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 09:15:00 | 4474.00 | 4374.84 | 4369.37 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-07-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 12:15:00 | 4318.00 | 4362.87 | 4365.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 13:15:00 | 4287.10 | 4347.72 | 4358.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-12 09:15:00 | 4144.05 | 4134.39 | 4190.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 09:15:00 | 4144.05 | 4134.39 | 4190.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 4144.05 | 4134.39 | 4190.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:30:00 | 4178.00 | 4134.39 | 4190.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 3950.00 | 3944.97 | 3993.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-18 14:45:00 | 3924.00 | 3934.27 | 3969.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 09:45:00 | 3902.25 | 3920.03 | 3956.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 14:00:00 | 3906.05 | 3896.47 | 3931.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 10:15:00 | 3919.25 | 3904.86 | 3926.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 3882.00 | 3900.29 | 3922.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 12:45:00 | 3867.00 | 3888.45 | 3913.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 3830.75 | 3901.43 | 3908.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:15:00 | 3865.00 | 3900.38 | 3907.16 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 15:15:00 | 3860.00 | 3893.42 | 3902.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 4002.55 | 3909.90 | 3908.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 09:15:00 | 4002.55 | 3909.90 | 3908.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 09:15:00 | 4096.05 | 3996.13 | 3959.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 14:15:00 | 4007.70 | 4071.84 | 4039.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 14:15:00 | 4007.70 | 4071.84 | 4039.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 14:15:00 | 4007.70 | 4071.84 | 4039.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-26 15:00:00 | 4007.70 | 4071.84 | 4039.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 15:15:00 | 4023.00 | 4062.07 | 4037.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 09:15:00 | 4128.25 | 4062.07 | 4037.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 10:15:00 | 4190.10 | 4245.06 | 4246.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2024-08-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-05 10:15:00 | 4190.10 | 4245.06 | 4246.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 11:15:00 | 4105.65 | 4217.17 | 4233.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 4269.80 | 4198.85 | 4214.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 4269.80 | 4198.85 | 4214.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 4269.80 | 4198.85 | 4214.04 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2024-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 12:15:00 | 4257.85 | 4229.99 | 4226.30 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2024-08-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 15:15:00 | 4200.05 | 4222.27 | 4223.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-07 09:15:00 | 4090.55 | 4195.93 | 4211.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-08 12:15:00 | 4178.55 | 4113.91 | 4142.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-08 12:15:00 | 4178.55 | 4113.91 | 4142.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 4178.55 | 4113.91 | 4142.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 12:45:00 | 4179.00 | 4113.91 | 4142.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 13:15:00 | 4255.45 | 4142.22 | 4152.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 14:00:00 | 4255.45 | 4142.22 | 4152.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 12:15:00 | 3641.00 | 3615.62 | 3640.09 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2024-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 15:15:00 | 3711.00 | 3660.43 | 3656.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-22 10:15:00 | 3750.05 | 3685.93 | 3669.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 14:15:00 | 3705.30 | 3709.34 | 3688.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-22 14:15:00 | 3705.30 | 3709.34 | 3688.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 3705.30 | 3709.34 | 3688.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 14:30:00 | 3693.00 | 3709.34 | 3688.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 3727.65 | 3711.82 | 3692.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 10:15:00 | 3745.55 | 3711.82 | 3692.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 13:15:00 | 3773.55 | 3727.24 | 3721.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 13:00:00 | 3743.20 | 3733.22 | 3729.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 10:15:00 | 3682.20 | 3747.75 | 3756.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 3682.20 | 3747.75 | 3756.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-29 11:15:00 | 3671.05 | 3732.41 | 3748.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-30 09:15:00 | 3754.70 | 3715.91 | 3731.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-30 09:15:00 | 3754.70 | 3715.91 | 3731.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 09:15:00 | 3754.70 | 3715.91 | 3731.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 09:30:00 | 3753.00 | 3715.91 | 3731.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-30 10:15:00 | 3763.00 | 3725.32 | 3734.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-30 10:45:00 | 3767.00 | 3725.32 | 3734.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2024-08-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 12:15:00 | 3813.05 | 3749.48 | 3744.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 13:15:00 | 3816.50 | 3762.89 | 3750.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 12:15:00 | 3802.75 | 3802.87 | 3781.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 13:15:00 | 3789.80 | 3800.26 | 3782.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 3789.80 | 3800.26 | 3782.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 13:30:00 | 3784.40 | 3800.26 | 3782.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 3784.30 | 3797.06 | 3782.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 14:45:00 | 3785.70 | 3797.06 | 3782.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 15:15:00 | 3791.10 | 3795.87 | 3783.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 09:15:00 | 3860.40 | 3795.87 | 3783.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-05 14:15:00 | 3794.00 | 3836.10 | 3838.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2024-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 14:15:00 | 3794.00 | 3836.10 | 3838.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 09:15:00 | 3791.25 | 3821.35 | 3831.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 15:15:00 | 3710.00 | 3706.11 | 3742.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 09:15:00 | 3731.00 | 3706.11 | 3742.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 3716.25 | 3708.14 | 3740.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-10 09:30:00 | 3754.00 | 3708.14 | 3740.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 09:15:00 | 3724.50 | 3708.79 | 3723.95 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 10:15:00 | 3786.25 | 3735.87 | 3730.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 13:15:00 | 3800.00 | 3762.29 | 3745.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 09:15:00 | 3771.00 | 3781.23 | 3759.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 10:00:00 | 3771.00 | 3781.23 | 3759.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 10:15:00 | 3750.75 | 3775.13 | 3758.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 11:00:00 | 3750.75 | 3775.13 | 3758.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 11:15:00 | 3754.00 | 3770.91 | 3758.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 12:00:00 | 3754.00 | 3770.91 | 3758.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 12:15:00 | 3760.00 | 3768.72 | 3758.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 12:30:00 | 3750.70 | 3768.72 | 3758.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 13:15:00 | 3770.00 | 3768.98 | 3759.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 09:30:00 | 3786.60 | 3780.85 | 3767.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 09:15:00 | 3748.10 | 3788.58 | 3781.67 | SL hit (close<static) qty=1.00 sl=3749.10 alert=retest2 |

### Cycle 27 — SELL (started 2024-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 11:15:00 | 3749.05 | 3772.91 | 3775.27 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2024-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 14:15:00 | 3805.95 | 3764.51 | 3763.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 15:15:00 | 3830.00 | 3777.61 | 3769.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 11:15:00 | 3799.60 | 3832.34 | 3813.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-23 11:15:00 | 3799.60 | 3832.34 | 3813.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 11:15:00 | 3799.60 | 3832.34 | 3813.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 12:00:00 | 3799.60 | 3832.34 | 3813.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 12:15:00 | 3793.00 | 3824.47 | 3811.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 12:30:00 | 3788.00 | 3824.47 | 3811.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 14:15:00 | 3775.15 | 3810.00 | 3806.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 15:00:00 | 3775.15 | 3810.00 | 3806.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 3845.00 | 3863.90 | 3845.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:00:00 | 3845.00 | 3863.90 | 3845.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 3848.40 | 3860.80 | 3845.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:30:00 | 3837.45 | 3860.80 | 3845.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 3832.10 | 3855.06 | 3844.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 13:30:00 | 3831.90 | 3855.06 | 3844.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 3833.50 | 3850.75 | 3843.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 14:45:00 | 3831.30 | 3850.75 | 3843.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — SELL (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 09:15:00 | 3769.75 | 3832.03 | 3835.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 13:15:00 | 3757.85 | 3777.54 | 3795.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 14:15:00 | 3810.00 | 3784.03 | 3796.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 14:15:00 | 3810.00 | 3784.03 | 3796.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 14:15:00 | 3810.00 | 3784.03 | 3796.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 15:00:00 | 3810.00 | 3784.03 | 3796.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 15:15:00 | 3805.00 | 3788.22 | 3797.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 09:15:00 | 3774.75 | 3788.22 | 3797.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 15:00:00 | 3750.05 | 3785.06 | 3792.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 14:30:00 | 3791.50 | 3788.78 | 3789.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 3586.01 | 3654.64 | 3695.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 10:15:00 | 3601.92 | 3654.64 | 3695.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 09:15:00 | 3621.40 | 3617.99 | 3655.36 | SL hit (close>ema200) qty=0.50 sl=3617.99 alert=retest2 |

### Cycle 30 — BUY (started 2024-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 13:15:00 | 3719.25 | 3657.25 | 3650.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 14:15:00 | 3735.10 | 3672.82 | 3658.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 13:15:00 | 3731.30 | 3733.75 | 3701.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 14:00:00 | 3731.30 | 3733.75 | 3701.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 3718.55 | 3730.13 | 3708.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:30:00 | 3704.95 | 3730.13 | 3708.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 3703.00 | 3724.70 | 3707.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 10:30:00 | 3698.65 | 3724.70 | 3707.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 3706.50 | 3721.06 | 3707.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 15:00:00 | 3730.00 | 3717.19 | 3708.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:45:00 | 3729.95 | 3724.35 | 3713.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-16 14:15:00 | 3750.00 | 3771.55 | 3771.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2024-10-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 14:15:00 | 3750.00 | 3771.55 | 3771.86 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2024-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 15:15:00 | 3820.30 | 3781.30 | 3776.26 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 09:15:00 | 3652.55 | 3770.64 | 3776.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 09:15:00 | 3567.95 | 3645.33 | 3682.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-25 14:15:00 | 3375.30 | 3356.84 | 3408.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-25 15:00:00 | 3375.30 | 3356.84 | 3408.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 3348.90 | 3324.70 | 3356.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 09:45:00 | 3350.40 | 3324.70 | 3356.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 3297.20 | 3316.27 | 3338.08 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2024-10-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 14:15:00 | 3405.25 | 3355.96 | 3350.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 15:15:00 | 3420.90 | 3368.94 | 3356.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 3381.60 | 3393.61 | 3379.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 3381.60 | 3393.61 | 3379.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 3381.60 | 3393.61 | 3379.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-04 10:45:00 | 3453.95 | 3404.20 | 3385.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 09:15:00 | 3330.35 | 3399.56 | 3407.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2024-11-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-06 09:15:00 | 3330.35 | 3399.56 | 3407.69 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2024-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 15:15:00 | 3401.00 | 3391.05 | 3390.97 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-11-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 09:15:00 | 3379.50 | 3388.74 | 3389.93 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2024-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-08 11:15:00 | 3397.90 | 3391.20 | 3390.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-08 12:15:00 | 3409.70 | 3394.90 | 3392.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-11 15:15:00 | 3380.00 | 3402.26 | 3399.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 15:15:00 | 3380.00 | 3402.26 | 3399.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 15:15:00 | 3380.00 | 3402.26 | 3399.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 09:15:00 | 3365.10 | 3402.26 | 3399.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 3383.85 | 3398.58 | 3398.44 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2024-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-12 10:15:00 | 3372.10 | 3393.28 | 3396.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 11:15:00 | 3365.25 | 3387.68 | 3393.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 12:15:00 | 3332.40 | 3331.63 | 3356.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-13 13:00:00 | 3332.40 | 3331.63 | 3356.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 3358.25 | 3335.64 | 3350.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 3358.25 | 3335.64 | 3350.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 3322.95 | 3333.11 | 3347.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 11:15:00 | 3293.80 | 3333.11 | 3347.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-21 10:15:00 | 3298.45 | 3262.29 | 3258.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2024-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 10:15:00 | 3298.45 | 3262.29 | 3258.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-21 11:15:00 | 3337.90 | 3277.42 | 3265.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 12:15:00 | 3264.15 | 3274.76 | 3265.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 12:15:00 | 3264.15 | 3274.76 | 3265.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 12:15:00 | 3264.15 | 3274.76 | 3265.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 12:45:00 | 3254.50 | 3274.76 | 3265.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 13:15:00 | 3253.55 | 3270.52 | 3264.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 09:15:00 | 3283.05 | 3263.40 | 3262.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 10:15:00 | 3280.50 | 3262.68 | 3261.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 15:15:00 | 3320.00 | 3333.18 | 3334.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2024-11-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 15:15:00 | 3320.00 | 3333.18 | 3334.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-29 10:15:00 | 3309.45 | 3326.08 | 3330.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-29 13:15:00 | 3330.00 | 3324.56 | 3328.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 13:15:00 | 3330.00 | 3324.56 | 3328.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 13:15:00 | 3330.00 | 3324.56 | 3328.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 14:00:00 | 3330.00 | 3324.56 | 3328.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2024-11-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 14:15:00 | 3378.55 | 3335.35 | 3333.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 10:15:00 | 3400.05 | 3359.71 | 3345.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 14:15:00 | 3419.80 | 3451.13 | 3430.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 14:15:00 | 3419.80 | 3451.13 | 3430.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 3419.80 | 3451.13 | 3430.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 15:00:00 | 3419.80 | 3451.13 | 3430.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 15:15:00 | 3405.85 | 3442.08 | 3427.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 09:15:00 | 3392.85 | 3442.08 | 3427.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 10:15:00 | 3364.10 | 3416.59 | 3418.22 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2024-12-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 10:15:00 | 3476.00 | 3416.44 | 3412.49 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-12-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 13:15:00 | 3403.70 | 3410.42 | 3410.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 14:15:00 | 3388.40 | 3406.01 | 3408.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 14:15:00 | 3187.05 | 3186.17 | 3218.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-16 15:00:00 | 3187.05 | 3186.17 | 3218.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 09:15:00 | 3181.40 | 3181.83 | 3210.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 11:15:00 | 3153.80 | 3178.12 | 3206.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 11:15:00 | 3162.80 | 3153.18 | 3176.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 14:00:00 | 3162.45 | 3158.95 | 3174.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 14:45:00 | 3161.35 | 3160.15 | 3173.24 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 11:15:00 | 3153.80 | 3151.97 | 3164.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 11:30:00 | 3155.80 | 3151.97 | 3164.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 12:15:00 | 3167.85 | 3155.14 | 3165.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 13:00:00 | 3167.85 | 3155.14 | 3165.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 13:15:00 | 3179.95 | 3160.11 | 3166.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:00:00 | 3179.95 | 3160.11 | 3166.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 14:15:00 | 3195.20 | 3167.12 | 3169.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-19 14:30:00 | 3191.00 | 3167.12 | 3169.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-19 15:15:00 | 3195.00 | 3172.70 | 3171.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2024-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 15:15:00 | 3195.00 | 3172.70 | 3171.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-20 09:15:00 | 3244.65 | 3187.09 | 3178.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 15:15:00 | 3235.05 | 3240.03 | 3213.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-23 09:15:00 | 3215.90 | 3240.03 | 3213.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 09:15:00 | 3210.45 | 3234.11 | 3213.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 10:15:00 | 3212.20 | 3234.11 | 3213.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 10:15:00 | 3224.40 | 3232.17 | 3214.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-23 11:15:00 | 3245.80 | 3232.17 | 3214.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-23 13:15:00 | 3198.30 | 3227.91 | 3217.24 | SL hit (close<static) qty=1.00 sl=3200.05 alert=retest2 |

### Cycle 47 — SELL (started 2024-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 09:15:00 | 3187.25 | 3209.03 | 3210.41 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 13:15:00 | 3220.05 | 3212.61 | 3211.62 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2024-12-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 10:15:00 | 3201.35 | 3209.47 | 3210.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 13:15:00 | 3168.05 | 3191.78 | 3201.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 10:15:00 | 3081.00 | 3062.30 | 3097.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 10:15:00 | 3081.00 | 3062.30 | 3097.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 3081.00 | 3062.30 | 3097.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:45:00 | 3082.15 | 3062.30 | 3097.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 11:15:00 | 3116.90 | 3073.22 | 3099.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 12:00:00 | 3116.90 | 3073.22 | 3099.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 3087.35 | 3076.04 | 3097.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-01 09:45:00 | 3078.00 | 3090.48 | 3099.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-06 10:15:00 | 2924.10 | 2980.56 | 3009.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-08 09:15:00 | 2939.70 | 2932.43 | 2953.24 | SL hit (close>ema200) qty=0.50 sl=2932.43 alert=retest2 |

### Cycle 50 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 2860.20 | 2841.34 | 2840.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 11:15:00 | 2879.20 | 2848.91 | 2844.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 11:15:00 | 2854.05 | 2877.29 | 2865.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 11:15:00 | 2854.05 | 2877.29 | 2865.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 2854.05 | 2877.29 | 2865.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:45:00 | 2850.10 | 2877.29 | 2865.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 12:15:00 | 2838.80 | 2869.59 | 2863.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 13:00:00 | 2838.80 | 2869.59 | 2863.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 15:15:00 | 2860.80 | 2862.91 | 2861.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:15:00 | 2858.85 | 2862.91 | 2861.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 2856.00 | 2861.53 | 2860.72 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2025-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 11:15:00 | 2850.05 | 2860.08 | 2860.26 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2025-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 12:15:00 | 2871.20 | 2862.31 | 2861.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 13:15:00 | 2881.45 | 2866.14 | 2863.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-21 09:15:00 | 2821.50 | 2869.00 | 2866.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-21 09:15:00 | 2821.50 | 2869.00 | 2866.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 2821.50 | 2869.00 | 2866.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-21 10:00:00 | 2821.50 | 2869.00 | 2866.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 2809.70 | 2857.14 | 2860.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-22 09:15:00 | 2783.85 | 2825.18 | 2841.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 13:15:00 | 2821.80 | 2817.37 | 2831.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 14:00:00 | 2821.80 | 2817.37 | 2831.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 14:15:00 | 2819.25 | 2817.75 | 2830.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-22 14:30:00 | 2829.00 | 2817.75 | 2830.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 2877.30 | 2827.94 | 2832.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 2877.30 | 2827.94 | 2832.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 2896.65 | 2841.68 | 2838.72 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 2721.40 | 2839.19 | 2850.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 10:15:00 | 2703.95 | 2812.14 | 2836.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 09:15:00 | 2779.85 | 2775.23 | 2802.47 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-28 11:15:00 | 2739.05 | 2776.18 | 2800.42 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 2813.15 | 2755.25 | 2775.48 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 2813.15 | 2755.25 | 2775.48 | SL hit (close>ema400) qty=1.00 sl=2775.48 alert=retest1 |

### Cycle 56 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 2826.30 | 2790.25 | 2787.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 2868.40 | 2805.88 | 2794.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 14:15:00 | 2839.95 | 2844.86 | 2825.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 15:00:00 | 2839.95 | 2844.86 | 2825.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 2822.00 | 2840.28 | 2824.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:15:00 | 2823.50 | 2840.28 | 2824.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 2832.45 | 2838.72 | 2825.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 10:45:00 | 2853.35 | 2843.05 | 2828.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 12:45:00 | 2849.10 | 2843.79 | 2831.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 09:15:00 | 2788.10 | 2860.76 | 2854.68 | SL hit (close<static) qty=1.00 sl=2820.00 alert=retest2 |

### Cycle 57 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 2750.95 | 2838.80 | 2845.25 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2025-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 11:15:00 | 2830.00 | 2818.35 | 2817.94 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2025-02-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 13:15:00 | 2796.45 | 2813.98 | 2816.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 09:15:00 | 2730.05 | 2797.05 | 2807.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 12:15:00 | 2693.85 | 2676.27 | 2704.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-10 13:00:00 | 2693.85 | 2676.27 | 2704.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 10:15:00 | 2721.40 | 2686.25 | 2698.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 11:15:00 | 2727.80 | 2686.25 | 2698.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 11:15:00 | 2726.15 | 2694.23 | 2700.74 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2025-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-11 12:15:00 | 2755.25 | 2706.43 | 2705.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-11 13:15:00 | 2780.00 | 2721.15 | 2712.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-12 09:15:00 | 2712.20 | 2745.79 | 2728.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-12 09:15:00 | 2712.20 | 2745.79 | 2728.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 2712.20 | 2745.79 | 2728.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-12 10:00:00 | 2712.20 | 2745.79 | 2728.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 10:15:00 | 2732.55 | 2743.14 | 2728.85 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2025-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 09:15:00 | 2675.50 | 2721.54 | 2723.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-13 11:15:00 | 2665.10 | 2703.57 | 2714.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 15:15:00 | 2600.00 | 2599.50 | 2625.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-18 09:15:00 | 2575.00 | 2599.50 | 2625.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 09:15:00 | 2530.95 | 2519.40 | 2548.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 09:30:00 | 2533.40 | 2519.40 | 2548.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 11:15:00 | 2500.00 | 2517.39 | 2542.34 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2025-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 14:15:00 | 2570.20 | 2538.76 | 2537.16 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2025-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 10:15:00 | 2523.40 | 2534.70 | 2535.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 14:15:00 | 2496.70 | 2518.53 | 2527.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 12:15:00 | 2517.50 | 2504.22 | 2515.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-25 12:15:00 | 2517.50 | 2504.22 | 2515.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 12:15:00 | 2517.50 | 2504.22 | 2515.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 13:00:00 | 2517.50 | 2504.22 | 2515.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 13:15:00 | 2583.85 | 2520.14 | 2521.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 14:00:00 | 2583.85 | 2520.14 | 2521.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2025-02-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-25 14:15:00 | 2539.80 | 2524.08 | 2523.43 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2025-02-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 15:15:00 | 2522.60 | 2524.17 | 2524.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 2449.35 | 2509.21 | 2517.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 15:15:00 | 2482.85 | 2480.85 | 2496.30 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 09:15:00 | 2448.70 | 2480.85 | 2496.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 2406.35 | 2465.95 | 2488.13 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2025-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 10:15:00 | 2534.25 | 2488.55 | 2486.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 11:15:00 | 2550.50 | 2500.94 | 2491.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-04 13:15:00 | 2498.60 | 2503.04 | 2494.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-04 14:00:00 | 2498.60 | 2503.04 | 2494.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 2512.45 | 2504.93 | 2496.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-04 15:15:00 | 2517.70 | 2504.93 | 2496.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 09:15:00 | 2540.25 | 2579.98 | 2581.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 09:15:00 | 2540.25 | 2579.98 | 2581.90 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2025-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-12 14:15:00 | 2586.90 | 2577.47 | 2576.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 09:15:00 | 2592.75 | 2582.26 | 2578.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 10:15:00 | 2574.05 | 2580.62 | 2578.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 10:15:00 | 2574.05 | 2580.62 | 2578.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 10:15:00 | 2574.05 | 2580.62 | 2578.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 11:00:00 | 2574.05 | 2580.62 | 2578.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 11:15:00 | 2567.80 | 2578.05 | 2577.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-13 12:00:00 | 2567.80 | 2578.05 | 2577.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 12:15:00 | 2576.00 | 2577.64 | 2577.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 13:45:00 | 2586.30 | 2578.03 | 2577.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 14:15:00 | 2580.15 | 2578.03 | 2577.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 15:00:00 | 2580.40 | 2578.51 | 2577.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 09:15:00 | 2590.80 | 2577.82 | 2577.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 2581.45 | 2578.54 | 2577.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:45:00 | 2579.25 | 2578.54 | 2577.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 10:15:00 | 2581.15 | 2579.06 | 2578.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 10:30:00 | 2571.95 | 2579.06 | 2578.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 11:15:00 | 2581.80 | 2579.61 | 2578.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 13:45:00 | 2590.00 | 2580.97 | 2579.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 15:00:00 | 2591.10 | 2583.00 | 2580.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 14:15:00 | 2703.15 | 2749.67 | 2753.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2025-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 14:15:00 | 2703.15 | 2749.67 | 2753.23 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 09:15:00 | 2849.70 | 2764.13 | 2758.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-28 10:15:00 | 2911.90 | 2793.68 | 2772.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 11:15:00 | 2714.75 | 2777.90 | 2767.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 11:15:00 | 2714.75 | 2777.90 | 2767.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 11:15:00 | 2714.75 | 2777.90 | 2767.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-28 12:00:00 | 2714.75 | 2777.90 | 2767.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 2731.75 | 2768.67 | 2764.24 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 13:15:00 | 2729.45 | 2760.82 | 2761.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-01 10:15:00 | 2700.70 | 2739.47 | 2750.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 15:15:00 | 2707.50 | 2698.96 | 2723.09 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-02 09:15:00 | 2649.25 | 2698.96 | 2723.09 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-02 10:30:00 | 2649.10 | 2679.23 | 2709.38 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-02 11:00:00 | 2645.50 | 2679.23 | 2709.38 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 2703.25 | 2676.63 | 2697.78 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-02 14:15:00 | 2703.25 | 2676.63 | 2697.78 | SL hit (close>ema400) qty=1.00 sl=2697.78 alert=retest1 |

### Cycle 72 — BUY (started 2025-04-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-11 12:15:00 | 2436.50 | 2344.39 | 2343.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 13:15:00 | 2464.45 | 2368.40 | 2354.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 09:15:00 | 2534.90 | 2549.78 | 2510.24 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-17 12:30:00 | 2582.20 | 2556.38 | 2523.14 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 10:15:00 | 2581.70 | 2569.16 | 2540.66 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 11:15:00 | 2580.20 | 2570.49 | 2543.85 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 2604.90 | 2633.56 | 2608.62 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-23 09:15:00 | 2604.90 | 2633.56 | 2608.62 | SL hit (close<ema400) qty=1.00 sl=2608.62 alert=retest1 |

### Cycle 73 — SELL (started 2025-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 13:15:00 | 2586.80 | 2606.91 | 2607.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 2529.50 | 2586.15 | 2597.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-25 14:15:00 | 2555.00 | 2548.82 | 2571.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-25 15:00:00 | 2555.00 | 2548.82 | 2571.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 2517.70 | 2532.86 | 2548.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 09:15:00 | 2497.00 | 2533.60 | 2541.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-30 11:15:00 | 2503.20 | 2526.41 | 2536.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 12:15:00 | 2611.00 | 2514.45 | 2516.08 | SL hit (close>static) qty=1.00 sl=2555.40 alert=retest2 |

### Cycle 74 — BUY (started 2025-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 13:15:00 | 2697.60 | 2551.08 | 2532.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-02 14:15:00 | 2745.30 | 2589.93 | 2551.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-05 10:15:00 | 2593.00 | 2608.29 | 2571.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-05 11:00:00 | 2593.00 | 2608.29 | 2571.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 2577.30 | 2602.09 | 2571.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:30:00 | 2550.00 | 2602.09 | 2571.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 2633.30 | 2608.33 | 2577.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-05 14:15:00 | 2675.00 | 2612.35 | 2582.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-06 09:15:00 | 2645.30 | 2620.93 | 2591.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 09:15:00 | 2644.50 | 2618.99 | 2604.76 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-07 11:15:00 | 2648.70 | 2622.43 | 2608.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 2726.90 | 2751.61 | 2710.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-09 11:45:00 | 2757.50 | 2752.23 | 2717.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-14 09:15:00 | 2909.83 | 2864.72 | 2832.24 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2025-05-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 14:15:00 | 2979.00 | 3031.22 | 3036.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 10:15:00 | 2973.00 | 2994.52 | 3009.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 2995.00 | 2976.88 | 2991.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 2995.00 | 2976.88 | 2991.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 2995.00 | 2976.88 | 2991.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:30:00 | 2985.40 | 2976.88 | 2991.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 2983.60 | 2978.23 | 2990.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 11:15:00 | 2961.30 | 2978.23 | 2990.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 09:15:00 | 3159.20 | 2999.23 | 2992.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 3159.20 | 2999.23 | 2992.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-28 12:15:00 | 3179.90 | 3128.95 | 3097.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 15:15:00 | 3133.00 | 3136.47 | 3109.30 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 09:30:00 | 3179.10 | 3142.20 | 3114.37 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 11:00:00 | 3157.60 | 3145.28 | 3118.30 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-29 12:45:00 | 3161.90 | 3151.78 | 3126.07 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 3132.10 | 3145.21 | 3127.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:15:00 | 3122.50 | 3145.21 | 3127.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 3122.50 | 3140.67 | 3126.93 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-29 15:15:00 | 3122.50 | 3140.67 | 3126.93 | SL hit (close<ema400) qty=1.00 sl=3126.93 alert=retest1 |

### Cycle 77 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 3342.80 | 3373.43 | 3373.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 3283.40 | 3355.42 | 3365.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 10:15:00 | 3325.90 | 3321.05 | 3342.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 10:15:00 | 3325.90 | 3321.05 | 3342.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 3325.90 | 3321.05 | 3342.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 11:00:00 | 3325.90 | 3321.05 | 3342.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 3321.70 | 3321.18 | 3340.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 11:45:00 | 3333.60 | 3321.18 | 3340.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 3337.20 | 3324.99 | 3339.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 13:45:00 | 3338.30 | 3324.99 | 3339.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 3313.00 | 3322.60 | 3336.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 3290.90 | 3324.68 | 3336.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 15:15:00 | 3273.40 | 3235.84 | 3231.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2025-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 15:15:00 | 3273.40 | 3235.84 | 3231.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 3356.30 | 3263.00 | 3246.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 09:15:00 | 3355.00 | 3359.90 | 3317.69 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 13:30:00 | 3398.00 | 3368.81 | 3335.59 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 15:00:00 | 3397.30 | 3374.51 | 3341.20 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 3331.70 | 3361.70 | 3343.43 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-26 10:15:00 | 3331.70 | 3361.70 | 3343.43 | SL hit (close<ema400) qty=1.00 sl=3343.43 alert=retest1 |

### Cycle 79 — SELL (started 2025-07-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 14:15:00 | 3388.70 | 3409.06 | 3411.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 13:15:00 | 3371.00 | 3390.14 | 3399.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 12:15:00 | 3319.30 | 3317.53 | 3339.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-07 13:15:00 | 3324.70 | 3317.53 | 3339.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 3335.70 | 3321.17 | 3339.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 14:00:00 | 3335.70 | 3321.17 | 3339.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 14:15:00 | 3345.10 | 3325.95 | 3339.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 15:00:00 | 3345.10 | 3325.95 | 3339.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 3341.10 | 3328.98 | 3339.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 09:15:00 | 3361.50 | 3328.98 | 3339.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 3312.10 | 3327.37 | 3337.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:30:00 | 3320.70 | 3327.37 | 3337.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 3297.30 | 3313.91 | 3327.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 3297.30 | 3313.91 | 3327.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 3326.80 | 3314.41 | 3324.95 | EMA400 retest candle locked (from downside) |

### Cycle 80 — BUY (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 11:15:00 | 3366.50 | 3332.42 | 3331.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 12:15:00 | 3379.70 | 3341.88 | 3336.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 3346.00 | 3352.34 | 3344.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 11:00:00 | 3346.00 | 3352.34 | 3344.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 3327.80 | 3347.43 | 3343.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:00:00 | 3327.80 | 3347.43 | 3343.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 3317.80 | 3341.50 | 3340.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 12:45:00 | 3320.20 | 3341.50 | 3340.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 13:15:00 | 3323.80 | 3337.96 | 3339.38 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2025-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 14:15:00 | 3355.00 | 3341.37 | 3340.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 09:15:00 | 3389.20 | 3353.64 | 3347.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-15 13:15:00 | 3370.90 | 3372.04 | 3363.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-15 14:00:00 | 3370.90 | 3372.04 | 3363.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 3368.90 | 3371.41 | 3364.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:30:00 | 3370.30 | 3371.41 | 3364.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 3382.10 | 3374.93 | 3367.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 10:15:00 | 3415.00 | 3374.93 | 3367.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 12:30:00 | 3408.20 | 3391.76 | 3377.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:15:00 | 3412.80 | 3391.76 | 3377.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 11:15:00 | 3407.80 | 3397.01 | 3386.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 3436.90 | 3435.80 | 3422.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 11:00:00 | 3470.00 | 3442.64 | 3427.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-25 09:15:00 | 3424.20 | 3500.06 | 3501.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 3424.20 | 3500.06 | 3501.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 3396.10 | 3479.27 | 3491.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 3442.10 | 3421.05 | 3451.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 3442.10 | 3421.05 | 3451.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 3442.10 | 3421.05 | 3451.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:45:00 | 3462.80 | 3421.05 | 3451.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 3446.40 | 3430.87 | 3448.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 13:00:00 | 3446.40 | 3430.87 | 3448.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 13:15:00 | 3447.60 | 3434.22 | 3448.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 13:30:00 | 3444.30 | 3434.22 | 3448.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 14:15:00 | 3446.80 | 3436.74 | 3448.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 15:00:00 | 3446.80 | 3436.74 | 3448.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 15:15:00 | 3440.00 | 3437.39 | 3447.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-29 09:15:00 | 3377.80 | 3437.39 | 3447.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 14:15:00 | 3208.91 | 3246.21 | 3300.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-04 09:15:00 | 3040.02 | 3203.34 | 3270.74 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 84 — BUY (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 10:15:00 | 2909.70 | 2871.57 | 2867.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 12:15:00 | 2930.30 | 2889.13 | 2876.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 13:15:00 | 2964.50 | 2973.06 | 2948.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 14:00:00 | 2964.50 | 2973.06 | 2948.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 15:15:00 | 3018.70 | 3026.44 | 3008.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:15:00 | 2964.30 | 3026.44 | 3008.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 2955.70 | 3012.29 | 3003.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 2940.60 | 3012.29 | 3003.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 2962.40 | 3002.31 | 2999.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 10:30:00 | 2956.90 | 3002.31 | 2999.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2025-08-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 11:15:00 | 2956.60 | 2993.17 | 2995.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 2938.60 | 2973.31 | 2985.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 11:15:00 | 2906.50 | 2903.90 | 2929.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 12:00:00 | 2906.50 | 2903.90 | 2929.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 2897.80 | 2897.55 | 2916.48 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 2938.00 | 2923.08 | 2922.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 2946.10 | 2927.68 | 2924.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 15:15:00 | 2927.70 | 2932.74 | 2928.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 15:15:00 | 2927.70 | 2932.74 | 2928.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 2927.70 | 2932.74 | 2928.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 2938.00 | 2932.74 | 2928.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 2938.00 | 2933.79 | 2929.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 11:30:00 | 2944.20 | 2932.82 | 2929.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 13:00:00 | 2944.90 | 2935.23 | 2930.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 09:15:00 | 2903.60 | 2927.05 | 2928.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — SELL (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 09:15:00 | 2903.60 | 2927.05 | 2928.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 2874.30 | 2905.84 | 2916.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 2899.50 | 2887.41 | 2900.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 2899.50 | 2887.41 | 2900.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 2899.50 | 2887.41 | 2900.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:30:00 | 2902.10 | 2887.41 | 2900.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 2876.10 | 2885.15 | 2898.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:30:00 | 2891.40 | 2885.15 | 2898.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 2934.50 | 2892.91 | 2899.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 2934.50 | 2892.91 | 2899.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 2924.50 | 2899.23 | 2901.55 | EMA400 retest candle locked (from downside) |

### Cycle 88 — BUY (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 12:15:00 | 2928.80 | 2907.99 | 2905.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 14:15:00 | 2940.00 | 2922.91 | 2916.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 3027.20 | 3031.38 | 3010.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 14:45:00 | 3026.00 | 3031.38 | 3010.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 3025.00 | 3027.55 | 3017.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 13:30:00 | 3019.20 | 3027.55 | 3017.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 3043.60 | 3044.40 | 3035.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:45:00 | 3033.20 | 3044.40 | 3035.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 3033.90 | 3042.30 | 3035.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:00:00 | 3033.90 | 3042.30 | 3035.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 11:15:00 | 3030.00 | 3039.84 | 3034.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 11:30:00 | 3035.10 | 3039.84 | 3034.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 12:15:00 | 3053.70 | 3042.61 | 3036.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 12:45:00 | 3049.80 | 3042.61 | 3036.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 3038.00 | 3046.85 | 3040.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:30:00 | 3045.90 | 3046.85 | 3040.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 3022.00 | 3041.88 | 3039.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:30:00 | 3024.60 | 3041.88 | 3039.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 3043.20 | 3042.14 | 3039.49 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2025-09-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 11:15:00 | 3028.80 | 3039.31 | 3040.22 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 15:15:00 | 3055.00 | 3041.58 | 3040.81 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 10:15:00 | 3027.10 | 3038.08 | 3039.31 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 11:15:00 | 3053.30 | 3041.12 | 3040.58 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 3026.70 | 3038.77 | 3040.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 11:15:00 | 3023.00 | 3035.62 | 3038.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 3036.00 | 3035.70 | 3038.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 13:00:00 | 3036.00 | 3035.70 | 3038.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 3035.20 | 3035.60 | 3038.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 14:15:00 | 3027.40 | 3035.60 | 3038.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 3046.20 | 3037.72 | 3038.93 | SL hit (close>static) qty=1.00 sl=3038.60 alert=retest2 |

### Cycle 94 — BUY (started 2025-09-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 14:15:00 | 3010.10 | 2993.06 | 2992.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 3078.40 | 3015.48 | 3003.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 3041.00 | 3043.97 | 3028.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 09:15:00 | 3041.00 | 3043.97 | 3028.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 3041.00 | 3043.97 | 3028.93 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 2968.30 | 3018.53 | 3022.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 2954.20 | 3005.66 | 3016.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 12:15:00 | 2977.50 | 2974.01 | 2988.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 13:00:00 | 2977.50 | 2974.01 | 2988.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 2980.00 | 2977.44 | 2987.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 10:15:00 | 2977.90 | 2979.40 | 2986.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 10:45:00 | 2971.50 | 2976.72 | 2984.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 2950.80 | 2976.92 | 2981.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 13:15:00 | 2941.40 | 2931.26 | 2930.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 2941.40 | 2931.26 | 2930.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 14:15:00 | 2950.10 | 2935.03 | 2932.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 09:15:00 | 2925.60 | 2935.54 | 2932.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 09:15:00 | 2925.60 | 2935.54 | 2932.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 2925.60 | 2935.54 | 2932.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 10:00:00 | 2925.60 | 2935.54 | 2932.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 2939.00 | 2936.23 | 2933.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-17 11:30:00 | 2948.00 | 2939.17 | 2935.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-20 09:15:00 | 2912.50 | 2940.64 | 2938.63 | SL hit (close<static) qty=1.00 sl=2918.60 alert=retest2 |

### Cycle 97 — SELL (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 10:15:00 | 2923.70 | 2937.25 | 2937.27 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2025-10-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-21 13:15:00 | 2970.00 | 2937.58 | 2936.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 13:15:00 | 2981.40 | 2956.30 | 2947.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 09:15:00 | 2980.70 | 2981.91 | 2970.29 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-27 10:45:00 | 3015.70 | 2989.33 | 2974.72 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 3010.50 | 3022.66 | 3013.90 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-29 11:15:00 | 3010.50 | 3022.66 | 3013.90 | SL hit (close<ema400) qty=1.00 sl=3013.90 alert=retest1 |

### Cycle 99 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 3025.40 | 3072.58 | 3072.62 | EMA200 below EMA400 |

### Cycle 100 — BUY (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 11:15:00 | 3042.50 | 3032.31 | 3031.86 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2025-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 12:15:00 | 3018.50 | 3029.55 | 3030.65 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2025-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 14:15:00 | 3047.50 | 3034.52 | 3032.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 15:15:00 | 3057.90 | 3039.20 | 3035.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 09:15:00 | 3034.80 | 3038.32 | 3035.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 09:15:00 | 3034.80 | 3038.32 | 3035.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 3034.80 | 3038.32 | 3035.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 09:30:00 | 3020.70 | 3038.32 | 3035.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 3023.90 | 3035.43 | 3034.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:45:00 | 3026.90 | 3035.43 | 3034.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 3040.00 | 3036.35 | 3034.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 11:30:00 | 3025.50 | 3036.35 | 3034.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 3050.50 | 3059.14 | 3051.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 3024.60 | 3059.14 | 3051.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 3051.20 | 3057.55 | 3051.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 12:15:00 | 3060.10 | 3052.83 | 3049.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 10:30:00 | 3089.80 | 3065.42 | 3058.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 09:30:00 | 3069.10 | 3087.92 | 3083.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 10:30:00 | 3058.70 | 3081.51 | 3080.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 11:15:00 | 3058.90 | 3076.99 | 3078.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — SELL (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 11:15:00 | 3058.90 | 3076.99 | 3078.62 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2025-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 15:15:00 | 3099.00 | 3080.86 | 3079.51 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 09:15:00 | 3065.00 | 3077.69 | 3078.19 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 14:15:00 | 3088.20 | 3078.79 | 3078.01 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 3055.20 | 3074.94 | 3076.45 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 11:15:00 | 3085.60 | 3072.87 | 3072.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-24 14:15:00 | 3153.40 | 3093.15 | 3081.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 15:15:00 | 3090.00 | 3092.52 | 3082.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-25 09:15:00 | 3088.00 | 3092.52 | 3082.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 3067.10 | 3087.43 | 3081.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 3067.10 | 3087.43 | 3081.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 3077.10 | 3085.37 | 3080.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 12:15:00 | 3093.80 | 3085.11 | 3081.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 12:15:00 | 3095.10 | 3126.45 | 3128.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — SELL (started 2025-12-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 12:15:00 | 3095.10 | 3126.45 | 3128.37 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 15:15:00 | 3167.30 | 3134.68 | 3131.43 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2025-12-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 11:15:00 | 3107.30 | 3133.76 | 3135.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 12:15:00 | 3097.70 | 3126.55 | 3131.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 09:15:00 | 3111.00 | 3106.12 | 3118.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 10:00:00 | 3111.00 | 3106.12 | 3118.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 3112.80 | 3107.45 | 3118.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 3111.20 | 3107.45 | 3118.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 3110.00 | 3107.96 | 3117.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 11:30:00 | 3114.50 | 3107.96 | 3117.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 14:15:00 | 3142.00 | 3115.36 | 3118.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 14:30:00 | 3142.00 | 3115.36 | 3118.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 15:15:00 | 3123.40 | 3116.97 | 3119.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:15:00 | 3089.00 | 3116.97 | 3119.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 3106.50 | 3114.87 | 3117.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 09:15:00 | 3065.00 | 3099.97 | 3108.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-08 11:45:00 | 3068.20 | 3091.90 | 3102.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 13:15:00 | 3053.50 | 3038.97 | 3037.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 3053.50 | 3038.97 | 3037.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 15:15:00 | 3059.80 | 3044.24 | 3040.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 09:15:00 | 3078.10 | 3084.66 | 3068.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 09:15:00 | 3078.10 | 3084.66 | 3068.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 3078.10 | 3084.66 | 3068.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 12:30:00 | 3103.50 | 3084.54 | 3077.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 13:15:00 | 3105.30 | 3084.54 | 3077.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 13:45:00 | 3101.20 | 3087.63 | 3079.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 14:30:00 | 3100.90 | 3085.75 | 3079.12 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 3076.80 | 3083.96 | 3078.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 09:15:00 | 3051.70 | 3083.96 | 3078.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-18 09:15:00 | 3025.60 | 3072.29 | 3074.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2025-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 09:15:00 | 3025.60 | 3072.29 | 3074.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 10:15:00 | 3014.10 | 3060.65 | 3068.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 11:15:00 | 3023.40 | 3023.00 | 3039.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 12:00:00 | 3023.40 | 3023.00 | 3039.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 3035.00 | 3024.54 | 3036.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 15:00:00 | 3035.00 | 3024.54 | 3036.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 3037.00 | 3027.03 | 3036.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 3055.00 | 3027.03 | 3036.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 3074.50 | 3036.53 | 3039.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 10:00:00 | 3074.50 | 3036.53 | 3039.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 114 — BUY (started 2025-12-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 10:15:00 | 3089.60 | 3047.14 | 3044.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 15:15:00 | 3097.00 | 3073.56 | 3059.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 10:15:00 | 3073.40 | 3076.96 | 3064.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 10:30:00 | 3067.00 | 3076.96 | 3064.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 3072.10 | 3075.99 | 3064.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:30:00 | 3065.00 | 3075.99 | 3064.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 3062.00 | 3073.19 | 3064.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 3062.00 | 3073.19 | 3064.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 3059.80 | 3070.51 | 3064.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:00:00 | 3059.80 | 3070.51 | 3064.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 3036.90 | 3063.79 | 3061.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 15:00:00 | 3036.90 | 3063.79 | 3061.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — SELL (started 2025-12-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 15:15:00 | 3036.80 | 3058.39 | 3059.39 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 15:15:00 | 3074.10 | 3056.74 | 3056.26 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-26 09:15:00 | 3037.60 | 3052.91 | 3054.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 10:15:00 | 3030.20 | 3048.37 | 3052.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 12:15:00 | 2977.30 | 2970.57 | 2989.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-30 12:45:00 | 2972.10 | 2970.57 | 2989.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 2993.10 | 2975.08 | 2990.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 2993.10 | 2975.08 | 2990.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 2948.00 | 2969.66 | 2986.24 | EMA400 retest candle locked (from downside) |

### Cycle 118 — BUY (started 2025-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 15:15:00 | 3009.60 | 2991.25 | 2989.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 11:15:00 | 3030.00 | 3001.63 | 2994.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 3069.40 | 3076.01 | 3056.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 09:15:00 | 3069.40 | 3076.01 | 3056.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 3069.40 | 3076.01 | 3056.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 09:30:00 | 3059.00 | 3076.01 | 3056.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 10:15:00 | 3071.20 | 3075.05 | 3057.85 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2026-01-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 12:15:00 | 3031.00 | 3056.77 | 3057.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 3006.80 | 3034.51 | 3044.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 13:15:00 | 3036.00 | 3031.85 | 3041.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 13:15:00 | 3036.00 | 3031.85 | 3041.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 13:15:00 | 3036.00 | 3031.85 | 3041.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 14:00:00 | 3036.00 | 3031.85 | 3041.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 14:15:00 | 2980.00 | 3021.48 | 3036.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 09:15:00 | 2961.00 | 3011.78 | 3030.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 10:30:00 | 2959.80 | 2990.59 | 3017.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 15:15:00 | 2950.00 | 2946.97 | 2965.85 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-13 11:15:00 | 3023.90 | 2977.84 | 2975.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 11:15:00 | 3023.90 | 2977.84 | 2975.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 14:15:00 | 3036.40 | 3002.31 | 2988.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 09:15:00 | 3005.40 | 3035.45 | 3019.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-16 09:15:00 | 3005.40 | 3035.45 | 3019.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 09:15:00 | 3005.40 | 3035.45 | 3019.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-16 10:00:00 | 3005.40 | 3035.45 | 3019.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 3024.30 | 3033.22 | 3019.72 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2026-01-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 15:15:00 | 2973.00 | 3008.71 | 3012.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 09:15:00 | 2968.00 | 3000.57 | 3008.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-19 14:15:00 | 2984.90 | 2980.06 | 2992.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-19 15:00:00 | 2984.90 | 2980.06 | 2992.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 15:15:00 | 3002.80 | 2984.61 | 2993.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 2948.40 | 2984.61 | 2993.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 10:15:00 | 2946.00 | 2981.85 | 2991.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 13:00:00 | 2966.30 | 2972.19 | 2984.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 15:00:00 | 2970.20 | 2976.02 | 2984.13 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 2989.80 | 2978.77 | 2984.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 09:15:00 | 2946.30 | 2978.77 | 2984.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 10:30:00 | 2960.50 | 2976.72 | 2982.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 11:15:00 | 2952.90 | 2976.72 | 2982.73 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 12:15:00 | 3019.20 | 2988.30 | 2987.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — BUY (started 2026-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 12:15:00 | 3019.20 | 2988.30 | 2987.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-21 13:15:00 | 3029.80 | 2996.60 | 2991.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-22 12:15:00 | 3020.30 | 3022.43 | 3009.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-22 12:45:00 | 3015.60 | 3022.43 | 3009.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 13:15:00 | 3024.80 | 3022.90 | 3011.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-22 14:15:00 | 3039.70 | 3022.90 | 3011.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-23 14:45:00 | 3028.30 | 3015.91 | 3012.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-23 15:15:00 | 3045.00 | 3015.91 | 3012.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-27 10:00:00 | 3032.20 | 3023.82 | 3017.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 3027.10 | 3024.48 | 3017.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:45:00 | 3010.80 | 3024.48 | 3017.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 12:15:00 | 3025.70 | 3025.81 | 3019.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 12:45:00 | 3026.50 | 3025.81 | 3019.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 13:15:00 | 3001.30 | 3020.91 | 3018.07 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-27 13:15:00 | 3001.30 | 3020.91 | 3018.07 | SL hit (close<static) qty=1.00 sl=3002.00 alert=retest2 |

### Cycle 123 — SELL (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 14:15:00 | 2983.50 | 3013.43 | 3014.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 09:15:00 | 2941.90 | 2994.46 | 3005.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 14:15:00 | 2986.80 | 2978.26 | 2991.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 14:15:00 | 2986.80 | 2978.26 | 2991.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 14:15:00 | 2986.80 | 2978.26 | 2991.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 15:00:00 | 2986.80 | 2978.26 | 2991.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 2990.00 | 2980.60 | 2991.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:45:00 | 2977.00 | 2981.47 | 2989.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 11:15:00 | 2966.50 | 2981.47 | 2989.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 11:15:00 | 2975.00 | 2956.65 | 2961.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 3010.90 | 2973.52 | 2968.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 12:15:00 | 3010.90 | 2973.52 | 2968.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 13:15:00 | 3040.50 | 2986.92 | 2975.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 10:15:00 | 3003.10 | 3004.38 | 2988.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-02 11:00:00 | 3003.10 | 3004.38 | 2988.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 3030.80 | 3010.58 | 2996.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 14:30:00 | 2997.00 | 3010.58 | 2996.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 3102.10 | 3213.99 | 3179.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 3102.10 | 3213.99 | 3179.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 3095.30 | 3190.25 | 3172.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:30:00 | 3100.00 | 3190.25 | 3172.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 3158.60 | 3180.03 | 3170.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 13:00:00 | 3158.60 | 3180.03 | 3170.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 13:15:00 | 3148.00 | 3173.62 | 3168.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:00:00 | 3148.00 | 3173.62 | 3168.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 3176.20 | 3174.14 | 3168.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 15:15:00 | 3200.00 | 3174.14 | 3168.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 09:45:00 | 3197.60 | 3179.45 | 3172.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 10:30:00 | 3208.70 | 3189.76 | 3177.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-10 11:15:00 | 3214.50 | 3250.12 | 3238.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 3237.80 | 3239.76 | 3236.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:30:00 | 3225.90 | 3239.76 | 3236.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 3231.90 | 3238.19 | 3236.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 09:30:00 | 3261.60 | 3240.75 | 3237.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 11:15:00 | 3211.10 | 3231.47 | 3233.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 125 — SELL (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 11:15:00 | 3211.10 | 3231.47 | 3233.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 12:15:00 | 3205.60 | 3226.30 | 3231.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 15:15:00 | 3120.00 | 3109.10 | 3142.31 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 09:15:00 | 3038.90 | 3109.10 | 3142.31 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 10:15:00 | 3049.00 | 3098.70 | 3134.56 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-02-16 15:15:00 | 3049.00 | 3052.76 | 3094.06 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 3089.20 | 3059.45 | 3089.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 3089.20 | 3059.45 | 3089.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 3094.50 | 3066.46 | 3090.31 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-02-17 10:15:00 | 3094.50 | 3066.46 | 3090.31 | SL hit (close>ema400) qty=1.00 sl=3090.31 alert=retest1 |

### Cycle 126 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 3166.40 | 3107.82 | 3105.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 15:15:00 | 3180.00 | 3150.70 | 3134.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 3145.40 | 3158.68 | 3146.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 14:15:00 | 3145.40 | 3158.68 | 3146.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 3145.40 | 3158.68 | 3146.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 3145.40 | 3158.68 | 3146.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 3180.00 | 3162.95 | 3149.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 3130.00 | 3162.95 | 3149.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 3152.00 | 3160.76 | 3149.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:30:00 | 3144.20 | 3160.76 | 3149.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 3177.50 | 3164.11 | 3152.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 10:30:00 | 3168.00 | 3164.11 | 3152.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 12:15:00 | 3158.50 | 3164.73 | 3154.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:00:00 | 3158.50 | 3164.73 | 3154.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 3163.80 | 3164.54 | 3155.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 13:30:00 | 3159.70 | 3164.54 | 3155.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 3138.00 | 3159.23 | 3154.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 15:00:00 | 3138.00 | 3159.23 | 3154.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 15:15:00 | 3183.90 | 3164.17 | 3156.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:15:00 | 3136.80 | 3164.17 | 3156.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 3118.90 | 3155.11 | 3153.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 11:00:00 | 3183.00 | 3160.69 | 3156.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 11:30:00 | 3183.30 | 3165.51 | 3158.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-04 10:15:00 | 3313.70 | 3361.77 | 3362.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — SELL (started 2026-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 10:15:00 | 3313.70 | 3361.77 | 3362.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 3259.00 | 3341.22 | 3353.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 10:15:00 | 3283.30 | 3275.71 | 3308.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 11:00:00 | 3283.30 | 3275.71 | 3308.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 3315.00 | 3284.61 | 3307.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 12:45:00 | 3307.10 | 3284.61 | 3307.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 3325.00 | 3292.69 | 3308.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:45:00 | 3324.30 | 3292.69 | 3308.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 3350.00 | 3304.15 | 3312.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 3350.00 | 3304.15 | 3312.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 3378.20 | 3324.70 | 3320.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 11:15:00 | 3388.50 | 3344.69 | 3330.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 3268.20 | 3349.57 | 3341.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 3268.20 | 3349.57 | 3341.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 3268.20 | 3349.57 | 3341.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:00:00 | 3268.20 | 3349.57 | 3341.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 3253.20 | 3330.29 | 3333.57 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 3397.60 | 3332.42 | 3325.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 3405.00 | 3344.96 | 3332.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 10:15:00 | 3439.60 | 3456.80 | 3423.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-13 11:00:00 | 3439.60 | 3456.80 | 3423.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 13:15:00 | 3444.10 | 3450.53 | 3428.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 13:45:00 | 3436.80 | 3450.53 | 3428.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 3543.80 | 3471.18 | 3443.69 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2026-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-17 09:15:00 | 3342.90 | 3419.60 | 3429.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 09:15:00 | 3289.50 | 3360.32 | 3379.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 14:15:00 | 3107.50 | 3097.32 | 3172.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-23 15:00:00 | 3107.50 | 3097.32 | 3172.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 3163.00 | 3113.46 | 3156.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 3163.00 | 3113.46 | 3156.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 3177.60 | 3126.29 | 3158.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 3168.40 | 3126.29 | 3158.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 14:15:00 | 3140.80 | 3135.13 | 3157.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 14:30:00 | 3164.30 | 3135.13 | 3157.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 3150.10 | 3138.12 | 3156.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 3186.50 | 3138.12 | 3156.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 3249.90 | 3160.48 | 3165.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 3249.90 | 3160.48 | 3165.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 3250.50 | 3178.48 | 3172.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 11:15:00 | 3269.90 | 3196.77 | 3181.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 3243.70 | 3262.66 | 3226.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 3243.70 | 3262.66 | 3226.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 3243.70 | 3262.66 | 3226.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 12:00:00 | 3287.70 | 3266.02 | 3234.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 14:45:00 | 3297.90 | 3289.88 | 3254.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 09:15:00 | 3285.20 | 3281.90 | 3253.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 13:30:00 | 3270.20 | 3260.61 | 3251.52 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 3213.00 | 3251.09 | 3248.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 15:00:00 | 3213.00 | 3251.09 | 3248.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-03-30 15:15:00 | 3222.00 | 3245.27 | 3245.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2026-03-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 15:15:00 | 3222.00 | 3245.27 | 3245.66 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2026-04-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 09:15:00 | 3382.00 | 3272.62 | 3258.05 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2026-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 11:15:00 | 3283.00 | 3296.90 | 3297.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-07 15:15:00 | 3260.00 | 3284.55 | 3290.84 | Break + close below crossover candle low |

### Cycle 136 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 3380.50 | 3303.74 | 3298.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 12:15:00 | 3415.00 | 3347.71 | 3322.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 11:15:00 | 3490.70 | 3499.72 | 3463.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-13 12:00:00 | 3490.70 | 3499.72 | 3463.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 12:15:00 | 3475.00 | 3494.78 | 3464.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 12:30:00 | 3474.20 | 3494.78 | 3464.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 13:15:00 | 3486.90 | 3493.20 | 3466.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 14:00:00 | 3486.90 | 3493.20 | 3466.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 14:15:00 | 3475.00 | 3489.56 | 3467.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 14:45:00 | 3470.90 | 3489.56 | 3467.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 3447.50 | 3481.15 | 3465.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 3556.00 | 3481.15 | 3465.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 10:15:00 | 3459.70 | 3497.10 | 3501.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 10:15:00 | 3459.70 | 3497.10 | 3501.54 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2026-04-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 10:15:00 | 3523.50 | 3497.51 | 3497.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 13:15:00 | 3534.70 | 3514.26 | 3505.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 14:15:00 | 3506.00 | 3512.61 | 3505.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 14:15:00 | 3506.00 | 3512.61 | 3505.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 3506.00 | 3512.61 | 3505.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 14:30:00 | 3474.30 | 3512.61 | 3505.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 3499.70 | 3510.03 | 3505.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:15:00 | 3526.60 | 3510.03 | 3505.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 3541.80 | 3516.38 | 3508.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 11:15:00 | 3573.90 | 3523.22 | 3515.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-24 10:00:00 | 3586.50 | 3614.20 | 3592.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 12:15:00 | 3530.00 | 3574.26 | 3577.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 139 — SELL (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 12:15:00 | 3530.00 | 3574.26 | 3577.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 14:15:00 | 3515.10 | 3556.85 | 3568.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 12:15:00 | 3559.90 | 3548.88 | 3559.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 12:15:00 | 3559.90 | 3548.88 | 3559.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 3559.90 | 3548.88 | 3559.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 13:00:00 | 3559.90 | 3548.88 | 3559.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 3551.00 | 3549.31 | 3558.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:30:00 | 3540.00 | 3549.53 | 3557.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 09:15:00 | 3578.20 | 3557.80 | 3560.30 | SL hit (close>static) qty=1.00 sl=3567.00 alert=retest2 |

### Cycle 140 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 3517.10 | 3452.15 | 3448.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 11:15:00 | 3599.70 | 3510.86 | 3483.01 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-27 09:15:00 | 4147.45 | 2024-05-27 09:15:00 | 4050.00 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2024-06-11 11:45:00 | 4092.50 | 2024-06-14 09:15:00 | 4501.75 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-06-12 13:30:00 | 4101.25 | 2024-06-14 09:15:00 | 4511.38 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-07-18 14:45:00 | 3924.00 | 2024-07-24 09:15:00 | 4002.55 | STOP_HIT | 1.00 | -2.00% |
| SELL | retest2 | 2024-07-19 09:45:00 | 3902.25 | 2024-07-24 09:15:00 | 4002.55 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2024-07-19 14:00:00 | 3906.05 | 2024-07-24 09:15:00 | 4002.55 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2024-07-22 10:15:00 | 3919.25 | 2024-07-24 09:15:00 | 4002.55 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2024-07-22 12:45:00 | 3867.00 | 2024-07-24 09:15:00 | 4002.55 | STOP_HIT | 1.00 | -3.51% |
| SELL | retest2 | 2024-07-23 12:15:00 | 3830.75 | 2024-07-24 09:15:00 | 4002.55 | STOP_HIT | 1.00 | -4.48% |
| SELL | retest2 | 2024-07-23 13:15:00 | 3865.00 | 2024-07-24 09:15:00 | 4002.55 | STOP_HIT | 1.00 | -3.56% |
| SELL | retest2 | 2024-07-23 15:15:00 | 3860.00 | 2024-07-24 09:15:00 | 4002.55 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest2 | 2024-07-29 09:15:00 | 4128.25 | 2024-08-05 10:15:00 | 4190.10 | STOP_HIT | 1.00 | 1.50% |
| BUY | retest2 | 2024-08-23 10:15:00 | 3745.55 | 2024-08-29 10:15:00 | 3682.20 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2024-08-26 13:15:00 | 3773.55 | 2024-08-29 10:15:00 | 3682.20 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2024-08-27 13:00:00 | 3743.20 | 2024-08-29 10:15:00 | 3682.20 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-09-03 09:15:00 | 3860.40 | 2024-09-05 14:15:00 | 3794.00 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2024-09-16 09:30:00 | 3786.60 | 2024-09-17 09:15:00 | 3748.10 | STOP_HIT | 1.00 | -1.02% |
| BUY | retest2 | 2024-09-17 09:45:00 | 3780.65 | 2024-09-17 10:15:00 | 3740.05 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2024-09-30 09:15:00 | 3774.75 | 2024-10-07 10:15:00 | 3586.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-09-30 15:00:00 | 3750.05 | 2024-10-07 10:15:00 | 3601.92 | PARTIAL | 0.50 | 3.95% |
| SELL | retest2 | 2024-09-30 09:15:00 | 3774.75 | 2024-10-08 09:15:00 | 3621.40 | STOP_HIT | 0.50 | 4.06% |
| SELL | retest2 | 2024-09-30 15:00:00 | 3750.05 | 2024-10-08 09:15:00 | 3621.40 | STOP_HIT | 0.50 | 3.43% |
| SELL | retest2 | 2024-10-01 14:30:00 | 3791.50 | 2024-10-09 13:15:00 | 3719.25 | STOP_HIT | 1.00 | 1.91% |
| BUY | retest2 | 2024-10-11 15:00:00 | 3730.00 | 2024-10-16 14:15:00 | 3750.00 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2024-10-14 09:45:00 | 3729.95 | 2024-10-16 14:15:00 | 3750.00 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2024-11-04 10:45:00 | 3453.95 | 2024-11-06 09:15:00 | 3330.35 | STOP_HIT | 1.00 | -3.58% |
| SELL | retest2 | 2024-11-14 11:15:00 | 3293.80 | 2024-11-21 10:15:00 | 3298.45 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2024-11-22 09:15:00 | 3283.05 | 2024-11-28 15:15:00 | 3320.00 | STOP_HIT | 1.00 | 1.13% |
| BUY | retest2 | 2024-11-22 10:15:00 | 3280.50 | 2024-11-28 15:15:00 | 3320.00 | STOP_HIT | 1.00 | 1.20% |
| SELL | retest2 | 2024-12-17 11:15:00 | 3153.80 | 2024-12-19 15:15:00 | 3195.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2024-12-18 11:15:00 | 3162.80 | 2024-12-19 15:15:00 | 3195.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-12-18 14:00:00 | 3162.45 | 2024-12-19 15:15:00 | 3195.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-12-18 14:45:00 | 3161.35 | 2024-12-19 15:15:00 | 3195.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-12-23 11:15:00 | 3245.80 | 2024-12-23 13:15:00 | 3198.30 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-01-01 09:45:00 | 3078.00 | 2025-01-06 10:15:00 | 2924.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-01 09:45:00 | 3078.00 | 2025-01-08 09:15:00 | 2939.70 | STOP_HIT | 0.50 | 4.49% |
| SELL | retest1 | 2025-01-28 11:15:00 | 2739.05 | 2025-01-29 09:15:00 | 2813.15 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2025-01-31 10:45:00 | 2853.35 | 2025-02-03 09:15:00 | 2788.10 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-01-31 12:45:00 | 2849.10 | 2025-02-03 09:15:00 | 2788.10 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-03-04 15:15:00 | 2517.70 | 2025-03-11 09:15:00 | 2540.25 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest2 | 2025-03-13 13:45:00 | 2586.30 | 2025-03-27 14:15:00 | 2703.15 | STOP_HIT | 1.00 | 4.52% |
| BUY | retest2 | 2025-03-13 14:15:00 | 2580.15 | 2025-03-27 14:15:00 | 2703.15 | STOP_HIT | 1.00 | 4.77% |
| BUY | retest2 | 2025-03-13 15:00:00 | 2580.40 | 2025-03-27 14:15:00 | 2703.15 | STOP_HIT | 1.00 | 4.76% |
| BUY | retest2 | 2025-03-17 09:15:00 | 2590.80 | 2025-03-27 14:15:00 | 2703.15 | STOP_HIT | 1.00 | 4.34% |
| BUY | retest2 | 2025-03-17 13:45:00 | 2590.00 | 2025-03-27 14:15:00 | 2703.15 | STOP_HIT | 1.00 | 4.37% |
| BUY | retest2 | 2025-03-17 15:00:00 | 2591.10 | 2025-03-27 14:15:00 | 2703.15 | STOP_HIT | 1.00 | 4.32% |
| SELL | retest1 | 2025-04-02 09:15:00 | 2649.25 | 2025-04-02 14:15:00 | 2703.25 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest1 | 2025-04-02 10:30:00 | 2649.10 | 2025-04-02 14:15:00 | 2703.25 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest1 | 2025-04-02 11:00:00 | 2645.50 | 2025-04-02 14:15:00 | 2703.25 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-04-03 09:15:00 | 2686.15 | 2025-04-04 09:15:00 | 2551.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-04-03 09:15:00 | 2686.15 | 2025-04-07 09:15:00 | 2417.54 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest1 | 2025-04-17 12:30:00 | 2582.20 | 2025-04-23 09:15:00 | 2604.90 | STOP_HIT | 1.00 | 0.88% |
| BUY | retest1 | 2025-04-21 10:15:00 | 2581.70 | 2025-04-23 09:15:00 | 2604.90 | STOP_HIT | 1.00 | 0.90% |
| BUY | retest1 | 2025-04-21 11:15:00 | 2580.20 | 2025-04-23 09:15:00 | 2604.90 | STOP_HIT | 1.00 | 0.96% |
| BUY | retest2 | 2025-04-23 14:15:00 | 2612.50 | 2025-04-24 12:15:00 | 2596.50 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-04-23 14:45:00 | 2618.80 | 2025-04-24 12:15:00 | 2596.50 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2025-04-24 09:30:00 | 2616.00 | 2025-04-24 12:15:00 | 2596.50 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-04-24 11:30:00 | 2611.70 | 2025-04-24 12:15:00 | 2596.50 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-04-30 09:15:00 | 2497.00 | 2025-05-02 12:15:00 | 2611.00 | STOP_HIT | 1.00 | -4.57% |
| SELL | retest2 | 2025-04-30 11:15:00 | 2503.20 | 2025-05-02 12:15:00 | 2611.00 | STOP_HIT | 1.00 | -4.31% |
| BUY | retest2 | 2025-05-05 14:15:00 | 2675.00 | 2025-05-14 09:15:00 | 2909.83 | TARGET_HIT | 1.00 | 8.78% |
| BUY | retest2 | 2025-05-06 09:15:00 | 2645.30 | 2025-05-14 09:15:00 | 2908.95 | TARGET_HIT | 1.00 | 9.97% |
| BUY | retest2 | 2025-05-07 09:15:00 | 2644.50 | 2025-05-14 09:15:00 | 2913.57 | TARGET_HIT | 1.00 | 10.17% |
| BUY | retest2 | 2025-05-07 11:15:00 | 2648.70 | 2025-05-14 14:15:00 | 2942.50 | TARGET_HIT | 1.00 | 11.09% |
| BUY | retest2 | 2025-05-09 11:45:00 | 2757.50 | 2025-05-16 09:15:00 | 3033.25 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-05-23 11:15:00 | 2961.30 | 2025-05-26 09:15:00 | 3159.20 | STOP_HIT | 1.00 | -6.68% |
| BUY | retest1 | 2025-05-29 09:30:00 | 3179.10 | 2025-05-29 15:15:00 | 3122.50 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest1 | 2025-05-29 11:00:00 | 3157.60 | 2025-05-29 15:15:00 | 3122.50 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest1 | 2025-05-29 12:45:00 | 3161.90 | 2025-05-29 15:15:00 | 3122.50 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-05-30 13:00:00 | 3120.00 | 2025-06-11 09:15:00 | 3432.00 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-16 09:15:00 | 3290.90 | 2025-06-20 15:15:00 | 3273.40 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest1 | 2025-06-25 13:30:00 | 3398.00 | 2025-06-26 10:15:00 | 3331.70 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest1 | 2025-06-25 15:00:00 | 3397.30 | 2025-06-26 10:15:00 | 3331.70 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2025-06-26 13:45:00 | 3340.00 | 2025-07-02 14:15:00 | 3388.70 | STOP_HIT | 1.00 | 1.46% |
| BUY | retest2 | 2025-06-26 14:15:00 | 3342.00 | 2025-07-02 14:15:00 | 3388.70 | STOP_HIT | 1.00 | 1.40% |
| BUY | retest2 | 2025-06-27 09:15:00 | 3347.20 | 2025-07-02 14:15:00 | 3388.70 | STOP_HIT | 1.00 | 1.24% |
| BUY | retest2 | 2025-06-27 15:15:00 | 3364.00 | 2025-07-02 14:15:00 | 3388.70 | STOP_HIT | 1.00 | 0.73% |
| BUY | retest2 | 2025-06-30 09:15:00 | 3387.10 | 2025-07-02 14:15:00 | 3388.70 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest2 | 2025-07-16 10:15:00 | 3415.00 | 2025-07-25 09:15:00 | 3424.20 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2025-07-16 12:30:00 | 3408.20 | 2025-07-25 09:15:00 | 3424.20 | STOP_HIT | 1.00 | 0.47% |
| BUY | retest2 | 2025-07-16 13:15:00 | 3412.80 | 2025-07-25 09:15:00 | 3424.20 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2025-07-17 11:15:00 | 3407.80 | 2025-07-25 09:15:00 | 3424.20 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2025-07-21 11:00:00 | 3470.00 | 2025-07-25 09:15:00 | 3424.20 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-07-29 09:15:00 | 3377.80 | 2025-08-01 14:15:00 | 3208.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-29 09:15:00 | 3377.80 | 2025-08-04 09:15:00 | 3040.02 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-09-03 11:30:00 | 2944.20 | 2025-09-04 09:15:00 | 2903.60 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-09-03 13:00:00 | 2944.90 | 2025-09-04 09:15:00 | 2903.60 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-09-23 14:15:00 | 3027.40 | 2025-09-23 14:15:00 | 3046.20 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-09-23 15:15:00 | 3020.50 | 2025-09-30 14:15:00 | 3010.10 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2025-09-25 15:15:00 | 3020.10 | 2025-09-30 14:15:00 | 3010.10 | STOP_HIT | 1.00 | 0.33% |
| SELL | retest2 | 2025-10-10 10:15:00 | 2977.90 | 2025-10-16 13:15:00 | 2941.40 | STOP_HIT | 1.00 | 1.23% |
| SELL | retest2 | 2025-10-10 10:45:00 | 2971.50 | 2025-10-16 13:15:00 | 2941.40 | STOP_HIT | 1.00 | 1.01% |
| SELL | retest2 | 2025-10-13 09:15:00 | 2950.80 | 2025-10-16 13:15:00 | 2941.40 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2025-10-17 11:30:00 | 2948.00 | 2025-10-20 09:15:00 | 2912.50 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest1 | 2025-10-27 10:45:00 | 3015.70 | 2025-10-29 11:15:00 | 3010.50 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-10-30 09:15:00 | 3030.70 | 2025-11-04 09:15:00 | 3025.40 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2025-10-30 11:00:00 | 3027.40 | 2025-11-04 09:15:00 | 3025.40 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-11-14 12:15:00 | 3060.10 | 2025-11-19 11:15:00 | 3058.90 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2025-11-17 10:30:00 | 3089.80 | 2025-11-19 11:15:00 | 3058.90 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-11-19 09:30:00 | 3069.10 | 2025-11-19 11:15:00 | 3058.90 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2025-11-19 10:30:00 | 3058.70 | 2025-11-19 11:15:00 | 3058.90 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-11-25 12:15:00 | 3093.80 | 2025-12-01 12:15:00 | 3095.10 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2025-12-08 09:15:00 | 3065.00 | 2025-12-12 13:15:00 | 3053.50 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2025-12-08 11:45:00 | 3068.20 | 2025-12-12 13:15:00 | 3053.50 | STOP_HIT | 1.00 | 0.48% |
| BUY | retest2 | 2025-12-17 12:30:00 | 3103.50 | 2025-12-18 09:15:00 | 3025.60 | STOP_HIT | 1.00 | -2.51% |
| BUY | retest2 | 2025-12-17 13:15:00 | 3105.30 | 2025-12-18 09:15:00 | 3025.60 | STOP_HIT | 1.00 | -2.57% |
| BUY | retest2 | 2025-12-17 13:45:00 | 3101.20 | 2025-12-18 09:15:00 | 3025.60 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-12-17 14:30:00 | 3100.90 | 2025-12-18 09:15:00 | 3025.60 | STOP_HIT | 1.00 | -2.43% |
| SELL | retest2 | 2026-01-09 09:15:00 | 2961.00 | 2026-01-13 11:15:00 | 3023.90 | STOP_HIT | 1.00 | -2.12% |
| SELL | retest2 | 2026-01-09 10:30:00 | 2959.80 | 2026-01-13 11:15:00 | 3023.90 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2026-01-12 15:15:00 | 2950.00 | 2026-01-13 11:15:00 | 3023.90 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2026-01-20 09:15:00 | 2948.40 | 2026-01-21 12:15:00 | 3019.20 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2026-01-20 10:15:00 | 2946.00 | 2026-01-21 12:15:00 | 3019.20 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2026-01-20 13:00:00 | 2966.30 | 2026-01-21 12:15:00 | 3019.20 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-01-20 15:00:00 | 2970.20 | 2026-01-21 12:15:00 | 3019.20 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2026-01-21 09:15:00 | 2946.30 | 2026-01-21 12:15:00 | 3019.20 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2026-01-21 10:30:00 | 2960.50 | 2026-01-21 12:15:00 | 3019.20 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2026-01-21 11:15:00 | 2952.90 | 2026-01-21 12:15:00 | 3019.20 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2026-01-22 14:15:00 | 3039.70 | 2026-01-27 13:15:00 | 3001.30 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2026-01-23 14:45:00 | 3028.30 | 2026-01-27 13:15:00 | 3001.30 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2026-01-23 15:15:00 | 3045.00 | 2026-01-27 13:15:00 | 3001.30 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2026-01-27 10:00:00 | 3032.20 | 2026-01-27 13:15:00 | 3001.30 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2026-01-29 10:45:00 | 2977.00 | 2026-02-01 12:15:00 | 3010.90 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-01-29 11:15:00 | 2966.50 | 2026-02-01 12:15:00 | 3010.90 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2026-02-01 11:15:00 | 2975.00 | 2026-02-01 12:15:00 | 3010.90 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2026-02-05 15:15:00 | 3200.00 | 2026-02-11 11:15:00 | 3211.10 | STOP_HIT | 1.00 | 0.35% |
| BUY | retest2 | 2026-02-06 09:45:00 | 3197.60 | 2026-02-11 11:15:00 | 3211.10 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2026-02-06 10:30:00 | 3208.70 | 2026-02-11 11:15:00 | 3211.10 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2026-02-10 11:15:00 | 3214.50 | 2026-02-11 11:15:00 | 3211.10 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2026-02-11 09:30:00 | 3261.60 | 2026-02-11 11:15:00 | 3211.10 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest1 | 2026-02-16 09:15:00 | 3038.90 | 2026-02-17 10:15:00 | 3094.50 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest1 | 2026-02-16 10:15:00 | 3049.00 | 2026-02-17 10:15:00 | 3094.50 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest1 | 2026-02-16 15:15:00 | 3049.00 | 2026-02-17 10:15:00 | 3094.50 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2026-02-23 11:00:00 | 3183.00 | 2026-03-04 10:15:00 | 3313.70 | STOP_HIT | 1.00 | 4.11% |
| BUY | retest2 | 2026-02-23 11:30:00 | 3183.30 | 2026-03-04 10:15:00 | 3313.70 | STOP_HIT | 1.00 | 4.10% |
| BUY | retest2 | 2026-03-27 12:00:00 | 3287.70 | 2026-03-30 15:15:00 | 3222.00 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2026-03-27 14:45:00 | 3297.90 | 2026-03-30 15:15:00 | 3222.00 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2026-03-30 09:15:00 | 3285.20 | 2026-03-30 15:15:00 | 3222.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2026-03-30 13:30:00 | 3270.20 | 2026-03-30 15:15:00 | 3222.00 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2026-04-15 09:15:00 | 3556.00 | 2026-04-17 10:15:00 | 3459.70 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2026-04-22 11:15:00 | 3573.90 | 2026-04-24 12:15:00 | 3530.00 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2026-04-24 10:00:00 | 3586.50 | 2026-04-24 12:15:00 | 3530.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2026-04-27 14:30:00 | 3540.00 | 2026-04-28 09:15:00 | 3578.20 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-04-28 10:30:00 | 3537.00 | 2026-05-07 09:15:00 | 3517.10 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest2 | 2026-04-28 11:15:00 | 3536.70 | 2026-05-07 09:15:00 | 3517.10 | STOP_HIT | 1.00 | 0.55% |
| SELL | retest2 | 2026-04-28 12:15:00 | 3545.50 | 2026-05-07 09:15:00 | 3517.10 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest2 | 2026-04-29 10:30:00 | 3528.20 | 2026-05-07 09:15:00 | 3517.10 | STOP_HIT | 1.00 | 0.31% |
