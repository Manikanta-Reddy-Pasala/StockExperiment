# J.K. Cement Ltd. (JKCEMENT)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 5555.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 151 |
| ALERT1 | 102 |
| ALERT2 | 101 |
| ALERT2_SKIP | 55 |
| ALERT3 | 279 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 8 |
| ENTRY2 | 104 |
| PARTIAL | 13 |
| TARGET_HIT | 1 |
| STOP_HIT | 111 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 125 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 55 / 70
- **Target hits / Stop hits / Partials:** 1 / 111 / 13
- **Avg / median % per leg:** 0.28% / -0.18%
- **Sum % (uncompounded):** 34.80%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 69 | 24 | 34.8% | 1 | 68 | 0 | -0.45% | -31.2% |
| BUY @ 2nd Alert (retest1) | 5 | 1 | 20.0% | 0 | 5 | 0 | -0.73% | -3.6% |
| BUY @ 3rd Alert (retest2) | 64 | 23 | 35.9% | 1 | 63 | 0 | -0.43% | -27.6% |
| SELL (all) | 56 | 31 | 55.4% | 0 | 43 | 13 | 1.18% | 66.0% |
| SELL @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 3 | 0 | 0.47% | 1.4% |
| SELL @ 3rd Alert (retest2) | 53 | 28 | 52.8% | 0 | 40 | 13 | 1.22% | 64.6% |
| retest1 (combined) | 8 | 4 | 50.0% | 0 | 8 | 0 | -0.28% | -2.2% |
| retest2 (combined) | 117 | 51 | 43.6% | 1 | 103 | 13 | 0.32% | 37.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 13:15:00 | 3927.90 | 3886.48 | 3884.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-13 15:15:00 | 3930.20 | 3901.70 | 3892.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-14 09:15:00 | 3882.50 | 3897.86 | 3891.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-14 09:15:00 | 3882.50 | 3897.86 | 3891.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 09:15:00 | 3882.50 | 3897.86 | 3891.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-14 09:30:00 | 3875.20 | 3897.86 | 3891.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-14 10:15:00 | 3886.95 | 3895.68 | 3891.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-14 12:45:00 | 3910.80 | 3897.83 | 3892.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-15 09:15:00 | 3850.50 | 3897.82 | 3895.69 | SL hit (close<static) qty=1.00 sl=3880.85 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 10:15:00 | 3862.45 | 3890.75 | 3892.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 13:15:00 | 3813.90 | 3862.33 | 3875.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 15:15:00 | 3864.00 | 3860.05 | 3872.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-17 09:15:00 | 3842.05 | 3860.05 | 3872.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 09:15:00 | 3832.80 | 3854.60 | 3868.59 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2024-05-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 09:15:00 | 3898.00 | 3872.31 | 3870.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 10:15:00 | 3965.00 | 3911.79 | 3895.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 09:15:00 | 3889.80 | 3938.86 | 3920.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 09:15:00 | 3889.80 | 3938.86 | 3920.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 3889.80 | 3938.86 | 3920.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:00:00 | 3889.80 | 3938.86 | 3920.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 10:15:00 | 3879.90 | 3927.06 | 3916.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:30:00 | 3865.25 | 3927.06 | 3916.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 15:15:00 | 3999.85 | 3993.92 | 3979.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 09:15:00 | 4043.40 | 3993.92 | 3979.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 09:15:00 | 3968.95 | 3989.53 | 3986.22 | SL hit (close<static) qty=1.00 sl=3970.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 10:15:00 | 3959.95 | 3983.61 | 3983.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 12:15:00 | 3945.05 | 3971.09 | 3977.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 15:15:00 | 3924.90 | 3917.49 | 3937.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-31 09:15:00 | 3924.35 | 3917.49 | 3937.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 09:15:00 | 3901.80 | 3914.35 | 3934.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 14:30:00 | 3868.80 | 3894.24 | 3916.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 4051.00 | 3922.73 | 3925.20 | SL hit (close>static) qty=1.00 sl=3935.55 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 10:15:00 | 4026.60 | 3943.50 | 3934.42 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 3802.85 | 3933.70 | 3946.03 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 11:15:00 | 3975.00 | 3938.98 | 3937.56 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2024-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-05 15:15:00 | 3894.80 | 3937.03 | 3938.15 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 10:15:00 | 3983.65 | 3944.53 | 3941.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 13:15:00 | 4002.35 | 3962.70 | 3950.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 13:15:00 | 4225.05 | 4237.57 | 4187.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-11 14:00:00 | 4225.05 | 4237.57 | 4187.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 4200.00 | 4222.85 | 4192.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 14:30:00 | 4239.45 | 4219.86 | 4201.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 09:15:00 | 4263.00 | 4222.88 | 4204.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 09:15:00 | 4280.00 | 4242.12 | 4226.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 11:15:00 | 4308.20 | 4326.86 | 4328.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 11:15:00 | 4308.20 | 4326.86 | 4328.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 09:15:00 | 4255.55 | 4309.59 | 4319.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 10:15:00 | 4262.00 | 4260.49 | 4284.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-24 11:00:00 | 4262.00 | 4260.49 | 4284.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 4277.05 | 4252.06 | 4268.42 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2024-06-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 12:15:00 | 4310.00 | 4280.34 | 4278.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-25 14:15:00 | 4335.65 | 4296.15 | 4286.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 14:15:00 | 4399.65 | 4402.37 | 4357.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-26 15:00:00 | 4399.65 | 4402.37 | 4357.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 4446.90 | 4477.18 | 4442.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 12:15:00 | 4442.75 | 4477.18 | 4442.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 4444.95 | 4470.74 | 4442.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 12:45:00 | 4435.60 | 4470.74 | 4442.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 13:15:00 | 4435.05 | 4463.60 | 4442.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 14:00:00 | 4435.05 | 4463.60 | 4442.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 14:15:00 | 4386.45 | 4448.17 | 4437.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 15:00:00 | 4386.45 | 4448.17 | 4437.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 15:15:00 | 4395.95 | 4437.73 | 4433.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 09:30:00 | 4414.45 | 4435.63 | 4432.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-01 11:15:00 | 4408.90 | 4427.94 | 4429.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-01 11:15:00 | 4408.90 | 4427.94 | 4429.61 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 13:15:00 | 4492.55 | 4441.98 | 4435.77 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2024-07-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 12:15:00 | 4357.40 | 4421.56 | 4430.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 14:15:00 | 4335.10 | 4394.11 | 4415.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-04 14:15:00 | 4293.85 | 4286.56 | 4323.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-04 14:45:00 | 4281.50 | 4286.56 | 4323.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 12:15:00 | 4233.65 | 4215.56 | 4233.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 13:00:00 | 4233.65 | 4215.56 | 4233.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 4253.85 | 4223.22 | 4235.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 14:00:00 | 4253.85 | 4223.22 | 4235.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 14:15:00 | 4254.90 | 4229.56 | 4237.51 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-10 10:15:00 | 4300.45 | 4252.64 | 4246.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-10 11:15:00 | 4354.65 | 4273.04 | 4256.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-11 14:15:00 | 4338.40 | 4352.08 | 4322.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-11 15:00:00 | 4338.40 | 4352.08 | 4322.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 4292.00 | 4339.72 | 4322.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 10:00:00 | 4292.00 | 4339.72 | 4322.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 4320.95 | 4335.97 | 4322.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 10:30:00 | 4315.95 | 4335.97 | 4322.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 4332.50 | 4335.27 | 4323.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 12:45:00 | 4335.00 | 4334.94 | 4324.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 13:15:00 | 4345.60 | 4334.94 | 4324.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 09:45:00 | 4338.40 | 4340.88 | 4330.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 09:15:00 | 4298.10 | 4365.77 | 4370.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-07-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 09:15:00 | 4298.10 | 4365.77 | 4370.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 10:15:00 | 4269.55 | 4346.53 | 4361.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 4437.35 | 4335.32 | 4344.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 4437.35 | 4335.32 | 4344.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 4437.35 | 4335.32 | 4344.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:45:00 | 4420.00 | 4335.32 | 4344.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2024-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 10:15:00 | 4433.00 | 4354.85 | 4352.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 14:15:00 | 4479.65 | 4414.11 | 4384.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-23 09:15:00 | 4364.45 | 4414.72 | 4390.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 09:15:00 | 4364.45 | 4414.72 | 4390.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 4364.45 | 4414.72 | 4390.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:30:00 | 4363.15 | 4414.72 | 4390.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 4493.70 | 4430.52 | 4399.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-23 15:15:00 | 4500.00 | 4445.71 | 4417.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-25 10:15:00 | 4382.00 | 4424.15 | 4425.76 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 10:15:00 | 4382.00 | 4424.15 | 4425.76 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2024-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 11:15:00 | 4441.50 | 4422.78 | 4420.39 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-07-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-26 14:15:00 | 4380.05 | 4414.56 | 4417.30 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 10:15:00 | 4457.00 | 4423.48 | 4420.55 | EMA200 above EMA400 |

### Cycle 22 — SELL (started 2024-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 13:15:00 | 4387.75 | 4416.92 | 4418.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-29 15:15:00 | 4371.55 | 4404.51 | 4412.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-30 11:15:00 | 4423.90 | 4401.74 | 4408.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 11:15:00 | 4423.90 | 4401.74 | 4408.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 11:15:00 | 4423.90 | 4401.74 | 4408.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 11:45:00 | 4415.10 | 4401.74 | 4408.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 12:15:00 | 4452.00 | 4411.79 | 4412.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 12:30:00 | 4450.80 | 4411.79 | 4412.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2024-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 13:15:00 | 4483.45 | 4426.12 | 4418.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-30 14:15:00 | 4500.40 | 4440.98 | 4426.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-31 11:15:00 | 4466.00 | 4470.60 | 4447.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-31 11:30:00 | 4471.30 | 4470.60 | 4447.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 4436.60 | 4460.91 | 4447.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 13:45:00 | 4451.25 | 4460.91 | 4447.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 4410.45 | 4450.82 | 4443.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:45:00 | 4421.00 | 4450.82 | 4443.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 4437.00 | 4448.05 | 4443.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 09:15:00 | 4390.85 | 4448.05 | 4443.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 4410.95 | 4440.63 | 4440.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 10:15:00 | 4370.00 | 4440.63 | 4440.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — SELL (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 10:15:00 | 4370.00 | 4426.51 | 4433.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 14:15:00 | 4298.25 | 4367.09 | 4394.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 4252.25 | 4243.62 | 4294.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 09:15:00 | 4285.70 | 4251.77 | 4274.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 4285.70 | 4251.77 | 4274.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 09:30:00 | 4276.35 | 4251.77 | 4274.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 4316.50 | 4264.71 | 4278.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:30:00 | 4318.45 | 4264.71 | 4278.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 14:15:00 | 4294.40 | 4285.97 | 4285.74 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 09:15:00 | 4234.00 | 4276.41 | 4281.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 11:15:00 | 4189.00 | 4254.18 | 4270.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 09:15:00 | 4263.60 | 4242.02 | 4256.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 09:15:00 | 4263.60 | 4242.02 | 4256.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 4263.60 | 4242.02 | 4256.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 09:45:00 | 4265.70 | 4242.02 | 4256.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 10:15:00 | 4247.85 | 4243.18 | 4255.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 10:45:00 | 4266.40 | 4243.18 | 4255.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 11:15:00 | 4242.00 | 4242.95 | 4254.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 11:45:00 | 4248.60 | 4242.95 | 4254.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 09:15:00 | 4201.75 | 4234.17 | 4245.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 12:45:00 | 4184.25 | 4210.18 | 4225.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-13 13:45:00 | 4186.00 | 4201.86 | 4220.00 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 15:15:00 | 4225.05 | 4179.47 | 4176.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2024-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 15:15:00 | 4225.05 | 4179.47 | 4176.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 10:15:00 | 4274.70 | 4206.05 | 4189.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 10:15:00 | 4279.45 | 4281.54 | 4244.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 10:45:00 | 4278.40 | 4281.54 | 4244.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 09:15:00 | 4265.10 | 4287.53 | 4264.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 10:00:00 | 4265.10 | 4287.53 | 4264.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-21 10:15:00 | 4292.90 | 4288.60 | 4267.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-21 10:30:00 | 4268.90 | 4288.60 | 4267.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 11:15:00 | 4368.60 | 4307.14 | 4287.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 13:45:00 | 4430.00 | 4336.70 | 4304.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 11:00:00 | 4398.35 | 4368.79 | 4332.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 15:00:00 | 4399.85 | 4384.62 | 4352.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 10:15:00 | 4413.30 | 4437.14 | 4440.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-08-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-29 10:15:00 | 4413.30 | 4437.14 | 4440.33 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-08-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-30 11:15:00 | 4458.05 | 4438.08 | 4435.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-02 09:15:00 | 4498.70 | 4457.39 | 4446.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 15:15:00 | 4454.05 | 4478.96 | 4465.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-02 15:15:00 | 4454.05 | 4478.96 | 4465.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 15:15:00 | 4454.05 | 4478.96 | 4465.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 09:15:00 | 4491.30 | 4478.96 | 4465.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-11 12:15:00 | 4696.95 | 4729.76 | 4733.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-09-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 12:15:00 | 4696.95 | 4729.76 | 4733.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 13:15:00 | 4625.50 | 4708.91 | 4723.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 12:15:00 | 4676.25 | 4656.03 | 4683.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 12:15:00 | 4676.25 | 4656.03 | 4683.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 12:15:00 | 4676.25 | 4656.03 | 4683.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 13:00:00 | 4676.25 | 4656.03 | 4683.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 13:15:00 | 4687.00 | 4662.22 | 4683.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-12 14:00:00 | 4687.00 | 4662.22 | 4683.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 14:15:00 | 4687.95 | 4667.37 | 4683.83 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2024-09-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 10:15:00 | 4771.65 | 4702.47 | 4696.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 12:15:00 | 4788.00 | 4729.30 | 4710.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 12:15:00 | 4749.75 | 4759.50 | 4739.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 13:00:00 | 4749.75 | 4759.50 | 4739.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 10:15:00 | 4752.00 | 4769.84 | 4761.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 11:00:00 | 4752.00 | 4769.84 | 4761.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 4714.55 | 4758.78 | 4757.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 12:00:00 | 4714.55 | 4758.78 | 4757.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2024-09-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 12:15:00 | 4645.10 | 4736.05 | 4747.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 13:15:00 | 4620.50 | 4712.94 | 4735.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-20 09:15:00 | 4703.35 | 4649.67 | 4675.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-20 09:15:00 | 4703.35 | 4649.67 | 4675.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 4703.35 | 4649.67 | 4675.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 13:30:00 | 4647.55 | 4671.71 | 4680.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 10:15:00 | 4763.65 | 4689.75 | 4684.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-23 10:15:00 | 4763.65 | 4689.75 | 4684.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-23 13:15:00 | 4793.00 | 4724.77 | 4703.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-24 14:15:00 | 4748.95 | 4764.12 | 4741.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-24 14:15:00 | 4748.95 | 4764.12 | 4741.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 4748.95 | 4764.12 | 4741.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 14:30:00 | 4758.00 | 4764.12 | 4741.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 4750.00 | 4761.30 | 4741.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-25 09:15:00 | 4764.80 | 4761.30 | 4741.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 10:15:00 | 4714.60 | 4749.45 | 4739.73 | SL hit (close<static) qty=1.00 sl=4732.50 alert=retest2 |

### Cycle 34 — SELL (started 2024-09-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 12:15:00 | 4685.95 | 4727.41 | 4730.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 14:15:00 | 4648.20 | 4704.64 | 4719.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-27 09:15:00 | 4638.60 | 4612.06 | 4648.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-27 09:15:00 | 4638.60 | 4612.06 | 4648.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 4638.60 | 4612.06 | 4648.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 10:00:00 | 4638.60 | 4612.06 | 4648.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 10:15:00 | 4623.95 | 4614.44 | 4646.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 14:15:00 | 4603.75 | 4627.68 | 4635.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 09:15:00 | 4570.10 | 4624.35 | 4632.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 13:15:00 | 4373.56 | 4475.82 | 4531.17 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-08 09:15:00 | 4341.60 | 4429.24 | 4493.43 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-08 13:15:00 | 4423.80 | 4418.60 | 4467.18 | SL hit (close>ema200) qty=0.50 sl=4418.60 alert=retest2 |

### Cycle 35 — BUY (started 2024-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 14:15:00 | 4317.25 | 4293.12 | 4291.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 15:15:00 | 4330.90 | 4300.68 | 4295.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 11:15:00 | 4304.30 | 4309.29 | 4301.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 11:15:00 | 4304.30 | 4309.29 | 4301.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 4304.30 | 4309.29 | 4301.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:45:00 | 4295.00 | 4309.29 | 4301.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 12:15:00 | 4326.10 | 4312.65 | 4303.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-16 14:30:00 | 4349.95 | 4322.14 | 4309.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 11:15:00 | 4278.10 | 4314.73 | 4310.34 | SL hit (close<static) qty=1.00 sl=4302.50 alert=retest2 |

### Cycle 36 — SELL (started 2024-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 12:15:00 | 4267.25 | 4305.24 | 4306.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 4243.00 | 4283.46 | 4294.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 4292.70 | 4273.39 | 4285.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 13:15:00 | 4292.70 | 4273.39 | 4285.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 4292.70 | 4273.39 | 4285.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:45:00 | 4292.05 | 4273.39 | 4285.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 4302.00 | 4279.11 | 4286.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 14:45:00 | 4317.90 | 4279.11 | 4286.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 15:15:00 | 4296.30 | 4282.55 | 4287.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:15:00 | 4260.95 | 4282.55 | 4287.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 4233.45 | 4272.73 | 4282.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:00:00 | 4209.00 | 4242.75 | 4263.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:45:00 | 4211.55 | 4236.54 | 4259.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 09:45:00 | 4212.10 | 4227.35 | 4250.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 12:15:00 | 4190.00 | 4219.19 | 4242.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 4253.20 | 4219.75 | 4228.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:00:00 | 4253.20 | 4219.75 | 4228.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 4250.00 | 4225.80 | 4230.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 14:00:00 | 4250.00 | 4225.80 | 4230.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 15:15:00 | 4212.75 | 4223.53 | 4228.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 09:15:00 | 4212.95 | 4223.53 | 4228.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 09:15:00 | 4246.05 | 4228.04 | 4230.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-24 09:45:00 | 4246.35 | 4228.04 | 4230.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 10:15:00 | 4236.65 | 4229.76 | 4231.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 11:30:00 | 4228.60 | 4228.49 | 4230.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:15:00 | 4206.90 | 4221.30 | 4225.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 09:15:00 | 3998.55 | 4152.15 | 4181.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 09:15:00 | 4000.97 | 4152.15 | 4181.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 09:15:00 | 4001.50 | 4152.15 | 4181.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 09:15:00 | 4017.17 | 4152.15 | 4181.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-28 09:15:00 | 3996.55 | 4152.15 | 4181.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-28 10:15:00 | 4190.35 | 4159.79 | 4182.37 | SL hit (close>ema200) qty=0.50 sl=4159.79 alert=retest2 |

### Cycle 37 — BUY (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 13:15:00 | 4259.90 | 4199.10 | 4196.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 09:15:00 | 4283.00 | 4230.96 | 4212.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 09:15:00 | 4294.95 | 4316.41 | 4293.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 09:15:00 | 4294.95 | 4316.41 | 4293.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 4294.95 | 4316.41 | 4293.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:30:00 | 4287.50 | 4316.41 | 4293.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 4279.35 | 4309.00 | 4291.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 11:00:00 | 4279.35 | 4309.00 | 4291.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 11:15:00 | 4336.45 | 4314.49 | 4295.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-31 13:30:00 | 4351.35 | 4319.54 | 4301.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:00:00 | 4346.95 | 4322.99 | 4307.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-01 18:45:00 | 4346.80 | 4327.76 | 4311.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-04 09:15:00 | 4274.90 | 4317.18 | 4307.88 | SL hit (close<static) qty=1.00 sl=4277.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 4271.55 | 4298.88 | 4300.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 09:15:00 | 4225.05 | 4272.48 | 4286.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-06 13:15:00 | 4160.00 | 4156.95 | 4193.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-06 13:45:00 | 4156.55 | 4156.95 | 4193.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 4115.50 | 4146.57 | 4179.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-07 09:30:00 | 4157.35 | 4146.57 | 4179.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 4119.85 | 4075.51 | 4087.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:00:00 | 4119.85 | 4075.51 | 4087.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 10:15:00 | 4111.95 | 4082.80 | 4090.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-12 10:30:00 | 4120.20 | 4082.80 | 4090.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 3986.80 | 3956.33 | 3979.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:00:00 | 3986.80 | 3956.33 | 3979.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 3999.00 | 3964.86 | 3981.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 10:30:00 | 3993.00 | 3964.86 | 3981.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 3999.90 | 3971.87 | 3983.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:30:00 | 4007.35 | 3971.87 | 3983.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 3990.75 | 3975.73 | 3981.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:00:00 | 3990.75 | 3975.73 | 3981.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 3999.80 | 3980.55 | 3983.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 11:15:00 | 4003.00 | 3980.55 | 3983.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 11:15:00 | 3999.95 | 3984.43 | 3984.96 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2024-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 12:15:00 | 4005.15 | 3988.57 | 3986.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 09:15:00 | 4045.35 | 4005.14 | 3995.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 4169.95 | 4185.43 | 4133.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-26 10:00:00 | 4169.95 | 4185.43 | 4133.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 4155.95 | 4177.66 | 4154.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 10:00:00 | 4155.95 | 4177.66 | 4154.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 4115.00 | 4165.13 | 4151.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:00:00 | 4115.00 | 4165.13 | 4151.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 4116.25 | 4155.35 | 4147.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:45:00 | 4117.00 | 4155.35 | 4147.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 4171.30 | 4159.75 | 4151.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 13:30:00 | 4175.00 | 4159.75 | 4151.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 4171.90 | 4170.98 | 4159.46 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 12:15:00 | 4120.55 | 4150.22 | 4151.90 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-11-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 09:15:00 | 4212.35 | 4154.78 | 4152.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 10:15:00 | 4240.00 | 4171.83 | 4160.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 15:15:00 | 4610.00 | 4612.53 | 4542.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 09:15:00 | 4624.55 | 4612.53 | 4542.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 4591.55 | 4611.86 | 4591.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 15:00:00 | 4591.55 | 4611.86 | 4591.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 15:15:00 | 4582.20 | 4605.92 | 4590.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:15:00 | 4565.00 | 4605.92 | 4590.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 4551.00 | 4594.94 | 4587.19 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2024-12-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 11:15:00 | 4561.00 | 4582.50 | 4582.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 13:15:00 | 4540.10 | 4566.58 | 4572.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-10 14:15:00 | 4599.05 | 4573.07 | 4575.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 14:15:00 | 4599.05 | 4573.07 | 4575.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 14:15:00 | 4599.05 | 4573.07 | 4575.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-10 15:00:00 | 4599.05 | 4573.07 | 4575.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2024-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 15:15:00 | 4601.95 | 4578.85 | 4577.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 09:15:00 | 4716.60 | 4606.40 | 4590.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 09:15:00 | 4638.30 | 4670.80 | 4640.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 09:15:00 | 4638.30 | 4670.80 | 4640.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 4638.30 | 4670.80 | 4640.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:00:00 | 4638.30 | 4670.80 | 4640.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 4642.40 | 4665.12 | 4641.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:45:00 | 4648.15 | 4665.12 | 4641.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 4639.15 | 4659.93 | 4640.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 12:00:00 | 4639.15 | 4659.93 | 4640.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 4646.95 | 4657.33 | 4641.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 09:30:00 | 4679.45 | 4653.77 | 4644.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 10:45:00 | 4681.00 | 4659.80 | 4647.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 12:45:00 | 4682.00 | 4668.47 | 4653.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-17 14:15:00 | 4666.55 | 4709.94 | 4706.04 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 14:15:00 | 4670.85 | 4702.12 | 4702.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2024-12-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 14:15:00 | 4670.85 | 4702.12 | 4702.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-18 11:15:00 | 4644.15 | 4673.75 | 4687.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 09:15:00 | 4590.75 | 4587.60 | 4619.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 09:15:00 | 4590.75 | 4587.60 | 4619.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 4590.75 | 4587.60 | 4619.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 15:00:00 | 4509.40 | 4577.41 | 4603.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 11:45:00 | 4565.70 | 4560.09 | 4584.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-23 13:45:00 | 4565.00 | 4566.38 | 4583.77 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-24 12:15:00 | 4610.55 | 4592.05 | 4590.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 12:15:00 | 4610.55 | 4592.05 | 4590.62 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2024-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 14:15:00 | 4563.85 | 4585.31 | 4587.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-26 09:15:00 | 4551.50 | 4575.11 | 4582.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 11:15:00 | 4580.00 | 4573.52 | 4580.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 11:15:00 | 4580.00 | 4573.52 | 4580.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 11:15:00 | 4580.00 | 4573.52 | 4580.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 12:00:00 | 4580.00 | 4573.52 | 4580.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 12:15:00 | 4586.30 | 4576.08 | 4580.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 12:30:00 | 4584.00 | 4576.08 | 4580.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 13:15:00 | 4581.25 | 4577.11 | 4580.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 13:30:00 | 4585.65 | 4577.11 | 4580.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 4583.95 | 4578.48 | 4581.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 4583.95 | 4578.48 | 4581.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 4600.00 | 4582.78 | 4582.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 4589.15 | 4582.78 | 4582.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — BUY (started 2024-12-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 09:15:00 | 4605.30 | 4587.29 | 4584.92 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-12-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-27 15:15:00 | 4566.25 | 4583.07 | 4584.61 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2024-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 10:15:00 | 4638.45 | 4595.57 | 4590.11 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 09:15:00 | 4552.10 | 4584.05 | 4587.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-31 11:15:00 | 4536.00 | 4573.63 | 4582.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 13:15:00 | 4611.85 | 4575.81 | 4581.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 13:15:00 | 4611.85 | 4575.81 | 4581.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 13:15:00 | 4611.85 | 4575.81 | 4581.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-31 14:00:00 | 4611.85 | 4575.81 | 4581.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 14:15:00 | 4601.35 | 4580.92 | 4583.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-31 15:15:00 | 4589.80 | 4580.92 | 4583.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-01 13:15:00 | 4591.20 | 4583.21 | 4583.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 13:15:00 | 4591.20 | 4583.21 | 4583.01 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 14:15:00 | 4580.60 | 4582.69 | 4582.79 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-01-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 15:15:00 | 4596.20 | 4585.39 | 4584.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 09:15:00 | 4658.00 | 4599.91 | 4590.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 12:15:00 | 4712.00 | 4717.30 | 4677.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 13:00:00 | 4712.00 | 4717.30 | 4677.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 4688.80 | 4717.92 | 4693.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 4688.80 | 4717.92 | 4693.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 4690.00 | 4712.34 | 4693.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 4674.95 | 4712.34 | 4693.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 12:15:00 | 4702.00 | 4710.27 | 4694.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 13:30:00 | 4712.00 | 4708.21 | 4694.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 15:15:00 | 4728.00 | 4706.37 | 4695.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-09 15:15:00 | 4720.10 | 4748.80 | 4749.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2025-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-09 15:15:00 | 4720.10 | 4748.80 | 4749.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 4570.00 | 4713.04 | 4733.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-10 15:15:00 | 4609.45 | 4607.64 | 4661.35 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 09:15:00 | 4515.70 | 4607.64 | 4661.35 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-01-13 11:15:00 | 4472.05 | 4572.50 | 4634.93 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 4464.90 | 4415.35 | 4462.52 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-01-15 10:15:00 | 4464.90 | 4415.35 | 4462.52 | SL hit (close>ema400) qty=1.00 sl=4462.52 alert=retest1 |

### Cycle 55 — BUY (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 10:15:00 | 4521.00 | 4478.95 | 4476.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 13:15:00 | 4559.00 | 4516.90 | 4496.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-17 10:15:00 | 4463.15 | 4517.75 | 4504.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 10:15:00 | 4463.15 | 4517.75 | 4504.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 4463.15 | 4517.75 | 4504.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:00:00 | 4463.15 | 4517.75 | 4504.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 4443.35 | 4502.87 | 4499.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 11:45:00 | 4455.25 | 4502.87 | 4499.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 4487.75 | 4520.01 | 4511.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 09:30:00 | 4480.05 | 4520.01 | 4511.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 4518.35 | 4519.67 | 4512.12 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 15:15:00 | 4502.35 | 4507.62 | 4508.21 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 10:15:00 | 4537.95 | 4514.07 | 4511.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-21 11:15:00 | 4605.65 | 4532.39 | 4519.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-22 09:15:00 | 4558.40 | 4568.89 | 4546.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 09:15:00 | 4558.40 | 4568.89 | 4546.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 4558.40 | 4568.89 | 4546.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:45:00 | 4574.20 | 4568.89 | 4546.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 4561.00 | 4567.31 | 4547.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 10:45:00 | 4545.70 | 4567.31 | 4547.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 11:15:00 | 4518.95 | 4557.64 | 4544.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:00:00 | 4518.95 | 4557.64 | 4544.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 12:15:00 | 4546.00 | 4555.31 | 4545.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:45:00 | 4506.80 | 4555.31 | 4545.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 13:15:00 | 4565.35 | 4557.32 | 4546.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 13:30:00 | 4515.00 | 4557.32 | 4546.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 09:15:00 | 4782.10 | 4764.15 | 4713.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-27 11:45:00 | 4849.75 | 4781.32 | 4730.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-28 14:45:00 | 4848.80 | 4784.76 | 4761.03 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-28 15:15:00 | 4848.00 | 4784.76 | 4761.03 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-29 09:30:00 | 4893.60 | 4826.24 | 4784.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 4846.95 | 4872.87 | 4840.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 11:00:00 | 4846.95 | 4872.87 | 4840.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 11:15:00 | 4814.70 | 4861.24 | 4837.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 12:00:00 | 4814.70 | 4861.24 | 4837.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 12:15:00 | 4756.15 | 4840.22 | 4830.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 13:00:00 | 4756.15 | 4840.22 | 4830.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-30 14:15:00 | 4782.65 | 4817.18 | 4821.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-30 14:15:00 | 4782.65 | 4817.18 | 4821.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-31 11:15:00 | 4762.00 | 4802.56 | 4813.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-31 13:15:00 | 4819.70 | 4802.50 | 4811.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-31 13:15:00 | 4819.70 | 4802.50 | 4811.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 13:15:00 | 4819.70 | 4802.50 | 4811.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 14:00:00 | 4819.70 | 4802.50 | 4811.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 14:15:00 | 4830.85 | 4808.17 | 4812.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-31 15:00:00 | 4830.85 | 4808.17 | 4812.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 15:15:00 | 4801.25 | 4806.79 | 4811.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-01 09:15:00 | 4840.00 | 4806.79 | 4811.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 09:15:00 | 4850.00 | 4815.43 | 4815.23 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 12:15:00 | 4686.05 | 4799.97 | 4809.36 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2025-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 11:15:00 | 4825.30 | 4787.52 | 4784.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 14:15:00 | 4871.20 | 4821.74 | 4802.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 4859.45 | 4912.06 | 4877.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 4859.45 | 4912.06 | 4877.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 4859.45 | 4912.06 | 4877.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:00:00 | 4859.45 | 4912.06 | 4877.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 4870.95 | 4903.84 | 4876.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 11:00:00 | 4870.95 | 4903.84 | 4876.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 4846.70 | 4892.41 | 4873.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:00:00 | 4846.70 | 4892.41 | 4873.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 4863.00 | 4886.53 | 4872.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:45:00 | 4831.30 | 4886.53 | 4872.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 14:15:00 | 4895.00 | 4883.50 | 4873.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:30:00 | 4917.25 | 4876.02 | 4872.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 12:15:00 | 4905.95 | 4876.02 | 4872.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 13:30:00 | 4912.95 | 4877.21 | 4874.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 4834.20 | 4870.69 | 4872.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 4834.20 | 4870.69 | 4872.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 10:15:00 | 4830.80 | 4862.71 | 4868.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 13:15:00 | 4809.90 | 4794.71 | 4819.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-11 14:00:00 | 4809.90 | 4794.71 | 4819.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 14:15:00 | 4818.65 | 4799.50 | 4819.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 14:45:00 | 4819.45 | 4799.50 | 4819.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 15:15:00 | 4790.05 | 4797.61 | 4816.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-12 09:15:00 | 4745.25 | 4797.61 | 4816.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 4690.60 | 4776.21 | 4805.12 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-02-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 12:15:00 | 4816.20 | 4798.57 | 4797.35 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2025-02-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-14 09:15:00 | 4718.30 | 4787.54 | 4793.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 10:15:00 | 4666.35 | 4763.30 | 4781.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 14:15:00 | 4638.55 | 4623.72 | 4671.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-17 14:45:00 | 4640.00 | 4623.72 | 4671.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 4545.25 | 4611.00 | 4657.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 10:15:00 | 4532.55 | 4611.00 | 4657.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:00:00 | 4541.20 | 4597.04 | 4646.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-18 11:45:00 | 4516.85 | 4584.78 | 4636.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-20 10:15:00 | 4656.00 | 4612.98 | 4608.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2025-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 10:15:00 | 4656.00 | 4612.98 | 4608.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 13:15:00 | 4665.00 | 4635.32 | 4621.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 4579.55 | 4634.82 | 4625.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 4579.55 | 4634.82 | 4625.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 4579.55 | 4634.82 | 4625.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 4579.55 | 4634.82 | 4625.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 4580.00 | 4623.85 | 4621.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:30:00 | 4585.70 | 4623.85 | 4621.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 11:15:00 | 4580.00 | 4615.08 | 4617.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 10:15:00 | 4545.65 | 4582.44 | 4595.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 13:15:00 | 4577.00 | 4571.30 | 4586.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-25 13:45:00 | 4578.05 | 4571.30 | 4586.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 14:15:00 | 4571.60 | 4571.36 | 4584.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 15:00:00 | 4571.60 | 4571.36 | 4584.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 15:15:00 | 4595.20 | 4576.13 | 4585.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 09:15:00 | 4556.05 | 4576.13 | 4585.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 11:15:00 | 4500.05 | 4439.30 | 4434.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-03-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 11:15:00 | 4500.05 | 4439.30 | 4434.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 10:15:00 | 4513.00 | 4466.08 | 4450.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 09:15:00 | 4448.85 | 4477.59 | 4464.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-07 09:15:00 | 4448.85 | 4477.59 | 4464.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 4448.85 | 4477.59 | 4464.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 10:00:00 | 4448.85 | 4477.59 | 4464.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 10:15:00 | 4455.40 | 4473.15 | 4463.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 10:45:00 | 4440.10 | 4473.15 | 4463.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2025-03-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 13:15:00 | 4444.00 | 4456.83 | 4457.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 14:15:00 | 4426.45 | 4450.75 | 4454.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-12 09:15:00 | 4369.90 | 4359.92 | 4383.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 09:15:00 | 4369.90 | 4359.92 | 4383.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 4369.90 | 4359.92 | 4383.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:45:00 | 4376.15 | 4359.92 | 4383.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 4322.95 | 4345.57 | 4363.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 10:30:00 | 4306.50 | 4335.77 | 4357.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 13:15:00 | 4374.90 | 4322.01 | 4328.04 | SL hit (close>static) qty=1.00 sl=4370.10 alert=retest2 |

### Cycle 69 — BUY (started 2025-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 15:15:00 | 4350.00 | 4333.70 | 4332.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 4493.75 | 4365.71 | 4347.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-24 10:15:00 | 4795.00 | 4797.33 | 4718.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-24 11:00:00 | 4795.00 | 4797.33 | 4718.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 14:15:00 | 4736.20 | 4783.74 | 4737.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 15:00:00 | 4736.20 | 4783.74 | 4737.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 15:15:00 | 4760.00 | 4778.99 | 4739.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 09:30:00 | 4790.10 | 4772.40 | 4740.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 11:45:00 | 4790.60 | 4775.96 | 4747.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 13:30:00 | 4784.95 | 4775.27 | 4752.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-26 11:15:00 | 4720.15 | 4753.15 | 4749.12 | SL hit (close<static) qty=1.00 sl=4726.25 alert=retest2 |

### Cycle 70 — SELL (started 2025-03-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 12:15:00 | 4707.65 | 4744.05 | 4745.35 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 10:15:00 | 4784.45 | 4748.27 | 4745.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 11:15:00 | 4795.00 | 4757.62 | 4749.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-27 15:15:00 | 4653.30 | 4771.56 | 4761.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 15:15:00 | 4653.30 | 4771.56 | 4761.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 15:15:00 | 4653.30 | 4771.56 | 4761.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 09:30:00 | 4943.05 | 4807.22 | 4778.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-28 10:00:00 | 4949.85 | 4807.22 | 4778.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 11:15:00 | 4941.75 | 4926.94 | 4872.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-01 12:30:00 | 4960.20 | 4931.71 | 4884.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 15:15:00 | 4960.10 | 4943.80 | 4902.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:15:00 | 4900.45 | 4943.80 | 4902.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 09:15:00 | 4883.55 | 4931.75 | 4900.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-02 09:30:00 | 4896.10 | 4931.75 | 4900.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 4938.30 | 4933.06 | 4904.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-02 11:15:00 | 4960.10 | 4933.06 | 4904.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 10:00:00 | 4955.00 | 4970.92 | 4941.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 10:45:00 | 4948.40 | 4966.08 | 4955.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 12:30:00 | 4946.15 | 4957.48 | 4952.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 13:15:00 | 4950.00 | 4955.98 | 4952.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 13:45:00 | 4941.25 | 4955.98 | 4952.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 14:15:00 | 4961.90 | 4957.17 | 4953.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 14:45:00 | 4964.25 | 4957.17 | 4953.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 15:15:00 | 4950.00 | 4955.73 | 4953.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-07 09:15:00 | 4848.30 | 4955.73 | 4953.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-07 09:15:00 | 4841.35 | 4932.86 | 4942.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — SELL (started 2025-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 09:15:00 | 4841.35 | 4932.86 | 4942.88 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2025-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 12:15:00 | 4919.75 | 4878.92 | 4875.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 12:15:00 | 4969.10 | 4919.15 | 4899.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 11:15:00 | 5036.10 | 5045.54 | 5002.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 11:45:00 | 5034.60 | 5045.54 | 5002.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 5017.70 | 5044.25 | 5019.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:30:00 | 5039.10 | 5044.25 | 5019.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 5024.90 | 5040.38 | 5019.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 11:30:00 | 5051.70 | 5041.90 | 5022.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-17 12:00:00 | 5048.00 | 5041.90 | 5022.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:30:00 | 5056.00 | 5048.37 | 5033.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 10:15:00 | 5047.90 | 5048.37 | 5033.48 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 10:15:00 | 5049.70 | 5048.64 | 5034.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 11:15:00 | 5065.20 | 5048.64 | 5034.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 12:15:00 | 5056.10 | 5048.69 | 5036.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 09:45:00 | 5114.90 | 5070.09 | 5051.50 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-28 13:15:00 | 5121.50 | 5193.41 | 5196.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-04-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-28 13:15:00 | 5121.50 | 5193.41 | 5196.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 14:15:00 | 5100.00 | 5174.73 | 5187.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 14:15:00 | 5170.80 | 5128.77 | 5150.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-29 14:15:00 | 5170.80 | 5128.77 | 5150.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 5170.80 | 5128.77 | 5150.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 15:00:00 | 5170.80 | 5128.77 | 5150.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 15:15:00 | 5105.00 | 5124.01 | 5146.76 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2025-05-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-02 13:15:00 | 5171.00 | 5141.53 | 5139.86 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-05-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 12:15:00 | 5128.00 | 5141.60 | 5142.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 14:15:00 | 5118.50 | 5134.89 | 5139.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-08 09:15:00 | 5074.00 | 5070.84 | 5093.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 09:15:00 | 5074.00 | 5070.84 | 5093.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 5074.00 | 5070.84 | 5093.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:30:00 | 5091.00 | 5070.84 | 5093.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 5075.50 | 5060.40 | 5077.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-09 09:15:00 | 5002.00 | 5060.40 | 5077.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-09 12:15:00 | 5086.00 | 5048.85 | 5063.42 | SL hit (close>static) qty=1.00 sl=5084.50 alert=retest2 |

### Cycle 77 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 5209.00 | 5098.46 | 5083.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 10:15:00 | 5307.00 | 5217.78 | 5173.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 12:15:00 | 5291.50 | 5293.88 | 5249.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 12:45:00 | 5267.50 | 5293.88 | 5249.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 5234.00 | 5277.84 | 5249.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 15:00:00 | 5234.00 | 5277.84 | 5249.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 5259.00 | 5274.08 | 5250.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:30:00 | 5229.50 | 5257.56 | 5245.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 5204.50 | 5246.95 | 5241.49 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 11:15:00 | 5139.00 | 5225.36 | 5232.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 12:15:00 | 5135.50 | 5207.39 | 5223.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 14:15:00 | 5254.50 | 5213.07 | 5222.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 14:15:00 | 5254.50 | 5213.07 | 5222.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 5254.50 | 5213.07 | 5222.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 15:00:00 | 5254.50 | 5213.07 | 5222.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 15:15:00 | 5208.00 | 5212.05 | 5221.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:15:00 | 5223.00 | 5212.05 | 5221.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 5234.50 | 5216.54 | 5222.77 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2025-05-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 10:15:00 | 5268.50 | 5226.93 | 5226.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-20 09:15:00 | 5287.50 | 5260.61 | 5246.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 11:15:00 | 5232.50 | 5262.17 | 5249.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 11:15:00 | 5232.50 | 5262.17 | 5249.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 5232.50 | 5262.17 | 5249.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:45:00 | 5232.50 | 5262.17 | 5249.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 5198.50 | 5249.44 | 5245.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 13:00:00 | 5198.50 | 5249.44 | 5245.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 5189.00 | 5237.35 | 5239.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 5137.00 | 5217.28 | 5230.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 5157.00 | 5115.92 | 5143.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 5157.00 | 5115.92 | 5143.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 5157.00 | 5115.92 | 5143.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 5157.00 | 5115.92 | 5143.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 5159.50 | 5124.64 | 5144.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:45:00 | 5159.00 | 5124.64 | 5144.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 5224.00 | 5144.51 | 5151.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:00:00 | 5224.00 | 5144.51 | 5151.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 5164.00 | 5148.41 | 5152.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 13:45:00 | 5124.00 | 5139.23 | 5148.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 09:15:00 | 5448.00 | 5192.66 | 5169.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 5448.00 | 5192.66 | 5169.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 10:15:00 | 5674.50 | 5580.41 | 5531.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 10:15:00 | 5805.50 | 5821.71 | 5741.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 11:00:00 | 5805.50 | 5821.71 | 5741.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 5768.50 | 5786.17 | 5761.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 12:00:00 | 5768.50 | 5786.17 | 5761.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 5767.50 | 5782.44 | 5762.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 09:15:00 | 5826.00 | 5773.77 | 5763.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-17 10:15:00 | 5944.00 | 5983.00 | 5985.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 10:15:00 | 5944.00 | 5983.00 | 5985.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 11:15:00 | 5903.00 | 5967.00 | 5977.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 5940.00 | 5930.43 | 5952.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 5940.00 | 5930.43 | 5952.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 5940.00 | 5930.43 | 5952.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 10:30:00 | 5896.50 | 5929.44 | 5949.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:30:00 | 5889.00 | 5917.75 | 5942.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 14:15:00 | 5832.00 | 5763.32 | 5759.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 14:15:00 | 5832.00 | 5763.32 | 5759.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 10:15:00 | 5893.00 | 5806.94 | 5781.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 09:15:00 | 5908.00 | 6002.47 | 5947.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 09:15:00 | 5908.00 | 6002.47 | 5947.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 5908.00 | 6002.47 | 5947.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:45:00 | 5890.50 | 6002.47 | 5947.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 5939.50 | 5989.88 | 5946.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 11:30:00 | 5954.00 | 5989.80 | 5950.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-07-11 09:15:00 | 6549.40 | 6404.14 | 6362.99 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 6455.00 | 6467.80 | 6467.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 14:15:00 | 6425.00 | 6459.24 | 6463.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-18 09:15:00 | 6499.00 | 6457.71 | 6461.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 09:15:00 | 6499.00 | 6457.71 | 6461.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 6499.00 | 6457.71 | 6461.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:15:00 | 6560.50 | 6457.71 | 6461.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 10:15:00 | 6578.50 | 6481.87 | 6472.47 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 12:15:00 | 6428.00 | 6481.31 | 6484.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 15:15:00 | 6404.00 | 6454.22 | 6470.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 09:15:00 | 6486.50 | 6460.68 | 6471.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 6486.50 | 6460.68 | 6471.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 6486.50 | 6460.68 | 6471.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 09:30:00 | 6473.00 | 6460.68 | 6471.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 10:15:00 | 6509.00 | 6470.34 | 6474.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 12:45:00 | 6459.00 | 6477.54 | 6477.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 13:15:00 | 6492.50 | 6480.53 | 6479.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-07-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 13:15:00 | 6492.50 | 6480.53 | 6479.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 09:15:00 | 6533.00 | 6491.77 | 6484.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 11:15:00 | 6585.50 | 6593.09 | 6555.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 12:00:00 | 6585.50 | 6593.09 | 6555.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 6623.00 | 6612.67 | 6580.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:45:00 | 6620.00 | 6612.67 | 6580.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 10:15:00 | 6594.50 | 6609.03 | 6581.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 10:45:00 | 6583.50 | 6609.03 | 6581.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 11:15:00 | 6567.50 | 6600.73 | 6580.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 12:00:00 | 6567.50 | 6600.73 | 6580.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 12:15:00 | 6535.50 | 6587.68 | 6576.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 13:00:00 | 6535.50 | 6587.68 | 6576.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 13:15:00 | 6532.00 | 6576.55 | 6572.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 14:00:00 | 6532.00 | 6576.55 | 6572.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2025-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 14:15:00 | 6477.00 | 6556.64 | 6563.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-28 12:15:00 | 6458.00 | 6520.38 | 6541.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 6479.00 | 6477.47 | 6501.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:00:00 | 6479.00 | 6477.47 | 6501.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 6486.00 | 6479.50 | 6498.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:30:00 | 6624.50 | 6506.60 | 6508.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 6614.00 | 6528.08 | 6518.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 09:15:00 | 6642.00 | 6585.99 | 6554.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 15:15:00 | 6639.50 | 6673.52 | 6639.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 15:15:00 | 6639.50 | 6673.52 | 6639.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 6639.50 | 6673.52 | 6639.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-04 09:45:00 | 6741.50 | 6692.82 | 6651.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 12:15:00 | 6978.50 | 6995.24 | 6995.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-13 12:15:00 | 6978.50 | 6995.24 | 6995.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-13 13:15:00 | 6946.50 | 6985.49 | 6991.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-14 09:15:00 | 7000.50 | 6974.62 | 6983.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-14 09:15:00 | 7000.50 | 6974.62 | 6983.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 09:15:00 | 7000.50 | 6974.62 | 6983.72 | EMA400 retest candle locked (from downside) |

### Cycle 91 — BUY (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 14:15:00 | 7000.00 | 6988.92 | 6988.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 7250.50 | 7043.01 | 7013.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 10:15:00 | 7188.50 | 7237.02 | 7160.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-19 11:00:00 | 7188.50 | 7237.02 | 7160.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 7349.00 | 7415.25 | 7359.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 12:30:00 | 7346.50 | 7415.25 | 7359.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 13:15:00 | 7294.50 | 7391.10 | 7353.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 13:30:00 | 7274.00 | 7391.10 | 7353.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-08-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 15:15:00 | 7107.50 | 7295.64 | 7314.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 09:15:00 | 6960.00 | 7228.51 | 7282.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 11:15:00 | 6932.00 | 6925.82 | 7046.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-25 12:00:00 | 6932.00 | 6925.82 | 7046.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 6849.00 | 6916.41 | 6965.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 6720.00 | 6916.41 | 6965.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 13:15:00 | 7000.00 | 6888.92 | 6923.74 | SL hit (close>static) qty=1.00 sl=6972.00 alert=retest2 |

### Cycle 93 — BUY (started 2025-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 13:15:00 | 6997.50 | 6942.20 | 6938.85 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-08-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 15:15:00 | 6902.00 | 6935.49 | 6936.47 | EMA200 below EMA400 |

### Cycle 95 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 7085.50 | 6965.49 | 6950.01 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 09:15:00 | 6936.00 | 6998.05 | 6998.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 11:15:00 | 6815.50 | 6951.45 | 6976.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 10:15:00 | 6706.00 | 6668.49 | 6720.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 10:15:00 | 6706.00 | 6668.49 | 6720.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 6706.00 | 6668.49 | 6720.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:00:00 | 6706.00 | 6668.49 | 6720.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 11:15:00 | 6710.00 | 6676.80 | 6719.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 11:30:00 | 6734.00 | 6676.80 | 6719.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 12:15:00 | 6748.50 | 6691.14 | 6722.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 12:45:00 | 6753.00 | 6691.14 | 6722.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 13:15:00 | 6764.50 | 6705.81 | 6726.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 13:45:00 | 6770.00 | 6705.81 | 6726.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2025-09-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 15:15:00 | 6820.00 | 6743.40 | 6740.80 | EMA200 above EMA400 |

### Cycle 98 — SELL (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 09:15:00 | 6707.50 | 6736.22 | 6737.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 14:15:00 | 6688.00 | 6717.43 | 6727.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 6760.00 | 6722.27 | 6727.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 6760.00 | 6722.27 | 6727.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 6760.00 | 6722.27 | 6727.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-10 10:15:00 | 6681.00 | 6722.27 | 6727.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 11:15:00 | 6693.50 | 6567.62 | 6564.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — BUY (started 2025-09-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 11:15:00 | 6693.50 | 6567.62 | 6564.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 6773.50 | 6660.12 | 6615.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 15:15:00 | 6771.00 | 6788.94 | 6742.54 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 09:15:00 | 6859.00 | 6788.94 | 6742.54 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 10:15:00 | 6815.00 | 6789.95 | 6747.22 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 11:15:00 | 6808.00 | 6792.26 | 6752.15 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-09-19 12:00:00 | 6803.00 | 6794.41 | 6756.78 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 6848.50 | 6844.61 | 6812.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:45:00 | 6832.00 | 6844.61 | 6812.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 6851.50 | 6861.84 | 6841.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 15:00:00 | 6851.50 | 6861.84 | 6841.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 6803.50 | 6850.17 | 6838.40 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-23 15:15:00 | 6803.50 | 6850.17 | 6838.40 | SL hit (close<ema400) qty=1.00 sl=6838.40 alert=retest1 |

### Cycle 100 — SELL (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 10:15:00 | 6761.00 | 6830.31 | 6831.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 11:15:00 | 6745.00 | 6813.25 | 6823.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-01 13:15:00 | 6314.50 | 6289.36 | 6341.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-01 14:00:00 | 6314.50 | 6289.36 | 6341.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 6305.50 | 6297.32 | 6332.83 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2025-10-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 14:15:00 | 6352.00 | 6330.47 | 6328.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 09:15:00 | 6495.00 | 6365.22 | 6344.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 15:15:00 | 6435.50 | 6445.05 | 6402.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 09:15:00 | 6436.50 | 6445.05 | 6402.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 6446.00 | 6445.24 | 6406.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 10:15:00 | 6467.50 | 6445.24 | 6406.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 11:00:00 | 6462.50 | 6448.69 | 6411.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 13:15:00 | 6466.50 | 6441.64 | 6414.70 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 14:30:00 | 6465.00 | 6438.55 | 6417.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 6438.50 | 6438.54 | 6419.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 09:15:00 | 6493.00 | 6438.54 | 6419.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 13:15:00 | 6560.00 | 6618.11 | 6620.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 13:15:00 | 6560.00 | 6618.11 | 6620.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-15 14:15:00 | 6496.00 | 6593.69 | 6608.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-16 10:15:00 | 6598.00 | 6577.39 | 6595.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-16 10:15:00 | 6598.00 | 6577.39 | 6595.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 10:15:00 | 6598.00 | 6577.39 | 6595.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 11:00:00 | 6598.00 | 6577.39 | 6595.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 6600.00 | 6581.91 | 6596.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:00:00 | 6600.00 | 6581.91 | 6596.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 6605.00 | 6586.53 | 6597.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 14:00:00 | 6575.50 | 6584.33 | 6595.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-24 14:15:00 | 6246.72 | 6313.19 | 6374.73 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-27 11:15:00 | 6301.00 | 6293.47 | 6344.20 | SL hit (close>ema200) qty=0.50 sl=6293.47 alert=retest2 |

### Cycle 103 — BUY (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 09:15:00 | 6399.00 | 6365.57 | 6364.71 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 12:15:00 | 6353.50 | 6379.56 | 6379.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-29 13:15:00 | 6347.00 | 6373.05 | 6376.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 13:15:00 | 6238.00 | 6235.36 | 6276.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-31 13:45:00 | 6249.50 | 6235.36 | 6276.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 5984.50 | 6181.05 | 6241.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 10:45:00 | 5910.50 | 6125.74 | 6210.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 09:15:00 | 5614.97 | 5729.25 | 5830.31 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 5648.50 | 5647.67 | 5724.76 | SL hit (close>ema200) qty=0.50 sl=5647.67 alert=retest2 |

### Cycle 105 — BUY (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 13:15:00 | 5619.50 | 5578.56 | 5577.55 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 11:15:00 | 5512.00 | 5569.02 | 5575.59 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 12:15:00 | 5618.50 | 5572.99 | 5571.95 | EMA200 above EMA400 |

### Cycle 108 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 5486.00 | 5557.87 | 5567.06 | EMA200 below EMA400 |

### Cycle 109 — BUY (started 2025-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 12:15:00 | 5631.50 | 5565.68 | 5556.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 13:15:00 | 5665.50 | 5585.65 | 5566.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 09:15:00 | 5609.50 | 5619.59 | 5589.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-20 10:00:00 | 5609.50 | 5619.59 | 5589.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 5586.00 | 5612.87 | 5589.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 10:45:00 | 5597.00 | 5612.87 | 5589.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 5637.00 | 5617.70 | 5593.82 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 10:15:00 | 5567.50 | 5602.31 | 5604.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 11:15:00 | 5514.00 | 5584.65 | 5596.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 15:15:00 | 5574.00 | 5570.48 | 5584.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 09:15:00 | 5606.50 | 5570.48 | 5584.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 5607.00 | 5577.78 | 5586.73 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 5657.00 | 5594.02 | 5589.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 10:15:00 | 5760.00 | 5627.22 | 5604.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 5806.00 | 5808.40 | 5729.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 10:45:00 | 5807.00 | 5808.40 | 5729.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 11:15:00 | 5777.00 | 5804.56 | 5771.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 11:30:00 | 5771.00 | 5804.56 | 5771.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 5770.00 | 5797.65 | 5771.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 12:45:00 | 5769.50 | 5797.65 | 5771.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 5778.00 | 5793.72 | 5772.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:00:00 | 5778.00 | 5793.72 | 5772.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 5760.00 | 5786.97 | 5771.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:00:00 | 5760.00 | 5786.97 | 5771.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 5741.00 | 5777.78 | 5768.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:15:00 | 5702.50 | 5777.78 | 5768.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 09:15:00 | 5698.00 | 5761.82 | 5762.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 10:15:00 | 5674.50 | 5744.36 | 5754.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 13:15:00 | 5780.00 | 5733.45 | 5745.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 13:15:00 | 5780.00 | 5733.45 | 5745.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 5780.00 | 5733.45 | 5745.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:30:00 | 5782.50 | 5733.45 | 5745.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 5786.50 | 5744.06 | 5748.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 15:00:00 | 5786.50 | 5744.06 | 5748.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 113 — BUY (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 15:15:00 | 5784.00 | 5752.05 | 5752.00 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 09:15:00 | 5694.00 | 5740.44 | 5746.73 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 12:15:00 | 5770.00 | 5752.56 | 5751.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 14:15:00 | 5797.50 | 5764.26 | 5756.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 5705.50 | 5757.75 | 5755.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 5705.50 | 5757.75 | 5755.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 5705.50 | 5757.75 | 5755.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:45:00 | 5703.00 | 5757.75 | 5755.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 116 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 5684.00 | 5743.00 | 5749.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-04 12:15:00 | 5669.50 | 5697.93 | 5718.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-05 10:15:00 | 5700.00 | 5671.55 | 5694.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 10:15:00 | 5700.00 | 5671.55 | 5694.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 10:15:00 | 5700.00 | 5671.55 | 5694.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 10:30:00 | 5704.50 | 5671.55 | 5694.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 11:15:00 | 5700.00 | 5677.24 | 5694.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 11:45:00 | 5700.00 | 5677.24 | 5694.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 12:15:00 | 5700.00 | 5681.79 | 5695.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 12:30:00 | 5705.50 | 5681.79 | 5695.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 5629.50 | 5664.81 | 5684.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:30:00 | 5660.50 | 5664.81 | 5684.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 5570.00 | 5505.30 | 5559.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:00:00 | 5570.00 | 5505.30 | 5559.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 5534.00 | 5511.04 | 5557.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 10:45:00 | 5501.00 | 5522.29 | 5551.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-10 13:30:00 | 5519.50 | 5512.32 | 5539.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 14:15:00 | 5613.00 | 5541.58 | 5537.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-12-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 14:15:00 | 5613.00 | 5541.58 | 5537.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 5675.50 | 5578.51 | 5555.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 10:15:00 | 5650.50 | 5663.74 | 5622.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 11:00:00 | 5650.50 | 5663.74 | 5622.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 5609.50 | 5648.81 | 5632.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:00:00 | 5609.50 | 5648.81 | 5632.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 10:15:00 | 5596.00 | 5638.24 | 5629.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 10:30:00 | 5590.00 | 5638.24 | 5629.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 5620.50 | 5630.61 | 5627.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:45:00 | 5616.50 | 5630.61 | 5627.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 5621.00 | 5628.69 | 5627.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 5634.00 | 5628.69 | 5627.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 5619.50 | 5626.85 | 5626.41 | EMA400 retest candle locked (from upside) |

### Cycle 118 — SELL (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 10:15:00 | 5606.00 | 5622.68 | 5624.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 11:15:00 | 5578.00 | 5613.74 | 5620.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 13:15:00 | 5607.50 | 5605.18 | 5614.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-17 14:00:00 | 5607.50 | 5605.18 | 5614.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 5596.00 | 5603.34 | 5613.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 5596.00 | 5603.34 | 5613.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 5470.50 | 5422.46 | 5447.68 | EMA400 retest candle locked (from downside) |

### Cycle 119 — BUY (started 2025-12-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 11:15:00 | 5556.50 | 5462.16 | 5462.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-23 13:15:00 | 5590.00 | 5505.86 | 5483.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 10:15:00 | 5663.50 | 5664.36 | 5605.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 11:00:00 | 5663.50 | 5664.36 | 5605.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 5615.00 | 5689.03 | 5666.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 10:00:00 | 5615.00 | 5689.03 | 5666.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 5577.50 | 5666.72 | 5658.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:00:00 | 5577.50 | 5666.72 | 5658.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — SELL (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 11:15:00 | 5532.00 | 5639.78 | 5647.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 12:15:00 | 5498.50 | 5611.52 | 5633.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 5571.50 | 5554.92 | 5591.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 10:45:00 | 5572.00 | 5554.92 | 5591.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 5543.50 | 5556.62 | 5581.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 14:45:00 | 5576.00 | 5556.62 | 5581.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 09:15:00 | 5508.50 | 5500.93 | 5530.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:00:00 | 5508.50 | 5500.93 | 5530.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 5543.50 | 5509.45 | 5531.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 10:45:00 | 5558.50 | 5509.45 | 5531.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 5555.50 | 5518.66 | 5533.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:00:00 | 5555.50 | 5518.66 | 5533.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 5550.00 | 5524.93 | 5534.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:15:00 | 5569.50 | 5524.93 | 5534.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 121 — BUY (started 2026-01-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 14:15:00 | 5567.00 | 5542.95 | 5542.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-05 09:15:00 | 5688.00 | 5577.89 | 5558.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 09:15:00 | 5717.00 | 5717.25 | 5656.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-06 12:15:00 | 5678.50 | 5704.51 | 5665.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 12:15:00 | 5678.50 | 5704.51 | 5665.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 12:30:00 | 5676.50 | 5704.51 | 5665.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 13:15:00 | 5675.50 | 5698.71 | 5666.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-06 14:00:00 | 5675.50 | 5698.71 | 5666.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 14:15:00 | 5748.00 | 5708.57 | 5673.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 09:15:00 | 5783.50 | 5715.86 | 5679.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 14:15:00 | 5718.50 | 5834.02 | 5842.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 122 — SELL (started 2026-01-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 14:15:00 | 5718.50 | 5834.02 | 5842.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 15:15:00 | 5700.00 | 5807.22 | 5829.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 5694.50 | 5645.99 | 5709.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 5694.50 | 5645.99 | 5709.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 5694.50 | 5645.99 | 5709.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:00:00 | 5694.50 | 5645.99 | 5709.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 5718.50 | 5660.50 | 5710.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 10:30:00 | 5700.50 | 5660.50 | 5710.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 11:15:00 | 5762.50 | 5680.90 | 5714.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:45:00 | 5769.50 | 5680.90 | 5714.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 15:15:00 | 5782.00 | 5737.22 | 5734.14 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 09:15:00 | 5699.50 | 5729.68 | 5730.99 | EMA200 below EMA400 |

### Cycle 125 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 5749.50 | 5733.64 | 5732.67 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 15:15:00 | 5678.00 | 5723.46 | 5728.72 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 5875.00 | 5758.26 | 5743.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 11:15:00 | 5907.50 | 5788.10 | 5758.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 5786.00 | 5836.46 | 5800.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 09:15:00 | 5786.00 | 5836.46 | 5800.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 5786.00 | 5836.46 | 5800.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 6078.50 | 5801.09 | 5795.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 13:00:00 | 5889.50 | 5864.34 | 5833.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-21 09:15:00 | 5722.50 | 5844.66 | 5836.57 | SL hit (close<static) qty=1.00 sl=5754.00 alert=retest2 |

### Cycle 128 — SELL (started 2026-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-21 10:15:00 | 5620.00 | 5799.73 | 5816.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-22 09:15:00 | 5525.50 | 5667.09 | 5736.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-23 15:15:00 | 5551.00 | 5536.13 | 5594.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 15:15:00 | 5551.00 | 5536.13 | 5594.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 5551.00 | 5536.13 | 5594.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 09:15:00 | 5431.50 | 5536.13 | 5594.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 14:15:00 | 5474.50 | 5491.80 | 5545.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 5662.50 | 5523.21 | 5545.79 | SL hit (close>static) qty=1.00 sl=5649.50 alert=retest2 |

### Cycle 129 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 5659.00 | 5574.42 | 5566.64 | EMA200 above EMA400 |

### Cycle 130 — SELL (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 13:15:00 | 5535.00 | 5570.66 | 5575.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 14:15:00 | 5515.50 | 5559.62 | 5569.96 | Break + close below crossover candle low |

### Cycle 131 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 5667.50 | 5579.66 | 5577.18 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 13:15:00 | 5530.50 | 5568.58 | 5572.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 15:15:00 | 5525.00 | 5553.69 | 5565.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 09:15:00 | 5493.00 | 5488.65 | 5516.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 5493.00 | 5488.65 | 5516.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 5493.00 | 5488.65 | 5516.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:30:00 | 5470.50 | 5488.65 | 5516.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 5465.00 | 5472.96 | 5493.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 5483.50 | 5472.96 | 5493.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 5555.00 | 5489.37 | 5499.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:45:00 | 5581.00 | 5489.37 | 5499.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 10:15:00 | 5538.50 | 5499.19 | 5503.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 10:30:00 | 5559.50 | 5499.19 | 5503.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 133 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 5624.00 | 5524.15 | 5514.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 10:15:00 | 5657.50 | 5603.78 | 5563.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 10:15:00 | 5657.00 | 5669.42 | 5625.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 11:00:00 | 5657.00 | 5669.42 | 5625.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 14:15:00 | 5724.00 | 5705.66 | 5678.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 14:30:00 | 5704.50 | 5705.66 | 5678.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 5832.50 | 5875.75 | 5834.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:00:00 | 5832.50 | 5875.75 | 5834.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 15:15:00 | 5810.00 | 5862.60 | 5831.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 09:15:00 | 5869.00 | 5862.60 | 5831.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 5865.50 | 5863.18 | 5834.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:30:00 | 5889.50 | 5865.12 | 5840.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 12:45:00 | 5907.00 | 5873.09 | 5846.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 14:00:00 | 5889.50 | 5876.37 | 5850.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 09:15:00 | 5774.50 | 5851.36 | 5845.16 | SL hit (close<static) qty=1.00 sl=5809.50 alert=retest2 |

### Cycle 134 — SELL (started 2026-02-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 10:15:00 | 5754.00 | 5831.89 | 5836.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 5682.50 | 5789.16 | 5812.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 13:15:00 | 5633.00 | 5614.60 | 5677.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 14:00:00 | 5633.00 | 5614.60 | 5677.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 5680.00 | 5627.68 | 5677.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 5680.00 | 5627.68 | 5677.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 15:15:00 | 5675.00 | 5637.15 | 5677.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 09:15:00 | 5686.00 | 5637.15 | 5677.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 5741.00 | 5657.92 | 5683.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 5741.00 | 5657.92 | 5683.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 5787.00 | 5683.73 | 5692.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 5799.50 | 5683.73 | 5692.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 135 — BUY (started 2026-02-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 11:15:00 | 5785.50 | 5704.09 | 5701.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 5834.00 | 5730.07 | 5713.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-18 15:15:00 | 5872.50 | 5873.63 | 5820.77 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-19 09:15:00 | 5952.00 | 5873.63 | 5820.77 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 5867.00 | 5870.64 | 5836.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:30:00 | 5848.50 | 5870.64 | 5836.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 5796.50 | 5853.07 | 5834.07 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-19 14:15:00 | 5796.50 | 5853.07 | 5834.07 | SL hit (close<ema400) qty=1.00 sl=5834.07 alert=retest1 |

### Cycle 136 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 5775.00 | 5820.64 | 5821.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 15:15:00 | 5720.00 | 5779.73 | 5799.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 13:15:00 | 5771.00 | 5769.23 | 5785.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 13:15:00 | 5771.00 | 5769.23 | 5785.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 5771.00 | 5769.23 | 5785.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 13:45:00 | 5789.00 | 5769.23 | 5785.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 5768.00 | 5768.98 | 5784.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:00:00 | 5768.00 | 5768.98 | 5784.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 5807.50 | 5776.69 | 5786.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-24 09:15:00 | 5729.00 | 5776.69 | 5786.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 14:15:00 | 5735.50 | 5701.09 | 5697.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — BUY (started 2026-02-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-26 14:15:00 | 5735.50 | 5701.09 | 5697.74 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 10:15:00 | 5650.00 | 5687.87 | 5692.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 11:15:00 | 5636.00 | 5677.50 | 5687.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 5550.50 | 5543.13 | 5578.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 5550.50 | 5543.13 | 5578.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 5550.50 | 5543.13 | 5578.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:15:00 | 5507.00 | 5540.07 | 5568.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:45:00 | 5507.00 | 5533.65 | 5563.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 15:00:00 | 5516.00 | 5530.12 | 5558.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 10:15:00 | 5514.00 | 5524.86 | 5551.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 10:15:00 | 5231.65 | 5374.42 | 5455.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 10:15:00 | 5231.65 | 5374.42 | 5455.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 10:15:00 | 5240.20 | 5374.42 | 5455.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 10:15:00 | 5238.30 | 5374.42 | 5455.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 5105.00 | 5095.23 | 5187.49 | SL hit (close>ema200) qty=0.50 sl=5095.23 alert=retest2 |

### Cycle 139 — BUY (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 14:15:00 | 5201.50 | 5188.38 | 5187.10 | EMA200 above EMA400 |

### Cycle 140 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 5031.50 | 5158.86 | 5174.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 4966.50 | 5120.39 | 5155.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 13:15:00 | 5011.00 | 4996.85 | 5049.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 13:45:00 | 5008.50 | 4996.85 | 5049.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 5050.50 | 5007.58 | 5049.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 5081.00 | 5007.58 | 5049.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 5050.00 | 5016.06 | 5049.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 5093.50 | 5016.06 | 5049.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 5085.00 | 5043.12 | 5057.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:45:00 | 5085.50 | 5043.12 | 5057.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — BUY (started 2026-03-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 12:15:00 | 5146.00 | 5076.48 | 5070.54 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 5035.00 | 5080.38 | 5084.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 5007.50 | 5058.14 | 5073.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 14:15:00 | 5008.50 | 5002.43 | 5027.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 15:00:00 | 5008.50 | 5002.43 | 5027.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 11:15:00 | 4854.00 | 4945.30 | 4990.21 | EMA400 retest candle locked (from downside) |

### Cycle 143 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 5113.50 | 4989.26 | 4977.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-24 15:15:00 | 5120.00 | 5036.09 | 5001.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 5104.50 | 5214.50 | 5144.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 5104.50 | 5214.50 | 5144.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 5104.50 | 5214.50 | 5144.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 5104.50 | 5214.50 | 5144.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 5101.00 | 5191.80 | 5140.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 5101.00 | 5191.80 | 5140.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 11:15:00 | 5112.00 | 5175.84 | 5138.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:30:00 | 5113.00 | 5175.84 | 5138.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 5089.00 | 5143.43 | 5131.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 5089.00 | 5143.43 | 5131.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 5080.00 | 5130.74 | 5126.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 5059.50 | 5130.74 | 5126.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 5032.50 | 5111.10 | 5118.05 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2026-04-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 10:15:00 | 5139.00 | 5109.78 | 5109.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 5160.50 | 5124.68 | 5116.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 5060.50 | 5117.68 | 5116.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 5060.50 | 5117.68 | 5116.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 5060.50 | 5117.68 | 5116.59 | EMA400 retest candle locked (from upside) |

### Cycle 146 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 5099.00 | 5113.94 | 5114.99 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 5162.50 | 5123.11 | 5118.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 5182.00 | 5134.89 | 5124.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 5216.00 | 5239.69 | 5198.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 5216.00 | 5239.69 | 5198.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 5575.00 | 5636.62 | 5566.57 | EMA400 retest candle locked (from upside) |

### Cycle 148 — SELL (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 14:15:00 | 5441.00 | 5526.23 | 5532.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 15:15:00 | 5434.00 | 5507.79 | 5523.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 10:15:00 | 5506.00 | 5505.14 | 5519.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 10:15:00 | 5506.00 | 5505.14 | 5519.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 10:15:00 | 5506.00 | 5505.14 | 5519.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-15 10:30:00 | 5514.50 | 5505.14 | 5519.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 5493.50 | 5473.15 | 5494.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:45:00 | 5494.00 | 5473.15 | 5494.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 5507.00 | 5479.92 | 5495.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-16 11:00:00 | 5507.00 | 5479.92 | 5495.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 5444.50 | 5472.84 | 5490.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-16 12:45:00 | 5425.00 | 5470.87 | 5488.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 09:15:00 | 5526.50 | 5490.50 | 5492.91 | SL hit (close>static) qty=1.00 sl=5514.00 alert=retest2 |

### Cycle 149 — BUY (started 2026-04-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 11:15:00 | 5510.00 | 5497.60 | 5495.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 10:15:00 | 5586.00 | 5534.26 | 5515.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-22 15:15:00 | 5877.50 | 5934.49 | 5847.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-23 09:15:00 | 5830.00 | 5934.49 | 5847.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 09:15:00 | 5846.00 | 5916.79 | 5847.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 09:30:00 | 5808.50 | 5916.79 | 5847.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 10:15:00 | 5818.00 | 5897.03 | 5844.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:00:00 | 5818.00 | 5897.03 | 5844.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 11:15:00 | 5797.00 | 5877.03 | 5840.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 11:45:00 | 5781.50 | 5877.03 | 5840.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 150 — SELL (started 2026-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 09:15:00 | 5640.50 | 5788.32 | 5807.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 10:15:00 | 5598.00 | 5750.26 | 5788.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 5517.00 | 5492.08 | 5553.29 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-29 13:45:00 | 5475.50 | 5494.98 | 5536.53 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 5468.00 | 5360.08 | 5407.62 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-05-04 10:15:00 | 5468.00 | 5360.08 | 5407.62 | SL hit (close>ema400) qty=1.00 sl=5407.62 alert=retest1 |

### Cycle 151 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 5446.00 | 5373.29 | 5369.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 14:15:00 | 5525.50 | 5419.53 | 5392.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 5577.00 | 5581.47 | 5514.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 10:00:00 | 5577.00 | 5581.47 | 5514.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-14 12:45:00 | 3910.80 | 2024-05-15 09:15:00 | 3850.50 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2024-05-28 09:15:00 | 4043.40 | 2024-05-29 09:15:00 | 3968.95 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2024-05-31 14:30:00 | 3868.80 | 2024-06-03 09:15:00 | 4051.00 | STOP_HIT | 1.00 | -4.71% |
| BUY | retest2 | 2024-06-12 14:30:00 | 4239.45 | 2024-06-20 11:15:00 | 4308.20 | STOP_HIT | 1.00 | 1.62% |
| BUY | retest2 | 2024-06-13 09:15:00 | 4263.00 | 2024-06-20 11:15:00 | 4308.20 | STOP_HIT | 1.00 | 1.06% |
| BUY | retest2 | 2024-06-14 09:15:00 | 4280.00 | 2024-06-20 11:15:00 | 4308.20 | STOP_HIT | 1.00 | 0.66% |
| BUY | retest2 | 2024-07-01 09:30:00 | 4414.45 | 2024-07-01 11:15:00 | 4408.90 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2024-07-12 12:45:00 | 4335.00 | 2024-07-19 09:15:00 | 4298.10 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2024-07-12 13:15:00 | 4345.60 | 2024-07-19 09:15:00 | 4298.10 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-07-15 09:45:00 | 4338.40 | 2024-07-19 09:15:00 | 4298.10 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-07-23 15:15:00 | 4500.00 | 2024-07-25 10:15:00 | 4382.00 | STOP_HIT | 1.00 | -2.62% |
| SELL | retest2 | 2024-08-13 12:45:00 | 4184.25 | 2024-08-16 15:15:00 | 4225.05 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-08-13 13:45:00 | 4186.00 | 2024-08-16 15:15:00 | 4225.05 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-08-22 13:45:00 | 4430.00 | 2024-08-29 10:15:00 | 4413.30 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2024-08-23 11:00:00 | 4398.35 | 2024-08-29 10:15:00 | 4413.30 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2024-08-23 15:00:00 | 4399.85 | 2024-08-29 10:15:00 | 4413.30 | STOP_HIT | 1.00 | 0.31% |
| BUY | retest2 | 2024-09-03 09:15:00 | 4491.30 | 2024-09-11 12:15:00 | 4696.95 | STOP_HIT | 1.00 | 4.58% |
| SELL | retest2 | 2024-09-20 13:30:00 | 4647.55 | 2024-09-23 10:15:00 | 4763.65 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2024-09-25 09:15:00 | 4764.80 | 2024-09-25 10:15:00 | 4714.60 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-10-01 14:15:00 | 4603.75 | 2024-10-07 13:15:00 | 4373.56 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-03 09:15:00 | 4570.10 | 2024-10-08 09:15:00 | 4341.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 14:15:00 | 4603.75 | 2024-10-08 13:15:00 | 4423.80 | STOP_HIT | 0.50 | 3.91% |
| SELL | retest2 | 2024-10-03 09:15:00 | 4570.10 | 2024-10-08 13:15:00 | 4423.80 | STOP_HIT | 0.50 | 3.20% |
| BUY | retest2 | 2024-10-16 14:30:00 | 4349.95 | 2024-10-17 11:15:00 | 4278.10 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2024-10-21 14:00:00 | 4209.00 | 2024-10-28 09:15:00 | 3998.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 14:45:00 | 4211.55 | 2024-10-28 09:15:00 | 4000.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 09:45:00 | 4212.10 | 2024-10-28 09:15:00 | 4001.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 12:15:00 | 4190.00 | 2024-10-28 09:15:00 | 4017.17 | PARTIAL | 0.50 | 4.12% |
| SELL | retest2 | 2024-10-24 11:30:00 | 4228.60 | 2024-10-28 09:15:00 | 3996.55 | PARTIAL | 0.50 | 5.49% |
| SELL | retest2 | 2024-10-21 14:00:00 | 4209.00 | 2024-10-28 10:15:00 | 4190.35 | STOP_HIT | 0.50 | 0.44% |
| SELL | retest2 | 2024-10-21 14:45:00 | 4211.55 | 2024-10-28 10:15:00 | 4190.35 | STOP_HIT | 0.50 | 0.50% |
| SELL | retest2 | 2024-10-22 09:45:00 | 4212.10 | 2024-10-28 10:15:00 | 4190.35 | STOP_HIT | 0.50 | 0.52% |
| SELL | retest2 | 2024-10-22 12:15:00 | 4190.00 | 2024-10-28 10:15:00 | 4190.35 | STOP_HIT | 0.50 | -0.01% |
| SELL | retest2 | 2024-10-24 11:30:00 | 4228.60 | 2024-10-28 10:15:00 | 4190.35 | STOP_HIT | 0.50 | 0.90% |
| SELL | retest2 | 2024-10-25 09:15:00 | 4206.90 | 2024-10-28 13:15:00 | 4259.90 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-10-31 13:30:00 | 4351.35 | 2024-11-04 09:15:00 | 4274.90 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-11-01 18:00:00 | 4346.95 | 2024-11-04 09:15:00 | 4274.90 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-11-01 18:45:00 | 4346.80 | 2024-11-04 09:15:00 | 4274.90 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2024-12-13 09:30:00 | 4679.45 | 2024-12-17 14:15:00 | 4670.85 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2024-12-13 10:45:00 | 4681.00 | 2024-12-17 14:15:00 | 4670.85 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest2 | 2024-12-13 12:45:00 | 4682.00 | 2024-12-17 14:15:00 | 4670.85 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2024-12-17 14:15:00 | 4666.55 | 2024-12-17 14:15:00 | 4670.85 | STOP_HIT | 1.00 | 0.09% |
| SELL | retest2 | 2024-12-20 15:00:00 | 4509.40 | 2024-12-24 12:15:00 | 4610.55 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2024-12-23 11:45:00 | 4565.70 | 2024-12-24 12:15:00 | 4610.55 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2024-12-23 13:45:00 | 4565.00 | 2024-12-24 12:15:00 | 4610.55 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-12-31 15:15:00 | 4589.80 | 2025-01-01 13:15:00 | 4591.20 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-01-06 13:30:00 | 4712.00 | 2025-01-09 15:15:00 | 4720.10 | STOP_HIT | 1.00 | 0.17% |
| BUY | retest2 | 2025-01-06 15:15:00 | 4728.00 | 2025-01-09 15:15:00 | 4720.10 | STOP_HIT | 1.00 | -0.17% |
| SELL | retest1 | 2025-01-13 09:15:00 | 4515.70 | 2025-01-15 10:15:00 | 4464.90 | STOP_HIT | 1.00 | 1.12% |
| SELL | retest1 | 2025-01-13 11:15:00 | 4472.05 | 2025-01-15 10:15:00 | 4464.90 | STOP_HIT | 1.00 | 0.16% |
| BUY | retest2 | 2025-01-27 11:45:00 | 4849.75 | 2025-01-30 14:15:00 | 4782.65 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-01-28 14:45:00 | 4848.80 | 2025-01-30 14:15:00 | 4782.65 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-01-28 15:15:00 | 4848.00 | 2025-01-30 14:15:00 | 4782.65 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-01-29 09:30:00 | 4893.60 | 2025-01-30 14:15:00 | 4782.65 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-02-07 11:30:00 | 4917.25 | 2025-02-10 09:15:00 | 4834.20 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2025-02-07 12:15:00 | 4905.95 | 2025-02-10 09:15:00 | 4834.20 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-02-07 13:30:00 | 4912.95 | 2025-02-10 09:15:00 | 4834.20 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-02-18 10:15:00 | 4532.55 | 2025-02-20 10:15:00 | 4656.00 | STOP_HIT | 1.00 | -2.72% |
| SELL | retest2 | 2025-02-18 11:00:00 | 4541.20 | 2025-02-20 10:15:00 | 4656.00 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2025-02-18 11:45:00 | 4516.85 | 2025-02-20 10:15:00 | 4656.00 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-02-27 09:15:00 | 4556.05 | 2025-03-05 11:15:00 | 4500.05 | STOP_HIT | 1.00 | 1.23% |
| SELL | retest2 | 2025-03-13 10:30:00 | 4306.50 | 2025-03-17 13:15:00 | 4374.90 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-03-25 09:30:00 | 4790.10 | 2025-03-26 11:15:00 | 4720.15 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-03-25 11:45:00 | 4790.60 | 2025-03-26 11:15:00 | 4720.15 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2025-03-25 13:30:00 | 4784.95 | 2025-03-26 11:15:00 | 4720.15 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-03-28 09:30:00 | 4943.05 | 2025-04-07 09:15:00 | 4841.35 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-03-28 10:00:00 | 4949.85 | 2025-04-07 09:15:00 | 4841.35 | STOP_HIT | 1.00 | -2.19% |
| BUY | retest2 | 2025-04-01 11:15:00 | 4941.75 | 2025-04-07 09:15:00 | 4841.35 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-04-01 12:30:00 | 4960.20 | 2025-04-07 09:15:00 | 4841.35 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-04-02 11:15:00 | 4960.10 | 2025-04-07 09:15:00 | 4841.35 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2025-04-03 10:00:00 | 4955.00 | 2025-04-07 09:15:00 | 4841.35 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-04-04 10:45:00 | 4948.40 | 2025-04-07 09:15:00 | 4841.35 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2025-04-04 12:30:00 | 4946.15 | 2025-04-07 09:15:00 | 4841.35 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-04-17 11:30:00 | 5051.70 | 2025-04-28 13:15:00 | 5121.50 | STOP_HIT | 1.00 | 1.38% |
| BUY | retest2 | 2025-04-17 12:00:00 | 5048.00 | 2025-04-28 13:15:00 | 5121.50 | STOP_HIT | 1.00 | 1.46% |
| BUY | retest2 | 2025-04-21 09:30:00 | 5056.00 | 2025-04-28 13:15:00 | 5121.50 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2025-04-21 10:15:00 | 5047.90 | 2025-04-28 13:15:00 | 5121.50 | STOP_HIT | 1.00 | 1.46% |
| BUY | retest2 | 2025-04-21 11:15:00 | 5065.20 | 2025-04-28 13:15:00 | 5121.50 | STOP_HIT | 1.00 | 1.11% |
| BUY | retest2 | 2025-04-21 12:15:00 | 5056.10 | 2025-04-28 13:15:00 | 5121.50 | STOP_HIT | 1.00 | 1.29% |
| BUY | retest2 | 2025-04-22 09:45:00 | 5114.90 | 2025-04-28 13:15:00 | 5121.50 | STOP_HIT | 1.00 | 0.13% |
| SELL | retest2 | 2025-05-09 09:15:00 | 5002.00 | 2025-05-09 12:15:00 | 5086.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-05-23 13:45:00 | 5124.00 | 2025-05-26 09:15:00 | 5448.00 | STOP_HIT | 1.00 | -6.32% |
| BUY | retest2 | 2025-06-11 09:15:00 | 5826.00 | 2025-06-17 10:15:00 | 5944.00 | STOP_HIT | 1.00 | 2.03% |
| SELL | retest2 | 2025-06-18 10:30:00 | 5896.50 | 2025-06-24 14:15:00 | 5832.00 | STOP_HIT | 1.00 | 1.09% |
| SELL | retest2 | 2025-06-18 11:30:00 | 5889.00 | 2025-06-24 14:15:00 | 5832.00 | STOP_HIT | 1.00 | 0.97% |
| BUY | retest2 | 2025-06-27 11:30:00 | 5954.00 | 2025-07-11 09:15:00 | 6549.40 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-07-22 12:45:00 | 6459.00 | 2025-07-22 13:15:00 | 6492.50 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-08-04 09:45:00 | 6741.50 | 2025-08-13 12:15:00 | 6978.50 | STOP_HIT | 1.00 | 3.52% |
| SELL | retest2 | 2025-08-28 09:15:00 | 6720.00 | 2025-08-28 13:15:00 | 7000.00 | STOP_HIT | 1.00 | -4.17% |
| SELL | retest2 | 2025-09-10 10:15:00 | 6681.00 | 2025-09-16 11:15:00 | 6693.50 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest1 | 2025-09-19 09:15:00 | 6859.00 | 2025-09-23 15:15:00 | 6803.50 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest1 | 2025-09-19 10:15:00 | 6815.00 | 2025-09-23 15:15:00 | 6803.50 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest1 | 2025-09-19 11:15:00 | 6808.00 | 2025-09-23 15:15:00 | 6803.50 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest1 | 2025-09-19 12:00:00 | 6803.00 | 2025-09-23 15:15:00 | 6803.50 | STOP_HIT | 1.00 | 0.01% |
| BUY | retest2 | 2025-10-08 10:15:00 | 6467.50 | 2025-10-15 13:15:00 | 6560.00 | STOP_HIT | 1.00 | 1.43% |
| BUY | retest2 | 2025-10-08 11:00:00 | 6462.50 | 2025-10-15 13:15:00 | 6560.00 | STOP_HIT | 1.00 | 1.51% |
| BUY | retest2 | 2025-10-08 13:15:00 | 6466.50 | 2025-10-15 13:15:00 | 6560.00 | STOP_HIT | 1.00 | 1.45% |
| BUY | retest2 | 2025-10-08 14:30:00 | 6465.00 | 2025-10-15 13:15:00 | 6560.00 | STOP_HIT | 1.00 | 1.47% |
| BUY | retest2 | 2025-10-09 09:15:00 | 6493.00 | 2025-10-15 13:15:00 | 6560.00 | STOP_HIT | 1.00 | 1.03% |
| SELL | retest2 | 2025-10-16 14:00:00 | 6575.50 | 2025-10-24 14:15:00 | 6246.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-16 14:00:00 | 6575.50 | 2025-10-27 11:15:00 | 6301.00 | STOP_HIT | 0.50 | 4.17% |
| SELL | retest2 | 2025-11-03 10:45:00 | 5910.50 | 2025-11-07 09:15:00 | 5614.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 10:45:00 | 5910.50 | 2025-11-10 10:15:00 | 5648.50 | STOP_HIT | 0.50 | 4.43% |
| SELL | retest2 | 2025-12-10 10:45:00 | 5501.00 | 2025-12-11 14:15:00 | 5613.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-12-10 13:30:00 | 5519.50 | 2025-12-11 14:15:00 | 5613.00 | STOP_HIT | 1.00 | -1.69% |
| BUY | retest2 | 2026-01-07 09:15:00 | 5783.50 | 2026-01-09 14:15:00 | 5718.50 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2026-01-20 09:15:00 | 6078.50 | 2026-01-21 09:15:00 | 5722.50 | STOP_HIT | 1.00 | -5.86% |
| BUY | retest2 | 2026-01-20 13:00:00 | 5889.50 | 2026-01-21 09:15:00 | 5722.50 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2026-01-27 09:15:00 | 5431.50 | 2026-01-28 09:15:00 | 5662.50 | STOP_HIT | 1.00 | -4.25% |
| SELL | retest2 | 2026-01-27 14:15:00 | 5474.50 | 2026-01-28 09:15:00 | 5662.50 | STOP_HIT | 1.00 | -3.43% |
| BUY | retest2 | 2026-02-11 11:30:00 | 5889.50 | 2026-02-12 09:15:00 | 5774.50 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2026-02-11 12:45:00 | 5907.00 | 2026-02-12 09:15:00 | 5774.50 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2026-02-11 14:00:00 | 5889.50 | 2026-02-12 09:15:00 | 5774.50 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest1 | 2026-02-19 09:15:00 | 5952.00 | 2026-02-19 14:15:00 | 5796.50 | STOP_HIT | 1.00 | -2.61% |
| SELL | retest2 | 2026-02-24 09:15:00 | 5729.00 | 2026-02-26 14:15:00 | 5735.50 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2026-03-05 13:15:00 | 5507.00 | 2026-03-09 10:15:00 | 5231.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 13:45:00 | 5507.00 | 2026-03-09 10:15:00 | 5231.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 15:00:00 | 5516.00 | 2026-03-09 10:15:00 | 5240.20 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 10:15:00 | 5514.00 | 2026-03-09 10:15:00 | 5238.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 13:15:00 | 5507.00 | 2026-03-11 09:15:00 | 5105.00 | STOP_HIT | 0.50 | 7.30% |
| SELL | retest2 | 2026-03-05 13:45:00 | 5507.00 | 2026-03-11 09:15:00 | 5105.00 | STOP_HIT | 0.50 | 7.30% |
| SELL | retest2 | 2026-03-05 15:00:00 | 5516.00 | 2026-03-11 09:15:00 | 5105.00 | STOP_HIT | 0.50 | 7.45% |
| SELL | retest2 | 2026-03-06 10:15:00 | 5514.00 | 2026-03-11 09:15:00 | 5105.00 | STOP_HIT | 0.50 | 7.42% |
| SELL | retest2 | 2026-04-16 12:45:00 | 5425.00 | 2026-04-17 09:15:00 | 5526.50 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest1 | 2026-04-29 13:45:00 | 5475.50 | 2026-05-04 10:15:00 | 5468.00 | STOP_HIT | 1.00 | 0.14% |
