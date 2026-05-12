# J.K. Cement Ltd. (JKCEMENT)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 5555.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 11 |
| ALERT1 | 11 |
| ALERT2 | 11 |
| ALERT2_SKIP | 5 |
| ALERT3 | 70 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 45 |
| PARTIAL | 9 |
| TARGET_HIT | 3 |
| STOP_HIT | 42 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 54 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 14 / 40
- **Target hits / Stop hits / Partials:** 3 / 42 / 9
- **Avg / median % per leg:** -0.67% / -1.60%
- **Sum % (uncompounded):** -36.23%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 2 | 9.1% | 2 | 20 | 0 | -1.33% | -29.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 22 | 2 | 9.1% | 2 | 20 | 0 | -1.33% | -29.3% |
| SELL (all) | 32 | 12 | 37.5% | 1 | 22 | 9 | -0.22% | -7.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 32 | 12 | 37.5% | 1 | 22 | 9 | -0.22% | -7.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 54 | 14 | 25.9% | 3 | 42 | 9 | -0.67% | -36.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-06 10:15:00 | 3151.85 | 3216.81 | 3217.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-06 11:15:00 | 3135.90 | 3216.00 | 3216.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-11 11:15:00 | 3239.95 | 3200.72 | 3208.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-11 11:15:00 | 3239.95 | 3200.72 | 3208.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 11:15:00 | 3239.95 | 3200.72 | 3208.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 12:00:00 | 3239.95 | 3200.72 | 3208.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 12:15:00 | 3205.00 | 3200.76 | 3208.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 12:30:00 | 3228.00 | 3200.76 | 3208.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 13:15:00 | 3201.00 | 3200.77 | 3208.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-11 13:30:00 | 3207.05 | 3200.77 | 3208.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 14:15:00 | 3185.15 | 3200.61 | 3208.30 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-10-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-18 12:15:00 | 3253.85 | 3215.07 | 3214.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-18 13:15:00 | 3275.30 | 3215.67 | 3215.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-23 11:15:00 | 3214.00 | 3229.99 | 3222.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-23 11:15:00 | 3214.00 | 3229.99 | 3222.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 11:15:00 | 3214.00 | 3229.99 | 3222.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-23 12:00:00 | 3214.00 | 3229.99 | 3222.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 12:15:00 | 3212.95 | 3229.82 | 3222.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-23 13:00:00 | 3212.95 | 3229.82 | 3222.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-23 13:15:00 | 3228.60 | 3229.81 | 3222.76 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2023-10-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-26 12:15:00 | 3102.50 | 3215.95 | 3216.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-26 13:15:00 | 3080.70 | 3214.60 | 3215.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-06 09:15:00 | 3351.05 | 3179.66 | 3195.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-11-06 09:15:00 | 3351.05 | 3179.66 | 3195.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-06 09:15:00 | 3351.05 | 3179.66 | 3195.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-06 10:00:00 | 3351.05 | 3179.66 | 3195.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2023-11-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-08 11:15:00 | 3457.85 | 3211.67 | 3210.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-09 09:15:00 | 3471.75 | 3222.88 | 3216.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 09:15:00 | 4112.25 | 4155.55 | 3978.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-15 09:45:00 | 4135.00 | 4155.55 | 3978.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 4133.65 | 4283.91 | 4146.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 10:00:00 | 4133.65 | 4283.91 | 4146.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 10:15:00 | 4124.45 | 4282.33 | 4146.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 11:00:00 | 4124.45 | 4282.33 | 4146.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 11:15:00 | 4088.50 | 4280.40 | 4146.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 12:00:00 | 4088.50 | 4280.40 | 4146.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 10:15:00 | 4119.00 | 4195.62 | 4127.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-22 10:30:00 | 4121.15 | 4195.62 | 4127.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 11:15:00 | 4106.25 | 4194.73 | 4127.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-22 11:45:00 | 4108.00 | 4194.73 | 4127.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-22 12:15:00 | 4108.00 | 4193.87 | 4127.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-22 13:00:00 | 4108.00 | 4193.87 | 4127.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 10:15:00 | 4200.65 | 4245.38 | 4181.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-16 11:30:00 | 4217.25 | 4244.97 | 4181.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-16 12:45:00 | 4221.50 | 4244.61 | 4181.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-16 13:15:00 | 4211.00 | 4244.61 | 4181.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-16 14:00:00 | 4214.30 | 4244.30 | 4181.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-18 09:15:00 | 4226.35 | 4243.32 | 4181.94 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-04-18 14:15:00 | 4108.60 | 4240.86 | 4182.22 | SL hit (close<static) qty=1.00 sl=4170.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-05-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-03 15:15:00 | 3980.00 | 4144.35 | 4144.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-06 09:15:00 | 3960.00 | 4142.51 | 4143.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 09:15:00 | 4028.55 | 3996.59 | 4053.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-24 10:00:00 | 4028.55 | 3996.59 | 4053.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 10:15:00 | 4008.65 | 3996.71 | 4053.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 12:15:00 | 3995.80 | 3996.78 | 4052.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 13:15:00 | 4000.40 | 3996.89 | 4052.70 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-24 14:15:00 | 3992.70 | 3996.98 | 4052.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 11:15:00 | 3999.30 | 3997.06 | 4049.53 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 09:15:00 | 3890.25 | 3992.54 | 4043.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 14:30:00 | 3868.80 | 3981.87 | 4035.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 15:00:00 | 3863.40 | 3981.87 | 4035.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 09:15:00 | 4051.00 | 3981.52 | 4034.74 | SL hit (close>static) qty=1.00 sl=4048.00 alert=retest2 |

### Cycle 6 — BUY (started 2024-06-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 14:15:00 | 4331.35 | 4065.54 | 4064.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-18 09:15:00 | 4352.80 | 4070.89 | 4067.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 10:15:00 | 4208.95 | 4242.01 | 4171.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 10:45:00 | 4208.95 | 4242.01 | 4171.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 09:15:00 | 4200.00 | 4242.61 | 4176.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 10:00:00 | 4200.00 | 4242.61 | 4176.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 15:15:00 | 4253.60 | 4351.51 | 4276.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-05 09:15:00 | 4232.60 | 4351.51 | 4276.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 4221.75 | 4350.22 | 4276.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-07 09:30:00 | 4276.35 | 4335.16 | 4273.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-08 10:30:00 | 4276.65 | 4331.43 | 4274.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 11:15:00 | 4189.00 | 4330.01 | 4273.88 | SL hit (close<static) qty=1.00 sl=4202.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-10-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-21 12:15:00 | 4228.85 | 4447.76 | 4447.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-22 10:15:00 | 4185.05 | 4435.96 | 4441.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-25 09:15:00 | 4284.45 | 4178.18 | 4273.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-25 09:15:00 | 4284.45 | 4178.18 | 4273.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 4284.45 | 4178.18 | 4273.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-25 09:45:00 | 4271.70 | 4178.18 | 4273.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 10:15:00 | 4204.00 | 4178.44 | 4273.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-26 09:15:00 | 4190.00 | 4180.21 | 4272.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-27 15:15:00 | 4180.10 | 4178.72 | 4265.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 09:15:00 | 4438.35 | 4182.49 | 4260.73 | SL hit (close>static) qty=1.00 sl=4292.70 alert=retest2 |

### Cycle 8 — BUY (started 2024-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 11:15:00 | 4580.35 | 4325.75 | 4324.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 14:15:00 | 4599.05 | 4332.87 | 4328.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 09:15:00 | 4570.00 | 4593.62 | 4503.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-10 10:00:00 | 4570.00 | 4593.62 | 4503.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 4552.75 | 4593.21 | 4504.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:30:00 | 4507.05 | 4593.21 | 4504.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 4494.50 | 4591.06 | 4505.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 09:30:00 | 4518.95 | 4591.06 | 4505.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 10:15:00 | 4522.45 | 4590.38 | 4505.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 11:15:00 | 4472.05 | 4590.38 | 4505.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 11:15:00 | 4427.60 | 4588.76 | 4505.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 12:00:00 | 4427.60 | 4588.76 | 4505.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 12:15:00 | 4424.50 | 4587.13 | 4504.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-13 13:00:00 | 4424.50 | 4587.13 | 4504.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 12:15:00 | 4479.95 | 4563.68 | 4498.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 13:00:00 | 4479.95 | 4563.68 | 4498.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 13:15:00 | 4457.50 | 4562.62 | 4498.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-15 13:45:00 | 4455.00 | 4562.62 | 4498.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 4545.75 | 4559.52 | 4499.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 11:45:00 | 4600.15 | 4553.22 | 4500.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 09:45:00 | 4574.20 | 4555.10 | 4503.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 10:15:00 | 4564.75 | 4555.10 | 4503.15 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 11:00:00 | 4561.00 | 4555.16 | 4503.43 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 12:15:00 | 4546.00 | 4554.71 | 4503.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:45:00 | 4506.80 | 4554.71 | 4503.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 11:15:00 | 4642.90 | 4735.67 | 4644.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 12:00:00 | 4642.90 | 4735.67 | 4644.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 12:15:00 | 4628.50 | 4734.61 | 4644.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 13:00:00 | 4628.50 | 4734.61 | 4644.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 13:15:00 | 4602.80 | 4733.30 | 4644.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-14 14:00:00 | 4602.80 | 4733.30 | 4644.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 15:15:00 | 4651.00 | 4731.80 | 4644.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-17 09:15:00 | 4633.65 | 4731.80 | 4644.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-17 09:15:00 | 4577.60 | 4730.26 | 4644.37 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-27 13:15:00 | 4488.00 | 4668.83 | 4629.85 | SL hit (close<static) qty=1.00 sl=4492.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-03-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-07 09:15:00 | 4448.85 | 4596.16 | 4596.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-07 11:15:00 | 4431.40 | 4593.14 | 4595.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 09:15:00 | 4530.90 | 4512.39 | 4549.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-19 10:15:00 | 4574.75 | 4513.01 | 4549.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 10:15:00 | 4574.75 | 4513.01 | 4549.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 11:00:00 | 4574.75 | 4513.01 | 4549.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 11:15:00 | 4589.30 | 4513.77 | 4549.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 12:00:00 | 4589.30 | 4513.77 | 4549.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2025-03-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 10:15:00 | 4750.10 | 4580.58 | 4579.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 10:15:00 | 4784.45 | 4590.98 | 4585.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-07 11:15:00 | 4690.25 | 4708.75 | 4651.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-07 12:00:00 | 4690.25 | 4708.75 | 4651.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 6627.50 | 6839.41 | 6557.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 10:30:00 | 6654.00 | 6836.15 | 6559.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 09:30:00 | 6670.00 | 6796.83 | 6573.32 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 10:15:00 | 6378.50 | 6776.91 | 6571.96 | SL hit (close<static) qty=1.00 sl=6455.50 alert=retest2 |

### Cycle 11 — SELL (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-27 12:15:00 | 6361.00 | 6538.86 | 6539.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 09:15:00 | 6279.50 | 6512.70 | 6525.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 14:15:00 | 5925.00 | 5888.49 | 6116.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-26 15:00:00 | 5925.00 | 5888.49 | 6116.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 12:15:00 | 5790.00 | 5638.46 | 5803.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 12:45:00 | 5791.00 | 5638.46 | 5803.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 5756.00 | 5639.63 | 5803.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 13:45:00 | 5798.00 | 5639.63 | 5803.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 5821.00 | 5648.33 | 5799.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 5821.00 | 5648.33 | 5799.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 5902.00 | 5650.86 | 5800.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 5902.00 | 5650.86 | 5800.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 5966.00 | 5653.99 | 5801.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:00:00 | 5966.00 | 5653.99 | 5801.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 5922.50 | 5670.72 | 5805.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:45:00 | 5916.00 | 5670.72 | 5805.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 5934.50 | 5673.35 | 5805.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 12:00:00 | 5934.50 | 5673.35 | 5805.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 5839.50 | 5688.55 | 5809.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:30:00 | 5820.50 | 5688.55 | 5809.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 5819.50 | 5689.85 | 5809.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 14:15:00 | 5784.50 | 5691.24 | 5809.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 10:15:00 | 5875.00 | 5693.91 | 5796.62 | SL hit (close>static) qty=1.00 sl=5850.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-04-16 11:30:00 | 4217.25 | 2024-04-18 14:15:00 | 4108.60 | STOP_HIT | 1.00 | -2.58% |
| BUY | retest2 | 2024-04-16 12:45:00 | 4221.50 | 2024-04-18 14:15:00 | 4108.60 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2024-04-16 13:15:00 | 4211.00 | 2024-04-18 14:15:00 | 4108.60 | STOP_HIT | 1.00 | -2.43% |
| BUY | retest2 | 2024-04-16 14:00:00 | 4214.30 | 2024-04-18 14:15:00 | 4108.60 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2024-05-24 12:15:00 | 3995.80 | 2024-06-03 09:15:00 | 4051.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-05-24 13:15:00 | 4000.40 | 2024-06-03 09:15:00 | 4051.00 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2024-05-24 14:15:00 | 3992.70 | 2024-06-04 11:15:00 | 3796.01 | PARTIAL | 0.50 | 4.93% |
| SELL | retest2 | 2024-05-28 11:15:00 | 3999.30 | 2024-06-04 11:15:00 | 3800.38 | PARTIAL | 0.50 | 4.97% |
| SELL | retest2 | 2024-05-31 14:30:00 | 3868.80 | 2024-06-04 11:15:00 | 3793.06 | PARTIAL | 0.50 | 1.96% |
| SELL | retest2 | 2024-05-31 15:00:00 | 3863.40 | 2024-06-04 11:15:00 | 3799.34 | PARTIAL | 0.50 | 1.66% |
| SELL | retest2 | 2024-06-04 11:00:00 | 3851.10 | 2024-06-04 12:15:00 | 3658.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-04 11:45:00 | 3848.50 | 2024-06-04 12:15:00 | 3656.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-24 14:15:00 | 3992.70 | 2024-06-05 10:15:00 | 4017.05 | STOP_HIT | 0.50 | -0.61% |
| SELL | retest2 | 2024-05-28 11:15:00 | 3999.30 | 2024-06-05 10:15:00 | 4017.05 | STOP_HIT | 0.50 | -0.44% |
| SELL | retest2 | 2024-05-31 14:30:00 | 3868.80 | 2024-06-05 10:15:00 | 4017.05 | STOP_HIT | 0.50 | -3.83% |
| SELL | retest2 | 2024-05-31 15:00:00 | 3863.40 | 2024-06-05 10:15:00 | 4017.05 | STOP_HIT | 0.50 | -3.98% |
| SELL | retest2 | 2024-06-04 11:00:00 | 3851.10 | 2024-06-05 10:15:00 | 4017.05 | STOP_HIT | 0.50 | -4.31% |
| SELL | retest2 | 2024-06-04 11:45:00 | 3848.50 | 2024-06-05 10:15:00 | 4017.05 | STOP_HIT | 0.50 | -4.38% |
| SELL | retest2 | 2024-06-05 12:15:00 | 3966.95 | 2024-06-07 09:15:00 | 4066.35 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2024-06-06 12:00:00 | 3968.00 | 2024-06-07 09:15:00 | 4066.35 | STOP_HIT | 1.00 | -2.48% |
| BUY | retest2 | 2024-08-07 09:30:00 | 4276.35 | 2024-08-08 11:15:00 | 4189.00 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-08-08 10:30:00 | 4276.65 | 2024-08-08 11:15:00 | 4189.00 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2024-08-19 11:00:00 | 4274.70 | 2024-09-05 09:15:00 | 4702.17 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-21 10:45:00 | 4281.55 | 2024-09-05 09:15:00 | 4709.71 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-26 09:15:00 | 4190.00 | 2024-12-02 09:15:00 | 4438.35 | STOP_HIT | 1.00 | -5.93% |
| SELL | retest2 | 2024-11-27 15:15:00 | 4180.10 | 2024-12-02 09:15:00 | 4438.35 | STOP_HIT | 1.00 | -6.18% |
| BUY | retest2 | 2025-01-21 11:45:00 | 4600.15 | 2025-02-27 13:15:00 | 4488.00 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-01-22 09:45:00 | 4574.20 | 2025-02-27 13:15:00 | 4488.00 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-01-22 10:15:00 | 4564.75 | 2025-02-27 13:15:00 | 4488.00 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-01-22 11:00:00 | 4561.00 | 2025-02-27 13:15:00 | 4488.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-09-08 10:30:00 | 6654.00 | 2025-09-15 10:15:00 | 6378.50 | STOP_HIT | 1.00 | -4.14% |
| BUY | retest2 | 2025-09-12 09:30:00 | 6670.00 | 2025-09-15 10:15:00 | 6378.50 | STOP_HIT | 1.00 | -4.37% |
| BUY | retest2 | 2025-09-16 11:30:00 | 6671.00 | 2025-09-26 11:15:00 | 6399.00 | STOP_HIT | 1.00 | -4.08% |
| BUY | retest2 | 2025-09-16 12:00:00 | 6693.50 | 2025-09-26 11:15:00 | 6399.00 | STOP_HIT | 1.00 | -4.40% |
| BUY | retest2 | 2025-10-09 13:30:00 | 6594.50 | 2025-10-15 14:15:00 | 6496.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-10-10 09:15:00 | 6659.50 | 2025-10-15 14:15:00 | 6496.00 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2025-10-13 09:30:00 | 6601.50 | 2025-10-15 14:15:00 | 6496.00 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-10-15 12:15:00 | 6600.00 | 2025-10-15 14:15:00 | 6496.00 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-10-16 09:15:00 | 6568.50 | 2025-10-20 09:15:00 | 6434.00 | STOP_HIT | 1.00 | -2.05% |
| BUY | retest2 | 2025-10-17 12:30:00 | 6513.00 | 2025-10-20 09:15:00 | 6434.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2026-01-09 14:15:00 | 5784.50 | 2026-01-16 10:15:00 | 5875.00 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2026-01-19 09:15:00 | 5763.00 | 2026-01-20 09:15:00 | 6054.00 | STOP_HIT | 1.00 | -5.05% |
| SELL | retest2 | 2026-01-19 15:00:00 | 5775.00 | 2026-01-20 09:15:00 | 6054.00 | STOP_HIT | 1.00 | -4.83% |
| SELL | retest2 | 2026-01-21 09:15:00 | 5654.50 | 2026-01-27 10:15:00 | 5371.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-21 09:15:00 | 5654.50 | 2026-02-04 09:15:00 | 5636.50 | STOP_HIT | 0.50 | 0.32% |
| SELL | retest2 | 2026-02-06 09:15:00 | 5671.00 | 2026-02-09 09:15:00 | 5849.00 | STOP_HIT | 1.00 | -3.14% |
| SELL | retest2 | 2026-02-06 12:00:00 | 5705.00 | 2026-02-09 09:15:00 | 5849.00 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2026-02-13 09:15:00 | 5704.00 | 2026-02-17 10:15:00 | 5787.00 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-02-13 09:45:00 | 5703.00 | 2026-02-17 10:15:00 | 5787.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2026-02-23 10:30:00 | 5735.00 | 2026-02-23 15:15:00 | 5807.50 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-02-24 09:15:00 | 5729.00 | 2026-03-02 09:15:00 | 5442.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-24 09:15:00 | 5729.00 | 2026-03-09 10:15:00 | 5156.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-04-24 09:15:00 | 5683.50 | 2026-04-30 09:15:00 | 5399.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-24 09:15:00 | 5683.50 | 2026-05-06 14:15:00 | 5525.50 | STOP_HIT | 0.50 | 2.78% |
