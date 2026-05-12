# KEI Industries Ltd. (KEI)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 5117.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 4 |
| ALERT2 | 3 |
| ALERT2_SKIP | 1 |
| ALERT3 | 20 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 19 |
| PARTIAL | 0 |
| TARGET_HIT | 1 |
| STOP_HIT | 19 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 1 / 18
- **Target hits / Stop hits / Partials:** 1 / 18 / 0
- **Avg / median % per leg:** -1.77% / -1.67%
- **Sum % (uncompounded):** -33.67%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 15 | 1 | 6.7% | 1 | 14 | 0 | -0.99% | -14.8% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 15 | 1 | 6.7% | 1 | 14 | 0 | -0.99% | -14.8% |
| SELL (all) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.71% | -18.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 4 | 0 | 0.0% | 0 | 4 | 0 | -4.71% | -18.9% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 19 | 1 | 5.3% | 1 | 18 | 0 | -1.77% | -33.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 10:15:00 | 3554.40 | 3266.96 | 3266.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 11:15:00 | 3563.90 | 3269.92 | 3267.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 10:15:00 | 3685.90 | 3691.49 | 3576.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 10:45:00 | 3684.20 | 3691.49 | 3576.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 3731.60 | 3820.75 | 3730.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:00:00 | 3731.60 | 3820.75 | 3730.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 3741.50 | 3819.96 | 3730.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 10:45:00 | 3737.00 | 3819.96 | 3730.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 11:15:00 | 3729.40 | 3819.06 | 3730.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 11:45:00 | 3733.10 | 3819.06 | 3730.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 12:15:00 | 3770.60 | 3818.58 | 3730.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 12:30:00 | 3733.50 | 3818.58 | 3730.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 3813.00 | 3866.20 | 3785.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-29 10:00:00 | 3813.00 | 3866.20 | 3785.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 4054.40 | 4134.98 | 4051.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:45:00 | 4047.90 | 4134.98 | 4051.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 13:15:00 | 4052.60 | 4134.16 | 4051.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 15:00:00 | 4066.60 | 4133.49 | 4051.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 12:30:00 | 4075.00 | 4124.83 | 4054.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-31 14:15:00 | 4035.20 | 4123.46 | 4054.12 | SL hit (close<static) qty=1.00 sl=4047.40 alert=retest2 |

### Cycle 2 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 3907.10 | 4178.65 | 4179.84 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2026-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 12:15:00 | 4414.30 | 4176.62 | 4176.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-05 14:15:00 | 4439.70 | 4181.59 | 4178.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-10 11:15:00 | 4552.50 | 4666.93 | 4494.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-10 11:45:00 | 4545.50 | 4666.93 | 4494.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 4442.00 | 4664.69 | 4493.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 13:00:00 | 4442.00 | 4664.69 | 4493.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 13:15:00 | 4505.00 | 4663.10 | 4493.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 13:30:00 | 4476.00 | 4663.10 | 4493.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 14:15:00 | 4538.50 | 4661.86 | 4494.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 14:45:00 | 4507.50 | 4661.86 | 4494.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 4426.00 | 4658.42 | 4494.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:00:00 | 4426.00 | 4658.42 | 4494.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 4370.00 | 4655.55 | 4493.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:45:00 | 4403.00 | 4655.55 | 4493.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 4089.00 | 4388.96 | 4389.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 4015.50 | 4385.25 | 4388.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-08 09:15:00 | 4500.10 | 4308.83 | 4345.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 09:15:00 | 4500.10 | 4308.83 | 4345.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 4500.10 | 4308.83 | 4345.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 09:45:00 | 4425.00 | 4322.16 | 4351.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 10:45:00 | 4418.20 | 4323.24 | 4351.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 11:30:00 | 4425.20 | 4324.17 | 4351.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-09 12:30:00 | 4425.50 | 4325.15 | 4352.29 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 4406.00 | 4337.63 | 4357.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 10:45:00 | 4442.70 | 4337.63 | 4357.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-15 11:15:00 | 4632.00 | 4350.60 | 4362.97 | SL hit (close>static) qty=1.00 sl=4580.50 alert=retest2 |

### Cycle 5 — BUY (started 2026-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 14:15:00 | 4646.00 | 4376.85 | 4375.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 4779.30 | 4383.62 | 4379.22 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-10-28 15:00:00 | 4066.60 | 2025-10-31 14:15:00 | 4035.20 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-10-31 12:30:00 | 4075.00 | 2025-10-31 14:15:00 | 4035.20 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-11-12 09:15:00 | 4082.30 | 2025-11-25 15:15:00 | 4048.00 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2025-11-12 12:15:00 | 4067.20 | 2025-11-25 15:15:00 | 4048.00 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2025-11-21 12:15:00 | 4088.10 | 2025-11-25 15:15:00 | 4048.00 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-11-21 13:45:00 | 4085.70 | 2025-11-25 15:15:00 | 4048.00 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-11-21 15:00:00 | 4084.10 | 2025-12-09 09:15:00 | 4030.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-11-24 09:15:00 | 4098.40 | 2025-12-09 09:15:00 | 4030.00 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-11-28 09:30:00 | 4198.00 | 2025-12-09 09:15:00 | 4030.00 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest2 | 2025-11-28 11:45:00 | 4167.00 | 2025-12-09 09:15:00 | 4030.00 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2025-11-28 13:30:00 | 4160.00 | 2025-12-09 09:15:00 | 4030.00 | STOP_HIT | 1.00 | -3.12% |
| BUY | retest2 | 2025-12-02 13:45:00 | 4166.00 | 2025-12-09 09:15:00 | 4030.00 | STOP_HIT | 1.00 | -3.26% |
| BUY | retest2 | 2025-12-15 10:00:00 | 4121.40 | 2025-12-18 09:15:00 | 4050.30 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2025-12-17 15:00:00 | 4110.10 | 2025-12-18 09:15:00 | 4050.30 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-12-19 09:15:00 | 4154.40 | 2026-01-02 09:15:00 | 4569.84 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-04-09 09:45:00 | 4425.00 | 2026-04-15 11:15:00 | 4632.00 | STOP_HIT | 1.00 | -4.68% |
| SELL | retest2 | 2026-04-09 10:45:00 | 4418.20 | 2026-04-15 11:15:00 | 4632.00 | STOP_HIT | 1.00 | -4.84% |
| SELL | retest2 | 2026-04-09 11:30:00 | 4425.20 | 2026-04-15 11:15:00 | 4632.00 | STOP_HIT | 1.00 | -4.67% |
| SELL | retest2 | 2026-04-09 12:30:00 | 4425.50 | 2026-04-15 11:15:00 | 4632.00 | STOP_HIT | 1.00 | -4.67% |
