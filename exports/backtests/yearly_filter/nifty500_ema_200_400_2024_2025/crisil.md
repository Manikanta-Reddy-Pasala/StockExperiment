# CRISIL Ltd. (CRISIL)

## Backtest Summary

- **Window:** 2023-04-10 09:15:00 → 2026-05-08 15:15:00 (5324 bars)
- **Last close:** 4160.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 6 |
| ALERT1 | 6 |
| ALERT2 | 7 |
| ALERT2_SKIP | 1 |
| ALERT3 | 53 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 27 |
| PARTIAL | 2 |
| TARGET_HIT | 3 |
| STOP_HIT | 22 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 27 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 5 / 22
- **Target hits / Stop hits / Partials:** 3 / 22 / 2
- **Avg / median % per leg:** -0.82% / -1.58%
- **Sum % (uncompounded):** -22.10%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 4 | 1 | 25.0% | 1 | 3 | 0 | 0.22% | 0.9% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 4 | 1 | 25.0% | 1 | 3 | 0 | 0.22% | 0.9% |
| SELL (all) | 23 | 4 | 17.4% | 2 | 19 | 2 | -1.00% | -23.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 23 | 4 | 17.4% | 2 | 19 | 2 | -1.00% | -23.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 27 | 5 | 18.5% | 3 | 22 | 2 | -0.82% | -22.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 11:15:00 | 4560.20 | 4324.48 | 4324.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 15:15:00 | 4619.00 | 4365.22 | 4345.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 11:15:00 | 4583.05 | 4587.74 | 4509.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-25 12:00:00 | 4583.05 | 4587.74 | 4509.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 10:15:00 | 4497.30 | 4590.60 | 4523.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 10:45:00 | 4498.90 | 4590.60 | 4523.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 4499.70 | 4589.70 | 4523.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 11:45:00 | 4498.75 | 4589.70 | 4523.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 4515.40 | 4548.28 | 4509.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 11:00:00 | 4515.40 | 4548.28 | 4509.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 11:15:00 | 4507.10 | 4547.87 | 4509.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 12:00:00 | 4507.10 | 4547.87 | 4509.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 12:15:00 | 4500.05 | 4547.40 | 4509.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-09 13:00:00 | 4500.05 | 4547.40 | 4509.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 13:15:00 | 4522.10 | 4547.14 | 4509.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 09:15:00 | 4528.00 | 4546.47 | 4509.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-10 10:30:00 | 4529.55 | 4545.82 | 4509.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 12:15:00 | 4476.60 | 4544.78 | 4509.48 | SL hit (close<static) qty=1.00 sl=4490.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-02-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 14:15:00 | 5122.65 | 5437.16 | 5437.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 15:15:00 | 5092.55 | 5433.73 | 5436.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-15 14:15:00 | 4369.80 | 4367.65 | 4625.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-15 15:00:00 | 4369.80 | 4367.65 | 4625.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 10:15:00 | 4610.00 | 4387.30 | 4614.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 11:00:00 | 4610.00 | 4387.30 | 4614.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 11:15:00 | 4637.50 | 4389.79 | 4614.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 12:00:00 | 4637.50 | 4389.79 | 4614.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 12:15:00 | 4660.00 | 4392.48 | 4614.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-21 13:00:00 | 4660.00 | 4392.48 | 4614.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 11:15:00 | 4735.50 | 4476.18 | 4610.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 11:45:00 | 4723.80 | 4476.18 | 4610.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 12:15:00 | 4678.40 | 4478.19 | 4610.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 12:30:00 | 4731.60 | 4478.19 | 4610.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 11:15:00 | 4621.90 | 4488.01 | 4612.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 11:30:00 | 4626.00 | 4488.01 | 4612.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 12:15:00 | 4626.10 | 4489.38 | 4612.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 12:45:00 | 4622.90 | 4489.38 | 4612.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 4626.00 | 4493.29 | 4612.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:15:00 | 4645.00 | 4493.29 | 4612.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 4635.50 | 4499.13 | 4612.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:00:00 | 4635.50 | 4499.13 | 4612.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 4650.00 | 4500.64 | 4613.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:45:00 | 4645.60 | 4500.64 | 4613.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 10:15:00 | 4686.00 | 4506.62 | 4613.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:00:00 | 4686.00 | 4506.62 | 4613.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 11:15:00 | 5019.10 | 4696.81 | 4695.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 13:15:00 | 5032.70 | 4703.38 | 4698.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 10:15:00 | 5779.50 | 5786.03 | 5551.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 10:45:00 | 5772.00 | 5786.03 | 5551.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 14:15:00 | 5538.50 | 5777.56 | 5559.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 15:00:00 | 5538.50 | 5777.56 | 5559.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 5520.00 | 5775.00 | 5559.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:15:00 | 5494.50 | 5775.00 | 5559.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-08-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 13:15:00 | 5302.50 | 5434.01 | 5434.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 5273.50 | 5418.35 | 5425.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-20 09:15:00 | 4786.40 | 4783.88 | 4955.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-20 10:00:00 | 4786.40 | 4783.88 | 4955.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 4970.00 | 4791.90 | 4953.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 4959.90 | 4791.90 | 4953.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 4957.00 | 4793.54 | 4953.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 09:15:00 | 4950.70 | 4793.54 | 4953.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 4945.80 | 4795.06 | 4953.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 10:45:00 | 4882.20 | 4861.16 | 4956.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 15:15:00 | 4860.00 | 4861.40 | 4954.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-12 15:15:00 | 4638.09 | 4819.91 | 4914.02 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-13 11:15:00 | 4617.00 | 4814.80 | 4910.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-11-28 10:15:00 | 4393.98 | 4680.81 | 4803.06 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 5 — BUY (started 2026-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-05 11:15:00 | 4653.60 | 4610.41 | 4610.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 09:15:00 | 4689.30 | 4612.46 | 4611.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 10:15:00 | 4634.00 | 4634.37 | 4623.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-12 10:15:00 | 4634.00 | 4634.37 | 4623.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 10:15:00 | 4634.00 | 4634.37 | 4623.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 11:00:00 | 4634.00 | 4634.37 | 4623.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 11:15:00 | 4611.00 | 4634.14 | 4623.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 12:00:00 | 4611.00 | 4634.14 | 4623.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 12:15:00 | 4666.80 | 4634.46 | 4623.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 13:30:00 | 4685.00 | 4634.87 | 4623.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 09:15:00 | 4365.80 | 4632.78 | 4622.91 | SL hit (close<static) qty=1.00 sl=4611.00 alert=retest2 |

### Cycle 6 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 4512.90 | 4613.57 | 4613.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 12:15:00 | 4465.80 | 4601.10 | 4607.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 13:15:00 | 4064.10 | 4063.83 | 4238.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-10 13:45:00 | 4062.60 | 4063.83 | 4238.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 11:15:00 | 4201.40 | 4074.76 | 4223.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 11:45:00 | 4208.00 | 4074.76 | 4223.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 12:15:00 | 4295.50 | 4076.96 | 4223.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 12:30:00 | 4314.60 | 4076.96 | 4223.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 13:15:00 | 4330.00 | 4079.47 | 4224.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 13:45:00 | 4339.00 | 4079.47 | 4224.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 4276.50 | 4165.90 | 4246.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:15:00 | 4286.50 | 4165.90 | 4246.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 4265.80 | 4175.29 | 4247.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:00:00 | 4265.80 | 4175.29 | 4247.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 4266.30 | 4195.01 | 4251.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 10:45:00 | 4240.50 | 4195.52 | 4251.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 11:45:00 | 4245.10 | 4196.01 | 4251.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2024-07-05 09:15:00 | 4251.55 | 2024-07-12 10:15:00 | 4345.15 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2024-07-05 10:45:00 | 4263.50 | 2024-07-12 10:15:00 | 4345.15 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2024-07-05 15:15:00 | 4261.80 | 2024-07-12 10:15:00 | 4345.15 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-07-08 10:30:00 | 4271.30 | 2024-07-12 10:15:00 | 4345.15 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2024-07-11 10:15:00 | 4285.95 | 2024-07-15 09:15:00 | 4536.00 | STOP_HIT | 1.00 | -5.83% |
| SELL | retest2 | 2024-07-11 11:30:00 | 4282.45 | 2024-07-15 09:15:00 | 4536.00 | STOP_HIT | 1.00 | -5.92% |
| SELL | retest2 | 2024-07-11 13:30:00 | 4284.95 | 2024-07-15 09:15:00 | 4536.00 | STOP_HIT | 1.00 | -5.86% |
| SELL | retest2 | 2024-07-11 15:15:00 | 4287.00 | 2024-07-15 09:15:00 | 4536.00 | STOP_HIT | 1.00 | -5.81% |
| SELL | retest2 | 2024-07-18 10:45:00 | 4307.90 | 2024-07-18 11:15:00 | 4361.00 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-07-18 15:00:00 | 4298.25 | 2024-07-29 15:15:00 | 4341.00 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2024-07-19 09:30:00 | 4324.20 | 2024-07-29 15:15:00 | 4341.00 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2024-07-22 10:00:00 | 4295.80 | 2024-07-29 15:15:00 | 4341.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-07-23 12:15:00 | 4171.00 | 2024-07-29 15:15:00 | 4341.00 | STOP_HIT | 1.00 | -4.08% |
| SELL | retest2 | 2024-07-23 13:30:00 | 4231.10 | 2024-07-30 09:15:00 | 4385.90 | STOP_HIT | 1.00 | -3.66% |
| SELL | retest2 | 2024-07-24 11:30:00 | 4234.80 | 2024-07-30 09:15:00 | 4385.90 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2024-07-24 15:00:00 | 4243.35 | 2024-07-30 09:15:00 | 4385.90 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2024-08-02 09:30:00 | 4290.95 | 2024-08-02 10:15:00 | 4323.95 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2024-08-05 09:15:00 | 4264.55 | 2024-08-07 10:15:00 | 4331.95 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2024-08-07 12:00:00 | 4272.05 | 2024-08-08 09:15:00 | 4318.00 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2024-10-10 09:15:00 | 4528.00 | 2024-10-10 12:15:00 | 4476.60 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-10-10 10:30:00 | 4529.55 | 2024-10-10 12:15:00 | 4476.60 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2024-10-11 09:45:00 | 4548.50 | 2024-10-17 09:15:00 | 5003.35 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-11-03 10:45:00 | 4882.20 | 2025-11-12 15:15:00 | 4638.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 15:15:00 | 4860.00 | 2025-11-13 11:15:00 | 4617.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-03 10:45:00 | 4882.20 | 2025-11-28 10:15:00 | 4393.98 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-11-03 15:15:00 | 4860.00 | 2025-12-08 13:15:00 | 4374.00 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-02-12 13:30:00 | 4685.00 | 2026-02-13 09:15:00 | 4365.80 | STOP_HIT | 1.00 | -6.81% |
