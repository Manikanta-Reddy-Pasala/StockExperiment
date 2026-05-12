# L&T Technology Services Ltd. (LTTS)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 3801.60
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 135 |
| ALERT1 | 101 |
| ALERT2 | 101 |
| ALERT2_SKIP | 43 |
| ALERT3 | 234 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 129 |
| PARTIAL | 15 |
| TARGET_HIT | 4 |
| STOP_HIT | 126 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 145 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 50 / 95
- **Target hits / Stop hits / Partials:** 4 / 126 / 15
- **Avg / median % per leg:** 0.20% / -0.56%
- **Sum % (uncompounded):** 28.87%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 56 | 12 | 21.4% | 0 | 56 | 0 | -0.67% | -37.6% |
| BUY @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.36% | -0.4% |
| BUY @ 3rd Alert (retest2) | 55 | 12 | 21.8% | 0 | 55 | 0 | -0.68% | -37.3% |
| SELL (all) | 89 | 38 | 42.7% | 4 | 70 | 15 | 0.75% | 66.5% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 89 | 38 | 42.7% | 4 | 70 | 15 | 0.75% | 66.5% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.36% | -0.4% |
| retest2 (combined) | 144 | 50 | 34.7% | 4 | 125 | 15 | 0.20% | 29.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 4481.55 | 4426.90 | 4420.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 13:15:00 | 4487.50 | 4446.55 | 4430.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 09:15:00 | 4456.10 | 4457.39 | 4440.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 10:00:00 | 4456.10 | 4457.39 | 4440.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 4466.00 | 4459.11 | 4442.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:30:00 | 4448.70 | 4459.11 | 4442.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 4442.95 | 4456.62 | 4448.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 09:15:00 | 4529.40 | 4456.62 | 4448.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-17 15:15:00 | 4450.00 | 4480.55 | 4480.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 15:15:00 | 4450.00 | 4480.55 | 4480.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 09:15:00 | 4430.00 | 4464.83 | 4472.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-21 10:15:00 | 4474.30 | 4466.73 | 4472.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 10:15:00 | 4474.30 | 4466.73 | 4472.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 10:15:00 | 4474.30 | 4466.73 | 4472.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-21 11:00:00 | 4474.30 | 4466.73 | 4472.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 11:15:00 | 4486.30 | 4470.64 | 4474.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-21 12:00:00 | 4486.30 | 4470.64 | 4474.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 12:15:00 | 4495.00 | 4475.51 | 4476.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-21 12:45:00 | 4496.80 | 4475.51 | 4476.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2024-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-21 14:15:00 | 4482.00 | 4477.27 | 4476.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-22 09:15:00 | 4504.80 | 4484.97 | 4480.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 09:15:00 | 4588.35 | 4604.09 | 4574.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 4588.35 | 4604.09 | 4574.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 4588.35 | 4604.09 | 4574.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 10:00:00 | 4644.00 | 4607.86 | 4594.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 12:45:00 | 4648.15 | 4623.91 | 4605.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 14:30:00 | 4644.90 | 4629.47 | 4611.57 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 15:00:00 | 4647.15 | 4629.47 | 4611.57 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 4663.15 | 4637.89 | 4618.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 10:15:00 | 4679.60 | 4637.89 | 4618.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 11:00:00 | 4672.00 | 4644.71 | 4623.47 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 11:30:00 | 4671.00 | 4653.17 | 4629.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-30 09:15:00 | 4590.15 | 4650.11 | 4639.08 | SL hit (close<static) qty=1.00 sl=4599.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 12:15:00 | 4588.00 | 4624.17 | 4628.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 13:15:00 | 4567.55 | 4612.84 | 4623.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 15:15:00 | 4508.85 | 4508.71 | 4549.01 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-03 09:15:00 | 4526.80 | 4508.71 | 4549.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 4499.05 | 4506.78 | 4544.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 10:30:00 | 4478.55 | 4505.83 | 4540.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 12:30:00 | 4481.95 | 4500.00 | 4531.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 14:15:00 | 4488.70 | 4499.09 | 4528.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 14:45:00 | 4489.80 | 4496.82 | 4524.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 4254.62 | 4480.84 | 4512.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 4257.85 | 4480.84 | 4512.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 4264.26 | 4480.84 | 4512.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 09:15:00 | 4265.31 | 4480.84 | 4512.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-04 12:15:00 | 4436.10 | 4434.69 | 4480.21 | SL hit (close>ema200) qty=0.50 sl=4434.69 alert=retest2 |

### Cycle 5 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 4623.55 | 4489.66 | 4488.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 11:15:00 | 4637.55 | 4519.24 | 4502.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 09:15:00 | 4740.00 | 4770.01 | 4709.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-10 10:00:00 | 4740.00 | 4770.01 | 4709.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 4854.15 | 4892.62 | 4876.47 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2024-06-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 14:15:00 | 4847.75 | 4869.71 | 4870.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 10:15:00 | 4845.70 | 4861.15 | 4865.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 14:15:00 | 4860.90 | 4859.40 | 4863.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 14:15:00 | 4860.90 | 4859.40 | 4863.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 14:15:00 | 4860.90 | 4859.40 | 4863.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 15:00:00 | 4860.90 | 4859.40 | 4863.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 15:15:00 | 4863.00 | 4860.12 | 4863.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:15:00 | 4859.90 | 4860.12 | 4863.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 4834.40 | 4854.98 | 4860.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 10:15:00 | 4820.25 | 4854.98 | 4860.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 11:15:00 | 4862.30 | 4860.91 | 4860.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2024-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-20 11:15:00 | 4862.30 | 4860.91 | 4860.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-21 09:15:00 | 4975.40 | 4890.34 | 4875.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-21 14:15:00 | 4888.00 | 4912.65 | 4895.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-21 14:15:00 | 4888.00 | 4912.65 | 4895.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 4888.00 | 4912.65 | 4895.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 15:15:00 | 4867.50 | 4912.65 | 4895.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 15:15:00 | 4867.50 | 4903.62 | 4892.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 09:15:00 | 4850.80 | 4903.62 | 4892.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 11:15:00 | 4860.90 | 4884.15 | 4885.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-24 13:15:00 | 4850.00 | 4874.12 | 4880.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 09:15:00 | 4860.35 | 4857.80 | 4870.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 09:15:00 | 4860.35 | 4857.80 | 4870.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 4860.35 | 4857.80 | 4870.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 09:30:00 | 4867.70 | 4857.80 | 4870.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 4861.00 | 4858.44 | 4869.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 10:45:00 | 4879.60 | 4858.44 | 4869.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 11:15:00 | 4860.00 | 4858.75 | 4868.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 11:45:00 | 4868.45 | 4858.75 | 4868.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 4852.85 | 4850.55 | 4860.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 14:30:00 | 4831.15 | 4847.62 | 4855.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 11:15:00 | 4890.15 | 4861.06 | 4859.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2024-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 11:15:00 | 4890.15 | 4861.06 | 4859.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 13:15:00 | 4909.05 | 4873.56 | 4865.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 14:15:00 | 4901.30 | 4909.61 | 4892.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-28 15:00:00 | 4901.30 | 4909.61 | 4892.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 09:15:00 | 4979.25 | 4922.00 | 4901.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-01 11:00:00 | 5009.40 | 4939.48 | 4911.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 13:15:00 | 5033.60 | 5029.13 | 4987.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 10:15:00 | 5010.00 | 5024.14 | 4998.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 11:00:00 | 5026.90 | 5024.69 | 5000.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 5036.30 | 5031.35 | 5012.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 09:15:00 | 5067.50 | 5031.51 | 5013.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 13:30:00 | 5056.90 | 5086.15 | 5084.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-10 09:15:00 | 5045.25 | 5075.41 | 5079.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 5045.25 | 5075.41 | 5079.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 10:15:00 | 5023.30 | 5064.99 | 5074.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 14:15:00 | 5042.85 | 5042.35 | 5059.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-10 15:00:00 | 5042.85 | 5042.35 | 5059.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 5012.75 | 5035.25 | 5052.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-11 10:30:00 | 5000.10 | 5028.20 | 5048.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 10:15:00 | 5104.00 | 5013.79 | 5023.40 | SL hit (close>static) qty=1.00 sl=5087.75 alert=retest2 |

### Cycle 11 — BUY (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 11:15:00 | 5125.85 | 5036.20 | 5032.71 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 14:15:00 | 4989.45 | 5035.98 | 5041.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-16 09:15:00 | 4979.95 | 5017.25 | 5031.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-18 14:15:00 | 4847.70 | 4830.07 | 4892.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-18 15:00:00 | 4847.70 | 4830.07 | 4892.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 12:15:00 | 4954.60 | 4844.65 | 4874.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-19 13:00:00 | 4954.60 | 4844.65 | 4874.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 13:15:00 | 4891.40 | 4854.00 | 4875.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 15:00:00 | 4884.00 | 4860.00 | 4876.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-22 09:15:00 | 4992.50 | 4887.36 | 4886.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-22 09:15:00 | 4992.50 | 4887.36 | 4886.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-22 10:15:00 | 5013.70 | 4912.63 | 4897.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 5118.65 | 5142.40 | 5089.03 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-25 12:00:00 | 5183.00 | 5152.53 | 5103.02 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 10:15:00 | 5176.30 | 5197.00 | 5173.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 10:30:00 | 5161.05 | 5197.00 | 5173.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 11:15:00 | 5204.30 | 5198.46 | 5176.37 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-30 09:15:00 | 5164.35 | 5194.88 | 5183.67 | SL hit (close<ema400) qty=1.00 sl=5183.67 alert=retest1 |

### Cycle 14 — SELL (started 2024-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 12:15:00 | 5154.95 | 5191.60 | 5195.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 13:15:00 | 5129.00 | 5179.08 | 5189.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-02 11:15:00 | 5155.00 | 5141.96 | 5163.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-02 11:15:00 | 5155.00 | 5141.96 | 5163.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 11:15:00 | 5155.00 | 5141.96 | 5163.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-02 11:30:00 | 5184.95 | 5141.96 | 5163.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-02 12:15:00 | 5143.05 | 5142.17 | 5161.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 13:15:00 | 5124.60 | 5142.17 | 5161.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 10:15:00 | 4868.37 | 5037.14 | 5102.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-06 09:15:00 | 4929.85 | 4918.30 | 5000.57 | SL hit (close>ema200) qty=0.50 sl=4918.30 alert=retest2 |

### Cycle 15 — BUY (started 2024-08-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 12:15:00 | 4918.05 | 4876.24 | 4875.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-14 13:15:00 | 4930.00 | 4887.00 | 4880.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-23 11:15:00 | 5475.00 | 5476.08 | 5429.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-23 12:00:00 | 5475.00 | 5476.08 | 5429.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 10:15:00 | 5477.90 | 5492.68 | 5473.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 10:45:00 | 5472.40 | 5492.68 | 5473.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 5476.25 | 5489.40 | 5473.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 12:00:00 | 5476.25 | 5489.40 | 5473.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 12:15:00 | 5494.80 | 5490.48 | 5475.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 12:30:00 | 5477.70 | 5490.48 | 5475.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 14:15:00 | 5492.75 | 5492.50 | 5479.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 15:00:00 | 5492.75 | 5492.50 | 5479.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 14:15:00 | 5742.95 | 5758.95 | 5732.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 09:30:00 | 5764.30 | 5755.16 | 5735.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 11:00:00 | 5752.65 | 5754.66 | 5736.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 11:30:00 | 5750.95 | 5758.13 | 5739.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 14:00:00 | 5753.05 | 5757.73 | 5742.90 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 5746.45 | 5755.48 | 5743.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 5746.45 | 5755.48 | 5743.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 5750.00 | 5754.38 | 5743.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 5670.15 | 5754.38 | 5743.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 5686.85 | 5740.87 | 5738.66 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-09-04 09:15:00 | 5686.85 | 5740.87 | 5738.66 | SL hit (close<static) qty=1.00 sl=5725.00 alert=retest2 |

### Cycle 16 — SELL (started 2024-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-04 10:15:00 | 5670.75 | 5726.85 | 5732.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 14:15:00 | 5623.70 | 5662.85 | 5678.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 5623.10 | 5605.12 | 5631.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 5623.10 | 5605.12 | 5631.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 5623.10 | 5605.12 | 5631.12 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2024-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 13:15:00 | 5698.00 | 5647.74 | 5644.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-11 10:15:00 | 5721.70 | 5675.54 | 5660.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-11 13:15:00 | 5684.30 | 5692.55 | 5673.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 14:00:00 | 5684.30 | 5692.55 | 5673.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 14:15:00 | 5672.80 | 5688.60 | 5673.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-11 15:00:00 | 5672.80 | 5688.60 | 5673.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 15:15:00 | 5692.75 | 5689.43 | 5674.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 09:15:00 | 5718.90 | 5689.43 | 5674.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-12 14:00:00 | 5697.00 | 5689.26 | 5680.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 09:30:00 | 5740.30 | 5730.68 | 5725.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-17 12:15:00 | 5686.90 | 5720.71 | 5722.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-09-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 12:15:00 | 5686.90 | 5720.71 | 5722.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 13:15:00 | 5665.00 | 5709.57 | 5717.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 5501.90 | 5491.71 | 5544.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-19 15:00:00 | 5501.90 | 5491.71 | 5544.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 5480.65 | 5493.23 | 5535.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 09:15:00 | 5456.00 | 5491.04 | 5517.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 10:15:00 | 5445.60 | 5485.87 | 5512.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 12:00:00 | 5452.40 | 5470.59 | 5500.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-24 10:00:00 | 5439.25 | 5473.25 | 5491.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 14:15:00 | 5495.35 | 5466.67 | 5479.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-24 14:45:00 | 5511.10 | 5466.67 | 5479.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 15:15:00 | 5510.00 | 5475.34 | 5482.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 09:15:00 | 5484.05 | 5475.34 | 5482.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 5487.80 | 5476.97 | 5481.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:45:00 | 5485.45 | 5476.97 | 5481.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 5442.20 | 5470.01 | 5478.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 12:45:00 | 5411.35 | 5451.54 | 5469.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 14:45:00 | 5415.20 | 5446.09 | 5463.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-25 15:15:00 | 5420.20 | 5446.09 | 5463.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-27 09:15:00 | 5512.40 | 5462.85 | 5460.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 09:15:00 | 5512.40 | 5462.85 | 5460.82 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-09-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 12:15:00 | 5400.40 | 5452.25 | 5456.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-27 13:15:00 | 5392.70 | 5440.34 | 5451.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 11:15:00 | 5359.00 | 5340.50 | 5370.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-01 12:00:00 | 5359.00 | 5340.50 | 5370.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 5392.65 | 5350.93 | 5372.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 13:00:00 | 5392.65 | 5350.93 | 5372.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 5377.85 | 5356.32 | 5373.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 14:15:00 | 5355.45 | 5356.32 | 5373.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-04 09:15:00 | 5087.68 | 5160.36 | 5239.20 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-04 10:15:00 | 5207.65 | 5169.82 | 5236.33 | SL hit (close>ema200) qty=0.50 sl=5169.82 alert=retest2 |

### Cycle 21 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 5200.00 | 5161.82 | 5159.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 11:15:00 | 5215.00 | 5172.46 | 5164.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 12:15:00 | 5205.80 | 5213.81 | 5195.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 12:15:00 | 5205.80 | 5213.81 | 5195.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 5205.80 | 5213.81 | 5195.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 12:45:00 | 5190.00 | 5213.81 | 5195.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 13:15:00 | 5180.05 | 5207.06 | 5194.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 13:30:00 | 5175.00 | 5207.06 | 5194.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 14:15:00 | 5209.80 | 5207.61 | 5195.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 15:15:00 | 5147.55 | 5207.61 | 5195.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 5147.55 | 5195.60 | 5191.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 09:15:00 | 5235.00 | 5195.60 | 5191.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 09:45:00 | 5217.70 | 5200.08 | 5193.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 10:30:00 | 5216.35 | 5204.06 | 5195.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 11:00:00 | 5220.00 | 5204.06 | 5195.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 5192.10 | 5201.67 | 5195.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:30:00 | 5185.35 | 5201.67 | 5195.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 12:15:00 | 5195.80 | 5200.50 | 5195.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-11 14:30:00 | 5206.00 | 5206.71 | 5199.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-18 09:15:00 | 5198.95 | 5276.63 | 5286.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2024-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-18 09:15:00 | 5198.95 | 5276.63 | 5286.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 12:15:00 | 5190.65 | 5236.77 | 5263.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 09:15:00 | 5245.15 | 5206.87 | 5238.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 09:15:00 | 5245.15 | 5206.87 | 5238.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 5245.15 | 5206.87 | 5238.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 09:30:00 | 5218.35 | 5206.87 | 5238.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 5272.65 | 5220.02 | 5241.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 10:30:00 | 5313.95 | 5220.02 | 5241.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 5262.05 | 5228.43 | 5243.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 12:00:00 | 5262.05 | 5228.43 | 5243.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 13:15:00 | 5278.55 | 5243.83 | 5247.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 14:00:00 | 5278.55 | 5243.83 | 5247.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 15:15:00 | 5255.00 | 5247.05 | 5248.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:15:00 | 5244.90 | 5247.05 | 5248.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 5208.95 | 5239.43 | 5245.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 10:15:00 | 5204.35 | 5239.43 | 5245.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 13:30:00 | 5198.70 | 5231.64 | 5240.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-23 10:15:00 | 5328.95 | 5235.50 | 5236.35 | SL hit (close>static) qty=1.00 sl=5282.15 alert=retest2 |

### Cycle 23 — BUY (started 2024-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 11:15:00 | 5399.95 | 5268.39 | 5251.22 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 5205.65 | 5268.37 | 5272.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 10:15:00 | 5164.85 | 5247.67 | 5262.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 11:15:00 | 5180.00 | 5176.80 | 5209.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-28 11:30:00 | 5176.90 | 5176.80 | 5209.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 5150.00 | 5164.09 | 5190.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 09:30:00 | 5168.85 | 5164.09 | 5190.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 13:15:00 | 5203.00 | 5165.64 | 5181.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:00:00 | 5203.00 | 5165.64 | 5181.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 14:15:00 | 5204.90 | 5173.49 | 5183.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-29 14:45:00 | 5212.35 | 5173.49 | 5183.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 5278.00 | 5196.66 | 5192.82 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-10-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-30 15:15:00 | 5152.95 | 5200.02 | 5200.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-31 09:15:00 | 4981.70 | 5156.36 | 5180.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-04 14:15:00 | 4965.40 | 4951.34 | 5003.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-04 15:00:00 | 4965.40 | 4951.34 | 5003.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 09:15:00 | 4941.55 | 4952.37 | 4995.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-05 11:30:00 | 4917.45 | 4944.82 | 4984.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 09:15:00 | 5059.55 | 4972.31 | 4982.10 | SL hit (close>static) qty=1.00 sl=4995.55 alert=retest2 |

### Cycle 27 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 5075.00 | 4992.85 | 4990.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 11:15:00 | 5132.50 | 5020.78 | 5003.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 11:15:00 | 5112.00 | 5125.75 | 5079.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 12:00:00 | 5112.00 | 5125.75 | 5079.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 14:15:00 | 5097.35 | 5150.57 | 5126.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 14:45:00 | 5100.35 | 5150.57 | 5126.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 15:15:00 | 5090.10 | 5138.47 | 5122.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 09:15:00 | 5056.75 | 5138.47 | 5122.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 10:15:00 | 5146.40 | 5136.14 | 5124.35 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-11-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 15:15:00 | 5100.25 | 5117.81 | 5119.21 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2024-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-12 09:15:00 | 5307.15 | 5155.68 | 5136.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-12 11:15:00 | 5355.90 | 5220.85 | 5170.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-13 09:15:00 | 5175.05 | 5240.38 | 5203.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-13 09:15:00 | 5175.05 | 5240.38 | 5203.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 09:15:00 | 5175.05 | 5240.38 | 5203.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 10:00:00 | 5175.05 | 5240.38 | 5203.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 10:15:00 | 5238.15 | 5239.93 | 5206.84 | EMA400 retest candle locked (from upside) |

### Cycle 30 — SELL (started 2024-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 10:15:00 | 5151.75 | 5196.57 | 5198.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-14 12:15:00 | 5141.95 | 5178.90 | 5189.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 15:15:00 | 5190.00 | 5171.55 | 5182.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 15:15:00 | 5190.00 | 5171.55 | 5182.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 15:15:00 | 5190.00 | 5171.55 | 5182.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 09:15:00 | 5049.60 | 5171.55 | 5182.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 09:15:00 | 5082.00 | 5153.64 | 5173.36 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2024-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 11:15:00 | 5185.20 | 5156.37 | 5155.48 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 15:15:00 | 5125.95 | 5149.91 | 5153.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-21 09:15:00 | 5077.80 | 5135.49 | 5146.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-21 10:15:00 | 5136.30 | 5135.65 | 5145.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-21 11:00:00 | 5136.30 | 5135.65 | 5145.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 11:15:00 | 5144.95 | 5137.51 | 5145.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 11:30:00 | 5164.20 | 5137.51 | 5145.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 12:15:00 | 5139.00 | 5137.81 | 5144.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 13:15:00 | 5161.65 | 5137.81 | 5144.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 13:15:00 | 5174.95 | 5145.24 | 5147.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-21 14:00:00 | 5174.95 | 5145.24 | 5147.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2024-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 14:15:00 | 5168.35 | 5149.86 | 5149.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 09:15:00 | 5224.80 | 5167.91 | 5157.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 09:15:00 | 5440.30 | 5467.47 | 5410.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 09:30:00 | 5426.30 | 5467.47 | 5410.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 15:15:00 | 5432.30 | 5450.39 | 5426.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 09:15:00 | 5363.05 | 5450.39 | 5426.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 5350.00 | 5430.31 | 5419.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:00:00 | 5350.00 | 5430.31 | 5419.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 10:15:00 | 5280.00 | 5400.25 | 5406.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 11:15:00 | 5256.30 | 5371.46 | 5393.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 14:15:00 | 5276.45 | 5264.94 | 5289.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-02 15:00:00 | 5276.45 | 5264.94 | 5289.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 5317.50 | 5277.53 | 5290.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 10:00:00 | 5317.50 | 5277.53 | 5290.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 5303.50 | 5282.73 | 5291.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 12:00:00 | 5289.25 | 5284.03 | 5291.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 15:15:00 | 5314.05 | 5298.46 | 5296.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2024-12-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 15:15:00 | 5314.05 | 5298.46 | 5296.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-04 09:15:00 | 5330.40 | 5304.85 | 5299.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-04 13:15:00 | 5298.80 | 5304.79 | 5301.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-04 13:15:00 | 5298.80 | 5304.79 | 5301.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 13:15:00 | 5298.80 | 5304.79 | 5301.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 09:15:00 | 5362.30 | 5301.41 | 5300.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-05 15:00:00 | 5342.35 | 5334.24 | 5320.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 15:15:00 | 5291.60 | 5321.08 | 5321.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-12-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 15:15:00 | 5291.60 | 5321.08 | 5321.35 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 09:15:00 | 5329.15 | 5322.70 | 5322.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-10 09:15:00 | 5400.00 | 5351.27 | 5338.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 14:15:00 | 5368.85 | 5373.06 | 5355.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-10 14:30:00 | 5360.25 | 5373.06 | 5355.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 11:15:00 | 5346.25 | 5371.29 | 5360.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 11:45:00 | 5350.00 | 5371.29 | 5360.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 12:15:00 | 5349.80 | 5366.99 | 5359.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-11 15:00:00 | 5369.00 | 5364.14 | 5359.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 10:15:00 | 5328.55 | 5380.72 | 5377.27 | SL hit (close<static) qty=1.00 sl=5342.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-12-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 10:15:00 | 5351.00 | 5376.55 | 5377.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 11:15:00 | 5343.05 | 5369.85 | 5374.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-27 10:15:00 | 4690.90 | 4687.54 | 4739.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-27 11:00:00 | 4690.90 | 4687.54 | 4739.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 4736.70 | 4697.20 | 4720.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:45:00 | 4726.20 | 4697.20 | 4720.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 4729.25 | 4703.61 | 4721.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:45:00 | 4727.50 | 4703.61 | 4721.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 11:15:00 | 4724.75 | 4707.84 | 4721.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 11:30:00 | 4739.70 | 4707.84 | 4721.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 13:15:00 | 4712.85 | 4715.34 | 4723.26 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2024-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 15:15:00 | 4755.00 | 4729.81 | 4728.87 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 09:15:00 | 4686.00 | 4721.05 | 4724.97 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 11:15:00 | 4745.80 | 4728.81 | 4727.99 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 12:15:00 | 4716.95 | 4726.44 | 4726.99 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2024-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 13:15:00 | 4752.00 | 4731.55 | 4729.26 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-01-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-01 11:15:00 | 4703.20 | 4726.58 | 4728.02 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 11:15:00 | 4749.10 | 4728.31 | 4725.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 13:15:00 | 4800.00 | 4748.40 | 4735.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 14:15:00 | 4793.25 | 4797.35 | 4774.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-03 15:00:00 | 4793.25 | 4797.35 | 4774.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 4770.00 | 4791.88 | 4773.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 09:30:00 | 4805.05 | 4797.49 | 4777.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 10:15:00 | 4750.75 | 4788.14 | 4775.44 | SL hit (close<static) qty=1.00 sl=4761.65 alert=retest2 |

### Cycle 46 — SELL (started 2025-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 15:15:00 | 4760.00 | 4769.02 | 4769.34 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 09:15:00 | 4853.15 | 4785.85 | 4776.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 13:15:00 | 4899.95 | 4854.15 | 4829.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-09 14:15:00 | 4890.95 | 4903.59 | 4875.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-09 15:00:00 | 4890.95 | 4903.59 | 4875.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 15:15:00 | 4894.00 | 4901.67 | 4877.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 09:15:00 | 4915.90 | 4901.67 | 4877.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 4875.40 | 4896.42 | 4877.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 4875.40 | 4896.42 | 4877.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 4949.35 | 4907.00 | 4883.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 11:15:00 | 4976.95 | 4907.00 | 4883.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 12:30:00 | 4951.50 | 4927.77 | 4897.99 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 13:45:00 | 4953.80 | 4930.58 | 4901.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-10 14:15:00 | 4952.05 | 4930.58 | 4901.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 4951.75 | 4934.24 | 4910.69 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-01-13 13:15:00 | 4839.50 | 4902.64 | 4902.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 48 — SELL (started 2025-01-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-13 13:15:00 | 4839.50 | 4902.64 | 4902.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 09:15:00 | 4793.00 | 4868.18 | 4885.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 10:15:00 | 4795.00 | 4772.73 | 4815.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-15 10:15:00 | 4795.00 | 4772.73 | 4815.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 4795.00 | 4772.73 | 4815.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:45:00 | 4807.05 | 4772.73 | 4815.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 4843.55 | 4786.90 | 4817.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 12:00:00 | 4843.55 | 4786.90 | 4817.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 12:15:00 | 4881.75 | 4805.87 | 4823.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 13:00:00 | 4881.75 | 4805.87 | 4823.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 15:15:00 | 4870.00 | 4831.04 | 4831.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 09:15:00 | 5288.35 | 4831.04 | 4831.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2025-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 09:15:00 | 5193.20 | 4903.47 | 4864.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-17 09:15:00 | 5340.35 | 5188.41 | 5054.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 09:15:00 | 5319.55 | 5323.44 | 5204.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-20 10:00:00 | 5319.55 | 5323.44 | 5204.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 09:15:00 | 5344.95 | 5382.34 | 5336.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 09:30:00 | 5300.95 | 5382.34 | 5336.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 10:15:00 | 5341.05 | 5374.08 | 5337.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 14:30:00 | 5366.95 | 5360.08 | 5340.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 09:15:00 | 5265.70 | 5401.72 | 5401.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 5265.70 | 5401.72 | 5401.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 13:15:00 | 5257.25 | 5328.82 | 5363.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-29 09:15:00 | 5288.05 | 5207.38 | 5259.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-29 09:15:00 | 5288.05 | 5207.38 | 5259.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 5288.05 | 5207.38 | 5259.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 5288.05 | 5207.38 | 5259.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 5255.60 | 5217.02 | 5259.07 | EMA400 retest candle locked (from downside) |

### Cycle 51 — BUY (started 2025-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 15:15:00 | 5331.50 | 5283.31 | 5278.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 5349.90 | 5296.63 | 5285.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-31 09:15:00 | 5342.70 | 5363.87 | 5331.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-31 10:00:00 | 5342.70 | 5363.87 | 5331.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 10:15:00 | 5365.80 | 5364.25 | 5334.74 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 11:45:00 | 5383.50 | 5368.76 | 5339.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 09:30:00 | 5406.55 | 5416.84 | 5394.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-03 10:00:00 | 5414.10 | 5416.84 | 5394.06 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 11:15:00 | 5484.50 | 5542.50 | 5544.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 11:15:00 | 5484.50 | 5542.50 | 5544.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 5413.55 | 5498.61 | 5520.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-18 09:15:00 | 4886.70 | 4862.10 | 4946.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-18 10:00:00 | 4886.70 | 4862.10 | 4946.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 13:15:00 | 4932.30 | 4894.97 | 4936.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 14:00:00 | 4932.30 | 4894.97 | 4936.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 4940.00 | 4903.98 | 4937.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-18 15:00:00 | 4940.00 | 4903.98 | 4937.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 15:15:00 | 4930.35 | 4909.25 | 4936.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:15:00 | 4915.15 | 4909.25 | 4936.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 4913.85 | 4910.17 | 4934.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 11:15:00 | 4881.35 | 4908.62 | 4931.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 12:30:00 | 4877.90 | 4897.13 | 4922.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-21 09:15:00 | 5070.00 | 4940.24 | 4925.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2025-02-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-21 09:15:00 | 5070.00 | 4940.24 | 4925.68 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-02-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-24 11:15:00 | 4856.35 | 4965.17 | 4970.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 10:15:00 | 4799.70 | 4883.70 | 4922.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 4563.00 | 4528.19 | 4598.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 4563.00 | 4528.19 | 4598.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 13:15:00 | 4560.00 | 4530.38 | 4563.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 14:00:00 | 4560.00 | 4530.38 | 4563.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 14:15:00 | 4566.70 | 4537.65 | 4563.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-04 14:45:00 | 4571.10 | 4537.65 | 4563.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-04 15:15:00 | 4541.90 | 4538.50 | 4561.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:15:00 | 4672.20 | 4538.50 | 4561.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 4687.40 | 4568.28 | 4573.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-05 09:45:00 | 4704.50 | 4568.28 | 4573.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 4728.00 | 4600.22 | 4587.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 4754.25 | 4678.81 | 4633.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 10:15:00 | 4806.80 | 4814.57 | 4754.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 10:45:00 | 4804.45 | 4814.57 | 4754.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 4761.50 | 4801.83 | 4773.57 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 13:15:00 | 4708.10 | 4759.92 | 4760.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 4660.70 | 4740.07 | 4751.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 14:15:00 | 4639.25 | 4633.58 | 4680.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 15:00:00 | 4639.25 | 4633.58 | 4680.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 14:15:00 | 4448.95 | 4415.02 | 4442.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-17 14:45:00 | 4455.95 | 4415.02 | 4442.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 15:15:00 | 4454.00 | 4422.82 | 4443.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:15:00 | 4528.10 | 4422.82 | 4443.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2025-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 10:15:00 | 4586.95 | 4481.39 | 4468.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 13:15:00 | 4607.50 | 4534.24 | 4497.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 15:15:00 | 4646.00 | 4646.50 | 4612.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 09:15:00 | 4632.50 | 4646.50 | 4612.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 4704.85 | 4658.17 | 4620.75 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2025-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-24 11:15:00 | 4567.05 | 4626.27 | 4628.33 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-25 14:15:00 | 4626.30 | 4621.89 | 4621.48 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 15:15:00 | 4581.00 | 4613.72 | 4617.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 09:15:00 | 4553.15 | 4601.60 | 4611.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 11:15:00 | 4561.85 | 4545.20 | 4568.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-27 11:45:00 | 4558.75 | 4545.20 | 4568.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 4562.00 | 4505.55 | 4536.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 09:45:00 | 4568.20 | 4505.55 | 4536.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 4514.50 | 4507.34 | 4534.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:30:00 | 4582.80 | 4507.34 | 4534.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 4531.65 | 4496.83 | 4514.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 10:15:00 | 4568.20 | 4496.83 | 4514.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 10:15:00 | 4525.00 | 4502.46 | 4515.46 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2025-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-01 12:15:00 | 4574.05 | 4528.68 | 4525.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 11:15:00 | 4647.85 | 4572.07 | 4550.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 09:15:00 | 4509.00 | 4588.80 | 4570.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-03 09:15:00 | 4509.00 | 4588.80 | 4570.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 4509.00 | 4588.80 | 4570.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 10:15:00 | 4519.35 | 4588.80 | 4570.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2025-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-03 11:15:00 | 4433.10 | 4538.01 | 4549.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 09:15:00 | 4290.75 | 4457.99 | 4503.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 4109.60 | 4106.45 | 4210.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 4109.60 | 4106.45 | 4210.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 4179.05 | 4096.09 | 4125.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 14:00:00 | 4117.90 | 4133.68 | 4137.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 15:15:00 | 4130.00 | 4133.94 | 4137.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 4283.40 | 4163.20 | 4150.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 4283.40 | 4163.20 | 4150.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 10:15:00 | 4311.90 | 4192.94 | 4164.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 10:15:00 | 4286.50 | 4303.66 | 4247.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 11:00:00 | 4286.50 | 4303.66 | 4247.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 12:15:00 | 4231.60 | 4284.41 | 4248.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 13:00:00 | 4231.60 | 4284.41 | 4248.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 13:15:00 | 4239.00 | 4275.32 | 4247.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-16 13:30:00 | 4229.20 | 4275.32 | 4247.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-16 15:15:00 | 4234.10 | 4260.00 | 4244.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 09:15:00 | 4150.10 | 4260.00 | 4244.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2025-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-17 10:15:00 | 4153.00 | 4218.76 | 4227.41 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2025-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-21 09:15:00 | 4262.60 | 4232.17 | 4230.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-21 10:15:00 | 4300.40 | 4245.81 | 4236.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-25 09:15:00 | 4250.20 | 4423.74 | 4419.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 09:15:00 | 4250.20 | 4423.74 | 4419.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 09:15:00 | 4250.20 | 4423.74 | 4419.03 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2025-04-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 10:15:00 | 4194.00 | 4377.79 | 4398.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 11:15:00 | 4155.10 | 4234.70 | 4299.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-29 12:15:00 | 4186.50 | 4181.25 | 4230.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-29 13:00:00 | 4186.50 | 4181.25 | 4230.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 13:15:00 | 4229.50 | 4190.90 | 4230.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 14:00:00 | 4229.50 | 4190.90 | 4230.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 4182.20 | 4189.16 | 4226.12 | EMA400 retest candle locked (from downside) |

### Cycle 67 — BUY (started 2025-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 15:15:00 | 4270.00 | 4235.96 | 4234.30 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-05-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-02 12:15:00 | 4220.00 | 4231.91 | 4233.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 14:15:00 | 4198.40 | 4222.68 | 4228.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 4229.60 | 4217.32 | 4224.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 4229.60 | 4217.32 | 4224.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 4229.60 | 4217.32 | 4224.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:30:00 | 4227.10 | 4217.32 | 4224.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 10:15:00 | 4217.20 | 4217.29 | 4224.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 11:30:00 | 4209.90 | 4216.39 | 4223.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 13:15:00 | 4215.00 | 4216.59 | 4222.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-05 15:00:00 | 4215.50 | 4216.14 | 4221.27 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 4004.25 | 4085.79 | 4111.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-09 09:15:00 | 4004.72 | 4085.79 | 4111.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-09 12:15:00 | 4098.50 | 4082.03 | 4102.89 | SL hit (close>ema200) qty=0.50 sl=4082.03 alert=retest2 |

### Cycle 69 — BUY (started 2025-05-12 09:15:00)

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

### Cycle 70 — SELL (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 09:15:00 | 4469.90 | 4486.14 | 4487.64 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 4573.30 | 4484.99 | 4477.27 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-05-27 10:15:00)

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

### Cycle 73 — BUY (started 2025-06-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-09 15:15:00 | 4334.50 | 4330.24 | 4329.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 09:15:00 | 4422.00 | 4348.59 | 4338.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 11:15:00 | 4531.80 | 4532.43 | 4483.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 12:00:00 | 4531.80 | 4532.43 | 4483.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 4456.20 | 4514.84 | 4493.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 4473.20 | 4514.84 | 4493.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 12:15:00 | 4428.10 | 4478.95 | 4480.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 12:15:00 | 4428.10 | 4478.95 | 4480.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 14:15:00 | 4421.10 | 4460.03 | 4471.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 09:15:00 | 4462.20 | 4458.49 | 4468.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-16 09:15:00 | 4462.20 | 4458.49 | 4468.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 4462.20 | 4458.49 | 4468.61 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2025-06-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 13:15:00 | 4534.00 | 4482.73 | 4477.04 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-06-17 12:15:00)

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

### Cycle 77 — BUY (started 2025-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 15:15:00 | 4365.00 | 4359.31 | 4358.74 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-06-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 09:15:00 | 4321.80 | 4351.81 | 4355.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 10:15:00 | 4311.20 | 4343.69 | 4351.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 09:15:00 | 4378.00 | 4337.96 | 4343.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 09:15:00 | 4378.00 | 4337.96 | 4343.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 4378.00 | 4337.96 | 4343.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:45:00 | 4357.80 | 4337.96 | 4343.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 4345.50 | 4339.47 | 4343.64 | EMA400 retest candle locked (from downside) |

### Cycle 79 — BUY (started 2025-06-27 12:15:00)

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

### Cycle 80 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 4325.20 | 4364.95 | 4365.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 13:15:00 | 4313.50 | 4346.67 | 4354.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 10:15:00 | 4348.60 | 4337.68 | 4346.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 10:15:00 | 4348.60 | 4337.68 | 4346.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 4348.60 | 4337.68 | 4346.85 | EMA400 retest candle locked (from downside) |

### Cycle 81 — BUY (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 11:15:00 | 4368.40 | 4348.59 | 4347.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 12:15:00 | 4379.10 | 4354.69 | 4350.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 11:15:00 | 4362.40 | 4368.25 | 4361.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 11:15:00 | 4362.40 | 4368.25 | 4361.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 11:15:00 | 4362.40 | 4368.25 | 4361.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 12:00:00 | 4362.40 | 4368.25 | 4361.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 4368.10 | 4368.22 | 4361.76 | EMA400 retest candle locked (from upside) |

### Cycle 82 — SELL (started 2025-07-08 14:15:00)

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

### Cycle 83 — BUY (started 2025-07-11 09:15:00)

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

### Cycle 84 — SELL (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 14:15:00 | 4339.90 | 4347.00 | 4347.77 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-07-15 10:15:00)

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

### Cycle 86 — SELL (started 2025-07-21 10:15:00)

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

### Cycle 87 — BUY (started 2025-07-30 10:15:00)

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

### Cycle 88 — SELL (started 2025-08-04 09:15:00)

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

### Cycle 89 — BUY (started 2025-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 11:15:00 | 4216.70 | 4196.59 | 4194.49 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-08-13 13:15:00)

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

### Cycle 91 — BUY (started 2025-08-19 11:15:00)

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

### Cycle 92 — SELL (started 2025-08-28 13:15:00)

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

### Cycle 93 — BUY (started 2025-09-10 09:15:00)

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

### Cycle 94 — SELL (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 10:15:00 | 4239.90 | 4315.12 | 4321.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 4210.70 | 4273.39 | 4298.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 15:15:00 | 4214.00 | 4207.47 | 4241.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 10:00:00 | 4211.60 | 4208.29 | 4239.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 4062.70 | 4073.43 | 4112.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 13:30:00 | 4031.00 | 4049.18 | 4088.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 15:15:00 | 4097.00 | 4083.47 | 4083.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-09-30 15:15:00)

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

### Cycle 96 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 4219.10 | 4261.92 | 4262.33 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 13:15:00 | 4266.50 | 4260.76 | 4260.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 14:15:00 | 4274.60 | 4263.53 | 4261.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 09:15:00 | 4239.10 | 4289.36 | 4282.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 09:15:00 | 4239.10 | 4289.36 | 4282.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 4239.10 | 4289.36 | 4282.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:00:00 | 4239.10 | 4289.36 | 4282.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-10-13 10:15:00)

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

### Cycle 99 — BUY (started 2025-10-20 11:15:00)

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

### Cycle 100 — SELL (started 2025-10-24 11:15:00)

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

### Cycle 101 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 4213.60 | 4188.28 | 4188.27 | EMA200 above EMA400 |

### Cycle 102 — SELL (started 2025-10-28 09:15:00)

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

### Cycle 103 — BUY (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 15:15:00 | 4117.00 | 4114.70 | 4114.59 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-11-04 09:15:00)

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

### Cycle 105 — BUY (started 2025-11-10 14:15:00)

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

### Cycle 106 — SELL (started 2025-11-14 11:15:00)

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

### Cycle 107 — BUY (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 10:15:00 | 4272.00 | 4133.33 | 4118.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 11:15:00 | 4376.20 | 4181.91 | 4141.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 15:15:00 | 4321.90 | 4345.23 | 4290.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 15:15:00 | 4321.90 | 4345.23 | 4290.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 4321.90 | 4345.23 | 4290.58 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2025-11-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 11:15:00 | 4254.50 | 4282.54 | 4283.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 12:15:00 | 4242.90 | 4274.61 | 4279.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 13:15:00 | 4310.00 | 4281.69 | 4282.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 13:15:00 | 4310.00 | 4281.69 | 4282.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 4310.00 | 4281.69 | 4282.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:00:00 | 4310.00 | 4281.69 | 4282.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 14:15:00 | 4326.20 | 4290.59 | 4286.61 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-11-25 09:15:00)

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

### Cycle 111 — BUY (started 2025-11-27 09:15:00)

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

### Cycle 112 — SELL (started 2025-12-08 11:15:00)

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

### Cycle 113 — BUY (started 2025-12-10 14:15:00)

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

### Cycle 114 — SELL (started 2025-12-16 09:15:00)

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

### Cycle 115 — BUY (started 2025-12-22 10:15:00)

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

### Cycle 116 — SELL (started 2025-12-26 10:15:00)

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

### Cycle 117 — BUY (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 09:15:00 | 4388.70 | 4374.44 | 4373.83 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 4357.30 | 4371.01 | 4372.33 | EMA200 below EMA400 |

### Cycle 119 — BUY (started 2026-01-08 12:15:00)

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

### Cycle 120 — SELL (started 2026-01-09 14:15:00)

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

### Cycle 121 — BUY (started 2026-02-03 10:15:00)

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

### Cycle 122 — SELL (started 2026-02-11 12:15:00)

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

### Cycle 123 — BUY (started 2026-02-26 15:15:00)

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

### Cycle 124 — SELL (started 2026-03-04 12:15:00)

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

### Cycle 125 — BUY (started 2026-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-13 10:15:00 | 3365.00 | 3181.79 | 3158.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-13 14:15:00 | 3441.40 | 3293.21 | 3223.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-16 09:15:00 | 3274.70 | 3317.79 | 3248.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 3274.70 | 3317.79 | 3248.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 3274.70 | 3317.79 | 3248.90 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2026-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 09:15:00 | 3272.00 | 3322.83 | 3329.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 10:15:00 | 3265.60 | 3311.38 | 3323.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 3162.30 | 3122.49 | 3174.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 13:00:00 | 3162.30 | 3122.49 | 3174.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 3185.30 | 3146.66 | 3170.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-25 12:45:00 | 3150.30 | 3154.35 | 3168.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 09:15:00 | 3196.80 | 3178.58 | 3177.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 127 — BUY (started 2026-03-27 09:15:00)

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

### Cycle 128 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 3123.70 | 3183.63 | 3184.15 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2026-04-01 09:15:00)

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

### Cycle 130 — SELL (started 2026-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 14:15:00 | 3332.90 | 3345.79 | 3347.04 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-04-10 09:15:00)

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

### Cycle 132 — SELL (started 2026-04-23 09:15:00)

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

### Cycle 133 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 3562.00 | 3477.06 | 3473.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-27 13:15:00 | 3572.90 | 3496.23 | 3482.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 12:15:00 | 3558.00 | 3566.84 | 3546.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-29 13:00:00 | 3558.00 | 3566.84 | 3546.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 3556.00 | 3564.91 | 3551.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 3549.00 | 3564.91 | 3551.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 3522.00 | 3556.33 | 3548.46 | EMA400 retest candle locked (from upside) |

### Cycle 134 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 3483.20 | 3541.70 | 3542.53 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 3595.90 | 3544.81 | 3542.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 14:15:00 | 3627.50 | 3561.35 | 3550.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 10:15:00 | 3777.20 | 3777.24 | 3738.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 10:45:00 | 3783.00 | 3777.24 | 3738.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-16 09:15:00 | 4529.40 | 2024-05-17 15:15:00 | 4450.00 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2024-05-28 10:00:00 | 4644.00 | 2024-05-30 09:15:00 | 4590.15 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-05-28 12:45:00 | 4648.15 | 2024-05-30 09:15:00 | 4590.15 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-05-28 14:30:00 | 4644.90 | 2024-05-30 09:15:00 | 4590.15 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2024-05-28 15:00:00 | 4647.15 | 2024-05-30 12:15:00 | 4588.00 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-05-29 10:15:00 | 4679.60 | 2024-05-30 12:15:00 | 4588.00 | STOP_HIT | 1.00 | -1.96% |
| BUY | retest2 | 2024-05-29 11:00:00 | 4672.00 | 2024-05-30 12:15:00 | 4588.00 | STOP_HIT | 1.00 | -1.80% |
| BUY | retest2 | 2024-05-29 11:30:00 | 4671.00 | 2024-05-30 12:15:00 | 4588.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2024-06-03 10:30:00 | 4478.55 | 2024-06-04 09:15:00 | 4254.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 12:30:00 | 4481.95 | 2024-06-04 09:15:00 | 4257.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 14:15:00 | 4488.70 | 2024-06-04 09:15:00 | 4264.26 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 14:45:00 | 4489.80 | 2024-06-04 09:15:00 | 4265.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 10:30:00 | 4478.55 | 2024-06-04 12:15:00 | 4436.10 | STOP_HIT | 0.50 | 0.95% |
| SELL | retest2 | 2024-06-03 12:30:00 | 4481.95 | 2024-06-04 12:15:00 | 4436.10 | STOP_HIT | 0.50 | 1.02% |
| SELL | retest2 | 2024-06-03 14:15:00 | 4488.70 | 2024-06-04 12:15:00 | 4436.10 | STOP_HIT | 0.50 | 1.17% |
| SELL | retest2 | 2024-06-03 14:45:00 | 4489.80 | 2024-06-04 12:15:00 | 4436.10 | STOP_HIT | 0.50 | 1.20% |
| SELL | retest2 | 2024-06-19 10:15:00 | 4820.25 | 2024-06-20 11:15:00 | 4862.30 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2024-06-26 14:30:00 | 4831.15 | 2024-06-27 11:15:00 | 4890.15 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-07-01 11:00:00 | 5009.40 | 2024-07-10 09:15:00 | 5045.25 | STOP_HIT | 1.00 | 0.72% |
| BUY | retest2 | 2024-07-02 13:15:00 | 5033.60 | 2024-07-10 09:15:00 | 5045.25 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2024-07-03 10:15:00 | 5010.00 | 2024-07-10 09:15:00 | 5045.25 | STOP_HIT | 1.00 | 0.70% |
| BUY | retest2 | 2024-07-03 11:00:00 | 5026.90 | 2024-07-10 09:15:00 | 5045.25 | STOP_HIT | 1.00 | 0.37% |
| BUY | retest2 | 2024-07-04 09:15:00 | 5067.50 | 2024-07-10 09:15:00 | 5045.25 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2024-07-09 13:30:00 | 5056.90 | 2024-07-10 09:15:00 | 5045.25 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2024-07-11 10:30:00 | 5000.10 | 2024-07-12 10:15:00 | 5104.00 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-07-19 15:00:00 | 4884.00 | 2024-07-22 09:15:00 | 4992.50 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest1 | 2024-07-25 12:00:00 | 5183.00 | 2024-07-30 09:15:00 | 5164.35 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2024-08-01 09:15:00 | 5229.30 | 2024-08-01 11:15:00 | 5158.75 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2024-08-02 13:15:00 | 5124.60 | 2024-08-05 10:15:00 | 4868.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-02 13:15:00 | 5124.60 | 2024-08-06 09:15:00 | 4929.85 | STOP_HIT | 0.50 | 3.80% |
| BUY | retest2 | 2024-09-03 09:30:00 | 5764.30 | 2024-09-04 09:15:00 | 5686.85 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-09-03 11:00:00 | 5752.65 | 2024-09-04 09:15:00 | 5686.85 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-09-03 11:30:00 | 5750.95 | 2024-09-04 09:15:00 | 5686.85 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2024-09-03 14:00:00 | 5753.05 | 2024-09-04 09:15:00 | 5686.85 | STOP_HIT | 1.00 | -1.15% |
| BUY | retest2 | 2024-09-12 09:15:00 | 5718.90 | 2024-09-17 12:15:00 | 5686.90 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2024-09-12 14:00:00 | 5697.00 | 2024-09-17 12:15:00 | 5686.90 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2024-09-17 09:30:00 | 5740.30 | 2024-09-17 12:15:00 | 5686.90 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2024-09-23 09:15:00 | 5456.00 | 2024-09-27 09:15:00 | 5512.40 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-09-23 10:15:00 | 5445.60 | 2024-09-27 09:15:00 | 5512.40 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-09-23 12:00:00 | 5452.40 | 2024-09-27 09:15:00 | 5512.40 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-09-24 10:00:00 | 5439.25 | 2024-09-27 09:15:00 | 5512.40 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2024-09-25 12:45:00 | 5411.35 | 2024-09-27 09:15:00 | 5512.40 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2024-09-25 14:45:00 | 5415.20 | 2024-09-27 09:15:00 | 5512.40 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2024-09-25 15:15:00 | 5420.20 | 2024-09-27 09:15:00 | 5512.40 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2024-10-01 14:15:00 | 5355.45 | 2024-10-04 09:15:00 | 5087.68 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 14:15:00 | 5355.45 | 2024-10-04 10:15:00 | 5207.65 | STOP_HIT | 0.50 | 2.76% |
| BUY | retest2 | 2024-10-11 09:15:00 | 5235.00 | 2024-10-18 09:15:00 | 5198.95 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-10-11 09:45:00 | 5217.70 | 2024-10-18 09:15:00 | 5198.95 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2024-10-11 10:30:00 | 5216.35 | 2024-10-18 09:15:00 | 5198.95 | STOP_HIT | 1.00 | -0.33% |
| BUY | retest2 | 2024-10-11 11:00:00 | 5220.00 | 2024-10-18 09:15:00 | 5198.95 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2024-10-11 14:30:00 | 5206.00 | 2024-10-18 09:15:00 | 5198.95 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2024-10-22 10:15:00 | 5204.35 | 2024-10-23 10:15:00 | 5328.95 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2024-10-22 13:30:00 | 5198.70 | 2024-10-23 10:15:00 | 5328.95 | STOP_HIT | 1.00 | -2.51% |
| SELL | retest2 | 2024-11-05 11:30:00 | 4917.45 | 2024-11-06 09:15:00 | 5059.55 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2024-12-03 12:00:00 | 5289.25 | 2024-12-03 15:15:00 | 5314.05 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-12-05 09:15:00 | 5362.30 | 2024-12-06 15:15:00 | 5291.60 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-12-05 15:00:00 | 5342.35 | 2024-12-06 15:15:00 | 5291.60 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-12-11 15:00:00 | 5369.00 | 2024-12-13 10:15:00 | 5328.55 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-12-13 12:00:00 | 5356.35 | 2024-12-16 10:15:00 | 5351.00 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2025-01-06 09:30:00 | 4805.05 | 2025-01-06 10:15:00 | 4750.75 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-01-10 11:15:00 | 4976.95 | 2025-01-13 13:15:00 | 4839.50 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2025-01-10 12:30:00 | 4951.50 | 2025-01-13 13:15:00 | 4839.50 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-01-10 13:45:00 | 4953.80 | 2025-01-13 13:15:00 | 4839.50 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-01-10 14:15:00 | 4952.05 | 2025-01-13 13:15:00 | 4839.50 | STOP_HIT | 1.00 | -2.27% |
| BUY | retest2 | 2025-01-22 14:30:00 | 5366.95 | 2025-01-27 09:15:00 | 5265.70 | STOP_HIT | 1.00 | -1.89% |
| BUY | retest2 | 2025-01-31 11:45:00 | 5383.50 | 2025-02-10 11:15:00 | 5484.50 | STOP_HIT | 1.00 | 1.88% |
| BUY | retest2 | 2025-02-03 09:30:00 | 5406.55 | 2025-02-10 11:15:00 | 5484.50 | STOP_HIT | 1.00 | 1.44% |
| BUY | retest2 | 2025-02-03 10:00:00 | 5414.10 | 2025-02-10 11:15:00 | 5484.50 | STOP_HIT | 1.00 | 1.30% |
| SELL | retest2 | 2025-02-19 11:15:00 | 4881.35 | 2025-02-21 09:15:00 | 5070.00 | STOP_HIT | 1.00 | -3.86% |
| SELL | retest2 | 2025-02-19 12:30:00 | 4877.90 | 2025-02-21 09:15:00 | 5070.00 | STOP_HIT | 1.00 | -3.94% |
| SELL | retest2 | 2025-04-11 14:00:00 | 4117.90 | 2025-04-15 09:15:00 | 4283.40 | STOP_HIT | 1.00 | -4.02% |
| SELL | retest2 | 2025-04-11 15:15:00 | 4130.00 | 2025-04-15 09:15:00 | 4283.40 | STOP_HIT | 1.00 | -3.71% |
| SELL | retest2 | 2025-05-05 11:30:00 | 4209.90 | 2025-05-09 09:15:00 | 4004.25 | PARTIAL | 0.50 | 4.88% |
| SELL | retest2 | 2025-05-05 13:15:00 | 4215.00 | 2025-05-09 09:15:00 | 4004.72 | PARTIAL | 0.50 | 4.99% |
| SELL | retest2 | 2025-05-05 11:30:00 | 4209.90 | 2025-05-09 12:15:00 | 4098.50 | STOP_HIT | 0.50 | 2.65% |
| SELL | retest2 | 2025-05-05 13:15:00 | 4215.00 | 2025-05-09 12:15:00 | 4098.50 | STOP_HIT | 0.50 | 2.76% |
| SELL | retest2 | 2025-05-05 15:00:00 | 4215.50 | 2025-05-12 09:15:00 | 4276.30 | STOP_HIT | 1.00 | -1.44% |
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
