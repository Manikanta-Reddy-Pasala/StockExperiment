# Eicher Motors Ltd. (EICHERMOT)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 7309.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 150 |
| ALERT1 | 102 |
| ALERT2 | 102 |
| ALERT2_SKIP | 58 |
| ALERT3 | 307 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 109 |
| PARTIAL | 9 |
| TARGET_HIT | 3 |
| STOP_HIT | 109 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 121 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 44 / 77
- **Target hits / Stop hits / Partials:** 3 / 109 / 9
- **Avg / median % per leg:** 0.47% / -0.45%
- **Sum % (uncompounded):** 56.71%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 51 | 22 | 43.1% | 1 | 50 | 0 | 0.49% | 25.2% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | 1.02% | 2.0% |
| BUY @ 3rd Alert (retest2) | 49 | 21 | 42.9% | 1 | 48 | 0 | 0.47% | 23.1% |
| SELL (all) | 70 | 22 | 31.4% | 2 | 59 | 9 | 0.45% | 31.6% |
| SELL @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -1.01% | -3.0% |
| SELL @ 3rd Alert (retest2) | 67 | 22 | 32.8% | 2 | 56 | 9 | 0.52% | 34.6% |
| retest1 (combined) | 5 | 1 | 20.0% | 0 | 5 | 0 | -0.20% | -1.0% |
| retest2 (combined) | 116 | 43 | 37.1% | 3 | 104 | 9 | 0.50% | 57.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 15:15:00 | 4648.00 | 4658.13 | 4659.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 09:15:00 | 4609.90 | 4648.49 | 4654.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 14:15:00 | 4689.05 | 4625.65 | 4637.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 14:15:00 | 4689.05 | 4625.65 | 4637.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 14:15:00 | 4689.05 | 4625.65 | 4637.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 14:45:00 | 4688.30 | 4625.65 | 4637.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 15:15:00 | 4663.05 | 4633.13 | 4639.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 09:15:00 | 4675.05 | 4633.13 | 4639.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 10:15:00 | 4673.95 | 4641.70 | 4642.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 11:00:00 | 4673.95 | 4641.70 | 4642.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-05-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 11:15:00 | 4682.75 | 4649.91 | 4645.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 12:15:00 | 4692.05 | 4658.34 | 4650.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 13:15:00 | 4661.60 | 4684.13 | 4674.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 13:15:00 | 4661.60 | 4684.13 | 4674.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 4661.60 | 4684.13 | 4674.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 14:00:00 | 4661.60 | 4684.13 | 4674.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 4678.75 | 4683.05 | 4675.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:15:00 | 4690.25 | 4680.68 | 4674.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:45:00 | 4692.10 | 4681.10 | 4675.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 10:30:00 | 4688.15 | 4683.69 | 4677.22 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 11:00:00 | 4694.05 | 4683.69 | 4677.22 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 12:15:00 | 4687.05 | 4684.87 | 4678.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 12:45:00 | 4682.50 | 4684.87 | 4678.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 13:15:00 | 4679.55 | 4683.81 | 4678.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:00:00 | 4679.55 | 4683.81 | 4678.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 14:15:00 | 4694.00 | 4685.84 | 4680.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 14:45:00 | 4675.30 | 4685.84 | 4680.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 4757.40 | 4701.14 | 4688.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 10:45:00 | 4765.20 | 4712.91 | 4694.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-23 11:15:00 | 4766.95 | 4712.91 | 4694.82 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-28 14:15:00 | 4769.10 | 4795.46 | 4798.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-05-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 14:15:00 | 4769.10 | 4795.46 | 4798.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 15:15:00 | 4765.00 | 4789.37 | 4795.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 4790.00 | 4789.49 | 4794.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 10:00:00 | 4790.00 | 4789.49 | 4794.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 4773.20 | 4786.23 | 4792.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:45:00 | 4765.20 | 4786.23 | 4792.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 12:15:00 | 4756.65 | 4748.39 | 4765.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-30 13:00:00 | 4756.65 | 4748.39 | 4765.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 15:15:00 | 4760.00 | 4749.04 | 4761.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 09:45:00 | 4719.60 | 4744.05 | 4757.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 10:45:00 | 4723.50 | 4739.17 | 4754.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 09:15:00 | 4709.90 | 4726.38 | 4740.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 11:15:00 | 4483.62 | 4598.63 | 4660.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 11:15:00 | 4487.32 | 4598.63 | 4660.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-04 11:15:00 | 4474.40 | 4598.63 | 4660.94 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-04 15:15:00 | 4548.95 | 4548.54 | 4613.03 | SL hit (close>ema200) qty=0.50 sl=4548.54 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 12:15:00 | 4670.20 | 4616.59 | 4613.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 14:15:00 | 4724.25 | 4649.39 | 4629.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-11 11:15:00 | 4749.65 | 4771.71 | 4744.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-11 11:15:00 | 4749.65 | 4771.71 | 4744.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 11:15:00 | 4749.65 | 4771.71 | 4744.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 12:00:00 | 4749.65 | 4771.71 | 4744.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 14:15:00 | 4759.60 | 4768.84 | 4749.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-11 15:00:00 | 4759.60 | 4768.84 | 4749.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 09:15:00 | 4777.00 | 4768.59 | 4752.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-12 10:45:00 | 4785.75 | 4773.25 | 4756.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 13:00:00 | 4792.30 | 4806.88 | 4792.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 12:15:00 | 4871.60 | 4886.73 | 4887.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 12:15:00 | 4871.60 | 4886.73 | 4887.59 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 10:15:00 | 4913.00 | 4890.06 | 4888.01 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 14:15:00 | 4837.50 | 4885.01 | 4887.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 15:15:00 | 4830.30 | 4874.07 | 4882.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 11:15:00 | 4872.00 | 4861.19 | 4873.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 11:15:00 | 4872.00 | 4861.19 | 4873.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 4872.00 | 4861.19 | 4873.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:45:00 | 4868.45 | 4861.19 | 4873.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 4874.65 | 4863.88 | 4873.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 13:00:00 | 4874.65 | 4863.88 | 4873.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 13:15:00 | 4869.00 | 4864.91 | 4872.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 13:30:00 | 4882.45 | 4864.91 | 4872.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 14:15:00 | 4874.30 | 4866.79 | 4872.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 15:00:00 | 4874.30 | 4866.79 | 4872.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 15:15:00 | 4875.00 | 4868.43 | 4873.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 09:15:00 | 4865.50 | 4868.43 | 4873.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 4859.70 | 4866.68 | 4871.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 11:00:00 | 4827.50 | 4858.85 | 4867.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-03 13:15:00 | 4686.00 | 4666.25 | 4665.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2024-07-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 13:15:00 | 4686.00 | 4666.25 | 4665.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 14:15:00 | 4694.00 | 4671.80 | 4668.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 09:15:00 | 4669.60 | 4675.87 | 4671.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 09:15:00 | 4669.60 | 4675.87 | 4671.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 4669.60 | 4675.87 | 4671.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:30:00 | 4672.85 | 4675.87 | 4671.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 10:15:00 | 4688.30 | 4678.36 | 4672.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 11:45:00 | 4706.10 | 4683.97 | 4675.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 12:45:00 | 4698.30 | 4690.46 | 4679.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 14:00:00 | 4704.20 | 4693.21 | 4681.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 09:15:00 | 4710.95 | 4690.69 | 4682.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 13:15:00 | 4730.00 | 4736.91 | 4721.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 13:45:00 | 4720.95 | 4736.91 | 4721.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-08 14:15:00 | 4742.20 | 4737.97 | 4723.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-08 14:30:00 | 4725.40 | 4737.97 | 4723.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 11:15:00 | 4830.00 | 4837.33 | 4813.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:00:00 | 4830.00 | 4837.33 | 4813.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 12:15:00 | 4815.00 | 4832.87 | 4813.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 12:30:00 | 4815.35 | 4832.87 | 4813.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 13:15:00 | 4822.90 | 4830.87 | 4814.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 13:30:00 | 4813.15 | 4830.87 | 4814.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 14:15:00 | 4827.05 | 4830.11 | 4815.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 14:45:00 | 4823.50 | 4830.11 | 4815.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 4830.10 | 4833.30 | 4819.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 12:15:00 | 4855.55 | 4836.02 | 4823.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 13:00:00 | 4865.00 | 4841.82 | 4827.10 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-15 10:00:00 | 4859.95 | 4851.60 | 4837.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 09:30:00 | 4865.45 | 4907.87 | 4889.22 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 4854.35 | 4897.17 | 4886.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 4854.35 | 4897.17 | 4886.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 4872.00 | 4892.13 | 4884.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-18 13:00:00 | 4890.95 | 4891.90 | 4885.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 09:15:00 | 4830.05 | 4894.02 | 4889.90 | SL hit (close<static) qty=1.00 sl=4845.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-07-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 10:15:00 | 4846.00 | 4884.42 | 4885.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 09:15:00 | 4807.40 | 4852.79 | 4867.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 13:15:00 | 4851.15 | 4845.85 | 4859.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 14:00:00 | 4851.15 | 4845.85 | 4859.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 4854.55 | 4847.59 | 4858.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 15:00:00 | 4854.55 | 4847.59 | 4858.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 4850.00 | 4848.07 | 4857.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 4943.75 | 4848.07 | 4857.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 4896.15 | 4857.69 | 4861.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:30:00 | 4944.95 | 4857.69 | 4861.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2024-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 10:15:00 | 4910.70 | 4868.29 | 4865.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 09:15:00 | 5020.30 | 4926.93 | 4908.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 09:15:00 | 4974.00 | 4984.11 | 4953.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 09:30:00 | 4969.15 | 4984.11 | 4953.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 10:15:00 | 4956.75 | 4978.64 | 4953.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 10:30:00 | 4945.60 | 4978.64 | 4953.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 12:15:00 | 4951.15 | 4970.80 | 4954.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 12:30:00 | 4945.25 | 4970.80 | 4954.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 13:15:00 | 4966.25 | 4969.89 | 4955.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 14:15:00 | 4970.00 | 4969.89 | 4955.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-30 09:15:00 | 4935.75 | 4961.46 | 4955.06 | SL hit (close<static) qty=1.00 sl=4945.00 alert=retest2 |

### Cycle 11 — SELL (started 2024-07-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 13:15:00 | 4923.95 | 4947.71 | 4950.24 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2024-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 09:15:00 | 4984.05 | 4956.04 | 4953.51 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 10:15:00 | 4926.40 | 4953.45 | 4955.44 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-08-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-01 15:15:00 | 4963.65 | 4955.25 | 4955.05 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 4812.00 | 4926.60 | 4942.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 11:15:00 | 4766.05 | 4875.36 | 4914.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 4720.40 | 4686.92 | 4749.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 09:15:00 | 4720.40 | 4686.92 | 4749.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 09:15:00 | 4720.40 | 4686.92 | 4749.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 09:45:00 | 4738.55 | 4686.92 | 4749.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 12:15:00 | 4592.20 | 4603.53 | 4639.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 12:30:00 | 4629.55 | 4603.53 | 4639.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 09:15:00 | 4773.80 | 4629.08 | 4639.07 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2024-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 10:15:00 | 4785.90 | 4660.44 | 4652.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-09 11:15:00 | 4810.00 | 4690.35 | 4666.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 13:15:00 | 4802.60 | 4813.22 | 4786.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 14:00:00 | 4802.60 | 4813.22 | 4786.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 4752.00 | 4797.81 | 4786.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 10:00:00 | 4752.00 | 4797.81 | 4786.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 10:15:00 | 4760.00 | 4790.25 | 4783.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 10:30:00 | 4768.35 | 4790.25 | 4783.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — SELL (started 2024-08-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 11:15:00 | 4721.05 | 4776.41 | 4778.06 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2024-08-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 12:15:00 | 4794.75 | 4770.22 | 4769.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 13:15:00 | 4808.00 | 4777.77 | 4773.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 10:15:00 | 4793.80 | 4793.93 | 4783.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-19 10:15:00 | 4793.80 | 4793.93 | 4783.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-19 10:15:00 | 4793.80 | 4793.93 | 4783.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-19 10:30:00 | 4794.20 | 4793.93 | 4783.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 4910.10 | 4928.88 | 4913.73 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2024-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 10:15:00 | 4880.35 | 4905.31 | 4906.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 09:15:00 | 4849.50 | 4876.89 | 4889.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 14:15:00 | 4869.60 | 4864.97 | 4877.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-27 15:00:00 | 4869.60 | 4864.97 | 4877.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 4885.00 | 4868.97 | 4878.58 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2024-08-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 11:15:00 | 4941.15 | 4891.42 | 4886.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-28 12:15:00 | 4946.00 | 4902.33 | 4892.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-29 10:15:00 | 4928.00 | 4930.83 | 4912.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-29 11:00:00 | 4928.00 | 4930.83 | 4912.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 4888.90 | 4922.44 | 4910.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-29 12:00:00 | 4888.90 | 4922.44 | 4910.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 12:15:00 | 4898.15 | 4917.58 | 4909.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-30 09:15:00 | 4936.35 | 4910.89 | 4907.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 12:15:00 | 4921.95 | 4934.90 | 4927.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-02 14:00:00 | 4923.65 | 4931.07 | 4927.19 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 10:00:00 | 4925.25 | 4925.26 | 4925.03 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 10:15:00 | 4916.00 | 4923.41 | 4924.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2024-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 10:15:00 | 4916.00 | 4923.41 | 4924.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 13:15:00 | 4898.25 | 4915.22 | 4920.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 12:15:00 | 4746.00 | 4741.96 | 4762.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-10 13:00:00 | 4746.00 | 4741.96 | 4762.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 10:15:00 | 4748.50 | 4740.35 | 4753.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-11 11:00:00 | 4748.50 | 4740.35 | 4753.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-11 11:15:00 | 4753.30 | 4742.94 | 4753.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 13:15:00 | 4740.00 | 4744.42 | 4753.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-11 14:00:00 | 4725.95 | 4740.73 | 4751.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-12 10:00:00 | 4728.05 | 4732.31 | 4744.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 13:15:00 | 4814.60 | 4754.89 | 4751.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2024-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 13:15:00 | 4814.60 | 4754.89 | 4751.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 14:15:00 | 4886.00 | 4781.11 | 4763.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 4847.45 | 4886.30 | 4860.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-17 09:15:00 | 4847.45 | 4886.30 | 4860.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 4847.45 | 4886.30 | 4860.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 09:45:00 | 4846.75 | 4886.30 | 4860.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 10:15:00 | 4843.40 | 4877.72 | 4859.00 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2024-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 10:15:00 | 4837.50 | 4853.81 | 4853.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-18 13:15:00 | 4825.60 | 4844.56 | 4849.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-18 14:15:00 | 4851.00 | 4845.85 | 4849.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 14:15:00 | 4851.00 | 4845.85 | 4849.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 14:15:00 | 4851.00 | 4845.85 | 4849.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-18 15:00:00 | 4851.00 | 4845.85 | 4849.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 15:15:00 | 4852.10 | 4847.10 | 4849.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 09:15:00 | 4904.45 | 4847.10 | 4849.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 09:15:00 | 4902.30 | 4858.14 | 4854.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 09:15:00 | 4932.40 | 4880.28 | 4868.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 09:15:00 | 4923.10 | 4954.10 | 4922.76 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-23 10:00:00 | 4923.10 | 4954.10 | 4922.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 10:15:00 | 4873.95 | 4938.07 | 4918.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 10:45:00 | 4887.00 | 4938.07 | 4918.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 11:15:00 | 4877.05 | 4925.87 | 4914.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 12:30:00 | 4885.00 | 4914.40 | 4910.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 14:00:00 | 4886.65 | 4908.85 | 4908.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-23 14:15:00 | 4879.40 | 4902.96 | 4905.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — SELL (started 2024-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-23 14:15:00 | 4879.40 | 4902.96 | 4905.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 09:15:00 | 4873.10 | 4893.64 | 4900.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 10:15:00 | 4893.60 | 4880.59 | 4887.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 10:15:00 | 4893.60 | 4880.59 | 4887.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 10:15:00 | 4893.60 | 4880.59 | 4887.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:00:00 | 4893.60 | 4880.59 | 4887.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 11:15:00 | 4882.00 | 4880.87 | 4887.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 11:30:00 | 4900.00 | 4880.87 | 4887.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 12:15:00 | 4886.75 | 4882.04 | 4887.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 12:30:00 | 4898.15 | 4882.04 | 4887.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 13:15:00 | 4877.80 | 4881.20 | 4886.47 | EMA400 retest candle locked (from downside) |

### Cycle 26 — BUY (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 09:15:00 | 4972.00 | 4902.94 | 4895.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 14:15:00 | 4995.25 | 4944.50 | 4920.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 10:15:00 | 5006.90 | 5038.13 | 4999.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 10:15:00 | 5006.90 | 5038.13 | 4999.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 10:15:00 | 5006.90 | 5038.13 | 4999.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 10:45:00 | 5000.40 | 5038.13 | 4999.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 13:15:00 | 5027.55 | 5030.68 | 5005.15 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2024-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 12:15:00 | 4962.10 | 4992.98 | 4995.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 09:15:00 | 4774.95 | 4941.61 | 4970.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 12:15:00 | 4684.80 | 4668.13 | 4717.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 13:00:00 | 4684.80 | 4668.13 | 4717.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 4730.00 | 4680.51 | 4718.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:00:00 | 4730.00 | 4680.51 | 4718.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 4691.05 | 4682.62 | 4716.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 11:45:00 | 4667.70 | 4689.15 | 4709.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-10 09:15:00 | 4748.65 | 4713.89 | 4714.84 | SL hit (close>static) qty=1.00 sl=4738.50 alert=retest2 |

### Cycle 28 — BUY (started 2024-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-10 10:15:00 | 4726.65 | 4716.45 | 4715.92 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 14:15:00 | 4697.80 | 4716.76 | 4716.87 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-10-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-11 09:15:00 | 4802.35 | 4730.62 | 4722.95 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2024-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 12:15:00 | 4698.55 | 4737.23 | 4739.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 09:15:00 | 4675.10 | 4720.32 | 4730.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 12:15:00 | 4647.35 | 4644.13 | 4676.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-17 13:00:00 | 4647.35 | 4644.13 | 4676.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 4746.90 | 4657.18 | 4671.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 09:45:00 | 4739.95 | 4657.18 | 4671.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 10:15:00 | 4771.25 | 4679.99 | 4680.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 10:30:00 | 4787.20 | 4679.99 | 4680.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2024-10-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-18 11:15:00 | 4767.90 | 4697.57 | 4688.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-18 12:15:00 | 4773.45 | 4712.75 | 4695.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-22 09:15:00 | 4770.70 | 4789.95 | 4761.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-22 10:00:00 | 4770.70 | 4789.95 | 4761.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 10:15:00 | 4799.90 | 4791.94 | 4764.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 10:30:00 | 4764.60 | 4791.94 | 4764.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 13:15:00 | 4786.75 | 4791.21 | 4771.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 14:00:00 | 4786.75 | 4791.21 | 4771.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 14:15:00 | 4752.95 | 4783.56 | 4769.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-22 15:00:00 | 4752.95 | 4783.56 | 4769.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 15:15:00 | 4759.45 | 4778.73 | 4768.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-23 09:15:00 | 4701.00 | 4778.73 | 4768.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2024-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-23 10:15:00 | 4703.25 | 4750.45 | 4756.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 13:15:00 | 4671.20 | 4718.55 | 4739.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-25 15:15:00 | 4601.00 | 4594.89 | 4635.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-28 09:15:00 | 4562.25 | 4594.89 | 4635.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 4589.65 | 4593.85 | 4631.29 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2024-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 13:15:00 | 4717.90 | 4647.26 | 4646.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 09:15:00 | 4735.30 | 4682.92 | 4664.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 14:15:00 | 4866.80 | 4875.18 | 4811.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 15:00:00 | 4866.80 | 4875.18 | 4811.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 4888.20 | 4876.34 | 4823.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:30:00 | 4848.80 | 4876.34 | 4823.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 4863.15 | 4903.74 | 4871.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 4863.15 | 4903.74 | 4871.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 4819.50 | 4886.89 | 4866.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 4819.50 | 4886.89 | 4866.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2024-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 14:15:00 | 4836.60 | 4853.33 | 4854.55 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2024-11-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 10:15:00 | 4894.85 | 4860.09 | 4857.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 4930.20 | 4888.27 | 4872.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-06 14:15:00 | 4912.20 | 4917.15 | 4895.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-06 15:00:00 | 4912.20 | 4917.15 | 4895.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 4846.55 | 4903.64 | 4893.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 4846.55 | 4903.64 | 4893.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 4851.55 | 4893.22 | 4889.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 4830.00 | 4893.22 | 4889.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2024-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 11:15:00 | 4836.50 | 4881.88 | 4884.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 10:15:00 | 4815.85 | 4862.85 | 4874.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 11:15:00 | 4814.15 | 4810.63 | 4833.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-11 12:00:00 | 4814.15 | 4810.63 | 4833.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 4816.15 | 4811.74 | 4831.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 12:45:00 | 4818.70 | 4811.74 | 4831.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 4946.80 | 4682.81 | 4707.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 4946.80 | 4682.81 | 4707.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2024-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 10:15:00 | 4936.45 | 4733.54 | 4727.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 09:15:00 | 4975.00 | 4895.91 | 4851.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 4956.70 | 4957.21 | 4903.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 15:00:00 | 4956.70 | 4957.21 | 4903.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 4940.10 | 4953.59 | 4911.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-22 14:00:00 | 4963.25 | 4928.07 | 4916.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-26 09:15:00 | 4880.95 | 4981.38 | 4966.47 | SL hit (close<static) qty=1.00 sl=4890.30 alert=retest2 |

### Cycle 39 — SELL (started 2024-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 11:15:00 | 4909.30 | 4957.22 | 4957.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 09:15:00 | 4858.90 | 4914.14 | 4928.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 14:15:00 | 4821.50 | 4813.56 | 4834.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-02 15:00:00 | 4821.50 | 4813.56 | 4834.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 4857.75 | 4824.22 | 4836.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 10:00:00 | 4819.95 | 4832.42 | 4835.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-05 14:45:00 | 4800.35 | 4819.20 | 4819.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-05 15:15:00 | 4826.05 | 4820.57 | 4819.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2024-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 15:15:00 | 4826.05 | 4820.57 | 4819.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-06 09:15:00 | 4864.50 | 4829.35 | 4824.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-09 09:15:00 | 4842.00 | 4860.61 | 4847.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 09:15:00 | 4842.00 | 4860.61 | 4847.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 4842.00 | 4860.61 | 4847.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:15:00 | 4844.95 | 4860.61 | 4847.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 4846.90 | 4857.86 | 4847.03 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2024-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 14:15:00 | 4835.70 | 4841.48 | 4841.58 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2024-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 15:15:00 | 4857.40 | 4844.66 | 4843.02 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2024-12-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 10:15:00 | 4833.40 | 4841.58 | 4841.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-10 11:15:00 | 4824.85 | 4838.23 | 4840.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 13:15:00 | 4799.75 | 4799.18 | 4809.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-12 14:00:00 | 4799.75 | 4799.18 | 4809.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 14:15:00 | 4812.25 | 4801.79 | 4810.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-12 15:00:00 | 4812.25 | 4801.79 | 4810.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 15:15:00 | 4802.70 | 4801.97 | 4809.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 09:15:00 | 4817.50 | 4801.97 | 4809.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 4791.95 | 4799.97 | 4807.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-13 10:15:00 | 4773.55 | 4799.97 | 4807.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-13 13:15:00 | 4848.95 | 4810.01 | 4809.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2024-12-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-13 13:15:00 | 4848.95 | 4810.01 | 4809.61 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-12-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 11:15:00 | 4795.85 | 4813.05 | 4813.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 4759.30 | 4799.05 | 4807.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 11:15:00 | 4786.75 | 4775.77 | 4789.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 11:15:00 | 4786.75 | 4775.77 | 4789.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 11:15:00 | 4786.75 | 4775.77 | 4789.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 11:30:00 | 4786.25 | 4775.77 | 4789.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 4763.70 | 4773.36 | 4787.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 15:00:00 | 4747.60 | 4766.18 | 4781.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:15:00 | 4706.45 | 4764.73 | 4779.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 12:45:00 | 4745.20 | 4743.11 | 4762.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 10:15:00 | 4813.80 | 4772.94 | 4770.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2024-12-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-20 10:15:00 | 4813.80 | 4772.94 | 4770.61 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2024-12-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 13:15:00 | 4750.10 | 4769.56 | 4769.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 4724.15 | 4760.47 | 4765.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 10:15:00 | 4757.95 | 4752.78 | 4760.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-23 11:00:00 | 4757.95 | 4752.78 | 4760.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 4751.30 | 4752.48 | 4759.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 11:30:00 | 4765.70 | 4752.48 | 4759.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 12:15:00 | 4750.30 | 4752.05 | 4758.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 12:45:00 | 4748.75 | 4752.05 | 4758.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 14:15:00 | 4751.95 | 4752.02 | 4757.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 14:30:00 | 4751.50 | 4752.02 | 4757.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 15:15:00 | 4756.95 | 4753.01 | 4757.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 09:15:00 | 4765.05 | 4753.01 | 4757.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 4770.85 | 4756.57 | 4758.69 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2024-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 10:15:00 | 4800.00 | 4765.26 | 4762.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 4935.45 | 4821.50 | 4796.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 14:15:00 | 4854.50 | 4873.47 | 4852.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 14:15:00 | 4854.50 | 4873.47 | 4852.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 14:15:00 | 4854.50 | 4873.47 | 4852.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 15:00:00 | 4854.50 | 4873.47 | 4852.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 15:15:00 | 4850.00 | 4868.77 | 4851.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:15:00 | 4837.95 | 4868.77 | 4851.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 4831.90 | 4861.40 | 4850.07 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2024-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 15:15:00 | 4825.00 | 4843.27 | 4845.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-01 09:15:00 | 4784.85 | 4831.58 | 4839.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-01 12:15:00 | 4837.90 | 4825.33 | 4834.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-01 12:15:00 | 4837.90 | 4825.33 | 4834.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 12:15:00 | 4837.90 | 4825.33 | 4834.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 13:00:00 | 4837.90 | 4825.33 | 4834.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 13:15:00 | 4882.05 | 4836.68 | 4838.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-01 14:00:00 | 4882.05 | 4836.68 | 4838.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2025-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-01 14:15:00 | 4886.30 | 4846.60 | 4842.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 10:15:00 | 4960.10 | 4880.07 | 4859.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 10:15:00 | 5242.00 | 5277.23 | 5183.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 11:00:00 | 5242.00 | 5277.23 | 5183.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 5250.35 | 5271.85 | 5189.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:45:00 | 5197.60 | 5271.85 | 5189.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 5217.65 | 5251.33 | 5214.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 11:00:00 | 5217.65 | 5251.33 | 5214.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 11:15:00 | 5196.15 | 5240.29 | 5212.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 12:00:00 | 5196.15 | 5240.29 | 5212.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 12:15:00 | 5197.20 | 5231.67 | 5211.20 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 5158.70 | 5196.36 | 5199.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 10:15:00 | 5145.10 | 5165.27 | 5178.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 15:15:00 | 5153.55 | 5148.76 | 5163.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-10 09:15:00 | 5112.35 | 5148.76 | 5163.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 5046.30 | 5128.27 | 5152.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-13 09:15:00 | 5008.00 | 5081.72 | 5115.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-15 12:15:00 | 5065.00 | 5037.43 | 5034.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2025-01-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 12:15:00 | 5065.00 | 5037.43 | 5034.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-15 13:15:00 | 5075.45 | 5045.03 | 5038.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-16 12:15:00 | 5044.05 | 5069.77 | 5057.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-16 12:15:00 | 5044.05 | 5069.77 | 5057.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 12:15:00 | 5044.05 | 5069.77 | 5057.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 12:30:00 | 5043.35 | 5069.77 | 5057.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 13:15:00 | 5057.15 | 5067.24 | 5057.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 13:45:00 | 5060.95 | 5067.24 | 5057.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 14:15:00 | 5047.10 | 5063.22 | 5056.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-16 15:00:00 | 5047.10 | 5063.22 | 5056.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 15:15:00 | 5060.95 | 5062.76 | 5056.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:15:00 | 5044.00 | 5062.76 | 5056.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 5060.80 | 5062.37 | 5057.27 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2025-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-17 11:15:00 | 5027.75 | 5050.19 | 5052.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-17 13:15:00 | 5008.25 | 5038.94 | 5046.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-20 11:15:00 | 5015.45 | 5009.46 | 5027.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 11:15:00 | 5015.45 | 5009.46 | 5027.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 11:15:00 | 5015.45 | 5009.46 | 5027.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 12:00:00 | 5015.45 | 5009.46 | 5027.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 14:15:00 | 5019.75 | 5014.02 | 5025.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-20 14:45:00 | 5024.95 | 5014.02 | 5025.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 09:15:00 | 5031.30 | 5018.13 | 5025.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-21 09:30:00 | 5065.00 | 5018.13 | 5025.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-21 10:15:00 | 5014.65 | 5017.43 | 5024.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 12:15:00 | 4996.70 | 5018.23 | 5023.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-21 13:45:00 | 4988.10 | 5012.77 | 5020.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 10:00:00 | 4989.65 | 4996.37 | 5009.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-22 15:15:00 | 4996.50 | 5000.35 | 5006.06 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 15:15:00 | 4996.50 | 4999.58 | 5005.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 09:45:00 | 5016.20 | 5004.70 | 5007.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-01-23 10:15:00 | 5056.50 | 5015.06 | 5011.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 5056.50 | 5015.06 | 5011.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-23 12:15:00 | 5109.80 | 5042.25 | 5025.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-27 10:15:00 | 5128.70 | 5162.32 | 5123.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-27 10:15:00 | 5128.70 | 5162.32 | 5123.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 10:15:00 | 5128.70 | 5162.32 | 5123.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 10:30:00 | 5125.10 | 5162.32 | 5123.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 11:15:00 | 5124.75 | 5154.81 | 5123.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 12:00:00 | 5124.75 | 5154.81 | 5123.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 12:15:00 | 5142.45 | 5152.34 | 5125.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 12:45:00 | 5126.80 | 5152.34 | 5125.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-27 13:15:00 | 5139.45 | 5149.76 | 5126.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-27 13:30:00 | 5139.30 | 5149.76 | 5126.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 09:15:00 | 5153.40 | 5148.30 | 5131.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-28 10:00:00 | 5153.40 | 5148.30 | 5131.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 5104.10 | 5149.10 | 5139.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-28 15:00:00 | 5104.10 | 5149.10 | 5139.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 15:15:00 | 5100.50 | 5139.38 | 5136.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-29 09:15:00 | 5129.90 | 5139.38 | 5136.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 14:15:00 | 5187.75 | 5180.69 | 5162.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-29 14:30:00 | 5173.55 | 5180.69 | 5162.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 5254.85 | 5195.61 | 5172.28 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2025-02-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 11:15:00 | 5161.85 | 5181.45 | 5182.56 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 12:15:00 | 5344.35 | 5214.03 | 5197.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-03 09:15:00 | 5497.40 | 5340.37 | 5268.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-04 13:15:00 | 5461.65 | 5472.58 | 5411.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-04 14:00:00 | 5461.65 | 5472.58 | 5411.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 5377.05 | 5447.35 | 5414.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:45:00 | 5368.00 | 5447.35 | 5414.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 5396.30 | 5437.14 | 5412.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 12:45:00 | 5429.70 | 5431.16 | 5414.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 15:00:00 | 5422.00 | 5427.01 | 5414.98 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-06 09:15:00 | 5354.20 | 5412.62 | 5410.54 | SL hit (close<static) qty=1.00 sl=5367.75 alert=retest2 |

### Cycle 57 — SELL (started 2025-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 10:15:00 | 5376.05 | 5405.31 | 5407.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 13:15:00 | 5345.05 | 5387.71 | 5398.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 14:15:00 | 5366.45 | 5355.46 | 5371.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 14:15:00 | 5366.45 | 5355.46 | 5371.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 14:15:00 | 5366.45 | 5355.46 | 5371.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 15:00:00 | 5366.45 | 5355.46 | 5371.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 15:15:00 | 5391.85 | 5362.74 | 5373.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:15:00 | 5392.00 | 5362.74 | 5373.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 5383.95 | 5366.98 | 5374.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 13:00:00 | 5353.70 | 5369.83 | 5374.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 13:45:00 | 5344.90 | 5360.53 | 5369.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 5086.01 | 5290.67 | 5334.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 5077.65 | 5290.67 | 5334.96 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-13 09:15:00 | 4818.33 | 4910.85 | 5027.04 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 58 — BUY (started 2025-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 14:15:00 | 4812.45 | 4767.00 | 4762.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 09:15:00 | 4843.80 | 4789.24 | 4773.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-25 12:15:00 | 5006.45 | 5013.43 | 4972.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-25 12:45:00 | 4994.30 | 5013.43 | 4972.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 4948.95 | 4992.78 | 4974.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 10:00:00 | 4948.95 | 4992.78 | 4974.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 10:15:00 | 4941.95 | 4982.62 | 4971.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-27 11:00:00 | 4941.95 | 4982.62 | 4971.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2025-02-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-27 12:15:00 | 4922.10 | 4961.77 | 4963.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-27 13:15:00 | 4906.40 | 4950.69 | 4958.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 09:15:00 | 4867.45 | 4833.65 | 4877.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-03 09:15:00 | 4867.45 | 4833.65 | 4877.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 4867.45 | 4833.65 | 4877.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 4815.15 | 4881.72 | 4885.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 10:00:00 | 4847.00 | 4874.78 | 4882.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 12:15:00 | 4837.30 | 4867.99 | 4877.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 15:00:00 | 4812.60 | 4841.18 | 4861.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 4945.45 | 4856.73 | 4864.92 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 4945.45 | 4856.73 | 4864.92 | SL hit (close>static) qty=1.00 sl=4910.95 alert=retest2 |

### Cycle 60 — BUY (started 2025-03-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 10:15:00 | 4966.45 | 4878.68 | 4874.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 4986.40 | 4900.22 | 4884.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 09:15:00 | 5087.85 | 5097.37 | 5052.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 09:30:00 | 5071.35 | 5097.37 | 5052.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 5050.50 | 5088.00 | 5052.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 5050.50 | 5088.00 | 5052.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 5025.10 | 5075.42 | 5049.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 5025.10 | 5075.42 | 5049.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 5044.95 | 5069.32 | 5049.42 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2025-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 15:15:00 | 4972.50 | 5031.85 | 5035.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 10:15:00 | 4968.55 | 5010.17 | 5024.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 5005.00 | 4997.32 | 5011.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-12 09:15:00 | 4971.30 | 4997.32 | 5011.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 4969.00 | 4991.66 | 5007.65 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2025-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 09:15:00 | 5025.35 | 5012.45 | 5011.54 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2025-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 12:15:00 | 4993.10 | 5007.62 | 5009.50 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2025-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 13:15:00 | 5023.25 | 5010.75 | 5010.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 14:15:00 | 5023.50 | 5013.30 | 5011.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 15:15:00 | 5010.00 | 5012.64 | 5011.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-13 15:15:00 | 5010.00 | 5012.64 | 5011.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 5010.00 | 5012.64 | 5011.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:15:00 | 5052.45 | 5012.64 | 5011.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 5046.30 | 5019.37 | 5014.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 09:15:00 | 5079.40 | 5044.99 | 5031.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 13:15:00 | 5366.85 | 5393.70 | 5393.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 13:15:00 | 5366.85 | 5393.70 | 5393.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 14:15:00 | 5348.80 | 5384.72 | 5389.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 10:15:00 | 5390.90 | 5373.17 | 5382.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 10:15:00 | 5390.90 | 5373.17 | 5382.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 5390.90 | 5373.17 | 5382.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 10:45:00 | 5397.25 | 5373.17 | 5382.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 11:15:00 | 5393.60 | 5377.26 | 5383.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 11:45:00 | 5394.60 | 5377.26 | 5383.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 5360.75 | 5373.96 | 5381.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 13:45:00 | 5334.40 | 5368.17 | 5377.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-28 15:15:00 | 5345.25 | 5364.96 | 5375.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 10:45:00 | 5341.85 | 5353.79 | 5367.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 11:00:00 | 5341.85 | 5330.76 | 5344.28 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 5346.15 | 5333.84 | 5344.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:45:00 | 5340.50 | 5333.84 | 5344.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 12:15:00 | 5327.65 | 5332.60 | 5342.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 13:30:00 | 5318.50 | 5332.00 | 5341.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 14:15:00 | 5356.80 | 5336.96 | 5343.08 | SL hit (close>static) qty=1.00 sl=5348.30 alert=retest2 |

### Cycle 66 — BUY (started 2025-04-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 10:15:00 | 5399.50 | 5349.76 | 5347.26 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 5280.45 | 5352.92 | 5353.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 5257.05 | 5310.09 | 5331.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 5181.30 | 5131.60 | 5198.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 10:00:00 | 5181.30 | 5131.60 | 5198.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 5183.00 | 5141.88 | 5196.88 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2025-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 11:15:00 | 5313.40 | 5230.45 | 5222.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 5388.05 | 5276.88 | 5249.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-22 14:15:00 | 5783.00 | 5802.81 | 5744.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-22 15:00:00 | 5783.00 | 5802.81 | 5744.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 09:15:00 | 5711.50 | 5782.34 | 5745.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 5711.50 | 5782.34 | 5745.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 5709.00 | 5767.67 | 5742.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 5676.00 | 5767.67 | 5742.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 15:15:00 | 5730.00 | 5733.71 | 5731.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 09:15:00 | 5728.50 | 5733.71 | 5731.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 5722.00 | 5731.37 | 5730.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 10:00:00 | 5722.00 | 5731.37 | 5730.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2025-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 10:15:00 | 5705.50 | 5726.20 | 5728.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 11:15:00 | 5686.50 | 5718.26 | 5724.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 5625.00 | 5598.08 | 5635.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 5625.00 | 5598.08 | 5635.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 5625.00 | 5598.08 | 5635.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 09:30:00 | 5643.50 | 5598.08 | 5635.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 5628.50 | 5603.11 | 5631.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:00:00 | 5628.50 | 5603.11 | 5631.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 5633.00 | 5609.09 | 5631.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:30:00 | 5638.00 | 5609.09 | 5631.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 13:15:00 | 5622.00 | 5611.67 | 5630.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 14:30:00 | 5604.00 | 5612.24 | 5629.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 09:30:00 | 5617.00 | 5608.81 | 5624.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 10:00:00 | 5590.50 | 5608.81 | 5624.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 10:15:00 | 5580.50 | 5502.59 | 5498.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — BUY (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 10:15:00 | 5580.50 | 5502.59 | 5498.84 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2025-05-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 13:15:00 | 5480.00 | 5503.32 | 5504.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 10:15:00 | 5438.00 | 5476.82 | 5490.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 5477.50 | 5362.04 | 5388.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 5477.50 | 5362.04 | 5388.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 5477.50 | 5362.04 | 5388.77 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 5500.00 | 5409.23 | 5406.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 5513.50 | 5444.13 | 5423.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 5439.50 | 5465.43 | 5443.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 10:15:00 | 5439.50 | 5465.43 | 5443.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 10:15:00 | 5439.50 | 5465.43 | 5443.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 11:00:00 | 5439.50 | 5465.43 | 5443.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 11:15:00 | 5468.00 | 5465.94 | 5445.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 11:45:00 | 5450.50 | 5465.94 | 5445.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 5414.50 | 5455.66 | 5442.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:00:00 | 5414.50 | 5455.66 | 5442.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 5407.00 | 5445.92 | 5439.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 14:00:00 | 5407.00 | 5445.92 | 5439.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 5440.00 | 5437.99 | 5436.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 10:45:00 | 5414.50 | 5437.99 | 5436.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 5451.50 | 5440.70 | 5438.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:45:00 | 5434.50 | 5440.70 | 5438.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 5453.50 | 5443.26 | 5439.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 5430.00 | 5443.26 | 5439.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 13:15:00 | 5445.00 | 5443.61 | 5440.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 14:00:00 | 5445.00 | 5443.61 | 5440.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 5445.50 | 5443.98 | 5440.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 15:00:00 | 5445.50 | 5443.98 | 5440.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 15:15:00 | 5447.00 | 5444.59 | 5441.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 09:15:00 | 5473.00 | 5444.59 | 5441.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 5428.00 | 5441.27 | 5439.94 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2025-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 10:15:00 | 5404.00 | 5433.82 | 5436.67 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2025-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 12:15:00 | 5479.50 | 5440.58 | 5439.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 13:15:00 | 5514.00 | 5455.27 | 5445.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 14:15:00 | 5531.50 | 5555.19 | 5525.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 15:00:00 | 5531.50 | 5555.19 | 5525.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 5528.00 | 5549.75 | 5525.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:15:00 | 5474.00 | 5549.75 | 5525.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 5470.50 | 5533.90 | 5520.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 5447.00 | 5533.90 | 5520.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 5495.50 | 5526.22 | 5518.46 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2025-05-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 11:15:00 | 5460.50 | 5513.08 | 5513.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 12:15:00 | 5445.50 | 5499.56 | 5507.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 10:15:00 | 5455.00 | 5450.01 | 5474.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 10:15:00 | 5455.00 | 5450.01 | 5474.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 5455.00 | 5450.01 | 5474.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:30:00 | 5456.50 | 5450.01 | 5474.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 5417.50 | 5394.62 | 5413.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 5417.50 | 5394.62 | 5413.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 5412.00 | 5398.09 | 5413.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:45:00 | 5417.00 | 5398.09 | 5413.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 11:15:00 | 5403.00 | 5399.08 | 5412.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 11:30:00 | 5411.00 | 5399.08 | 5412.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 12:15:00 | 5407.50 | 5400.76 | 5412.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 12:30:00 | 5419.50 | 5400.76 | 5412.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 5457.50 | 5411.45 | 5413.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-26 09:45:00 | 5462.50 | 5411.45 | 5413.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 5435.50 | 5416.26 | 5415.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 11:15:00 | 5469.00 | 5426.81 | 5420.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 13:15:00 | 5423.00 | 5427.36 | 5421.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 13:15:00 | 5423.00 | 5427.36 | 5421.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 13:15:00 | 5423.00 | 5427.36 | 5421.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:00:00 | 5423.00 | 5427.36 | 5421.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 14:15:00 | 5429.00 | 5427.69 | 5422.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 14:30:00 | 5421.50 | 5427.69 | 5422.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 15:15:00 | 5430.00 | 5428.15 | 5423.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:15:00 | 5407.50 | 5428.15 | 5423.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 5389.50 | 5420.42 | 5420.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:45:00 | 5382.00 | 5420.42 | 5420.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2025-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 10:15:00 | 5413.00 | 5418.94 | 5419.44 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 11:15:00 | 5440.00 | 5423.15 | 5421.31 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2025-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 13:15:00 | 5400.00 | 5418.90 | 5419.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 09:15:00 | 5357.00 | 5401.33 | 5411.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 13:15:00 | 5352.50 | 5340.20 | 5363.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-29 13:30:00 | 5351.50 | 5340.20 | 5363.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 5387.50 | 5349.66 | 5365.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 5387.50 | 5349.66 | 5365.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 5385.00 | 5356.73 | 5367.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 5396.00 | 5356.73 | 5367.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 10:15:00 | 5376.00 | 5364.22 | 5368.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:00:00 | 5376.00 | 5364.22 | 5368.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 5345.50 | 5360.48 | 5366.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 12:15:00 | 5344.50 | 5360.48 | 5366.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 10:00:00 | 5337.00 | 5345.61 | 5355.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 10:45:00 | 5344.50 | 5348.19 | 5356.07 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 11:15:00 | 5341.00 | 5348.19 | 5356.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 11:15:00 | 5351.00 | 5348.75 | 5355.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 11:45:00 | 5352.50 | 5348.75 | 5355.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 12:15:00 | 5348.00 | 5348.60 | 5354.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:45:00 | 5332.00 | 5352.08 | 5354.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 11:00:00 | 5317.50 | 5345.17 | 5351.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 09:15:00 | 5380.00 | 5349.59 | 5349.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 80 — BUY (started 2025-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 09:15:00 | 5380.00 | 5349.59 | 5349.45 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 10:15:00 | 5334.50 | 5346.57 | 5348.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 11:15:00 | 5318.50 | 5340.95 | 5345.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 11:15:00 | 5327.00 | 5320.95 | 5331.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 11:15:00 | 5327.00 | 5320.95 | 5331.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 11:15:00 | 5327.00 | 5320.95 | 5331.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 12:00:00 | 5327.00 | 5320.95 | 5331.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 5306.00 | 5317.08 | 5326.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:45:00 | 5316.00 | 5317.08 | 5326.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 5350.50 | 5321.52 | 5326.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:45:00 | 5360.00 | 5321.52 | 5326.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2025-06-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 11:15:00 | 5377.00 | 5332.61 | 5330.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 14:15:00 | 5396.50 | 5356.80 | 5343.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 15:15:00 | 5385.00 | 5385.03 | 5368.87 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-10 09:30:00 | 5399.50 | 5390.43 | 5372.79 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 5379.00 | 5393.45 | 5381.84 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-10 14:15:00 | 5379.00 | 5393.45 | 5381.84 | SL hit (close<ema400) qty=1.00 sl=5381.84 alert=retest1 |

### Cycle 83 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 5355.50 | 5375.86 | 5378.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 5313.00 | 5353.04 | 5364.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 5310.50 | 5301.68 | 5326.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 14:00:00 | 5310.50 | 5301.68 | 5326.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 5324.50 | 5306.24 | 5326.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 5324.50 | 5306.24 | 5326.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 5323.00 | 5309.59 | 5326.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 5340.50 | 5309.59 | 5326.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 5342.00 | 5316.07 | 5327.76 | EMA400 retest candle locked (from downside) |

### Cycle 84 — BUY (started 2025-06-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 12:15:00 | 5372.00 | 5340.89 | 5337.30 | EMA200 above EMA400 |

### Cycle 85 — SELL (started 2025-06-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 11:15:00 | 5322.50 | 5338.58 | 5339.40 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2025-06-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 15:15:00 | 5344.00 | 5339.83 | 5339.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 09:15:00 | 5449.50 | 5361.77 | 5349.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 15:15:00 | 5385.00 | 5390.18 | 5372.64 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-19 09:15:00 | 5437.00 | 5390.18 | 5372.64 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 14:15:00 | 5578.00 | 5592.05 | 5572.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 14:45:00 | 5571.00 | 5592.05 | 5572.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 15:15:00 | 5576.50 | 5588.94 | 5573.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:15:00 | 5571.00 | 5588.94 | 5573.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 5569.00 | 5584.95 | 5572.77 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-26 09:15:00 | 5569.00 | 5584.95 | 5572.77 | SL hit (close<ema400) qty=1.00 sl=5572.77 alert=retest1 |

### Cycle 87 — SELL (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 11:15:00 | 5638.50 | 5691.77 | 5697.57 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 5716.00 | 5680.59 | 5678.70 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2025-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 10:15:00 | 5662.50 | 5683.18 | 5683.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 15:15:00 | 5649.00 | 5665.17 | 5673.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 5591.00 | 5562.59 | 5592.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 5591.00 | 5562.59 | 5592.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 5591.00 | 5562.59 | 5592.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:15:00 | 5614.50 | 5562.59 | 5592.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 5618.50 | 5573.77 | 5595.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:45:00 | 5613.50 | 5573.77 | 5595.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 5632.50 | 5585.52 | 5598.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:30:00 | 5630.50 | 5585.52 | 5598.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 5585.00 | 5587.73 | 5597.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 5575.00 | 5589.81 | 5596.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 11:15:00 | 5573.00 | 5584.92 | 5593.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 13:15:00 | 5621.50 | 5592.45 | 5594.47 | SL hit (close>static) qty=1.00 sl=5601.00 alert=retest2 |

### Cycle 90 — BUY (started 2025-07-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 14:15:00 | 5627.50 | 5599.46 | 5597.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-17 09:15:00 | 5649.50 | 5611.63 | 5603.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 10:15:00 | 5637.00 | 5638.93 | 5625.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 10:15:00 | 5637.00 | 5638.93 | 5625.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 5637.00 | 5638.93 | 5625.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 5642.00 | 5638.93 | 5625.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 5635.00 | 5638.14 | 5626.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:30:00 | 5613.50 | 5638.14 | 5626.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 5632.00 | 5637.50 | 5629.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 14:45:00 | 5621.00 | 5637.50 | 5629.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 5620.50 | 5634.10 | 5628.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 5567.00 | 5634.10 | 5628.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 5613.50 | 5629.98 | 5626.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:30:00 | 5580.50 | 5629.98 | 5626.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 5607.50 | 5625.48 | 5625.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:45:00 | 5597.50 | 5625.48 | 5625.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — SELL (started 2025-07-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 11:15:00 | 5578.00 | 5615.99 | 5620.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 12:15:00 | 5569.00 | 5606.59 | 5616.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 5485.00 | 5457.70 | 5487.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 5485.00 | 5457.70 | 5487.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 5485.00 | 5457.70 | 5487.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:45:00 | 5490.00 | 5457.70 | 5487.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 5465.00 | 5459.16 | 5485.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:30:00 | 5487.00 | 5459.16 | 5485.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 5381.50 | 5448.09 | 5469.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 14:15:00 | 5372.50 | 5407.63 | 5428.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 15:15:00 | 5380.00 | 5404.61 | 5425.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 14:15:00 | 5474.00 | 5427.20 | 5425.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2025-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 14:15:00 | 5474.00 | 5427.20 | 5425.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 15:15:00 | 5479.50 | 5437.66 | 5430.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 13:15:00 | 5483.50 | 5484.13 | 5460.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 13:15:00 | 5483.50 | 5484.13 | 5460.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 5483.50 | 5484.13 | 5460.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:30:00 | 5498.50 | 5484.13 | 5460.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 5490.00 | 5485.52 | 5465.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 5415.00 | 5485.52 | 5465.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 5438.00 | 5476.02 | 5462.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:00:00 | 5458.00 | 5472.41 | 5462.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-18 09:15:00 | 6003.80 | 5793.53 | 5746.23 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2025-09-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 15:15:00 | 6795.00 | 6814.57 | 6817.02 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 6868.00 | 6825.26 | 6821.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 12:15:00 | 6877.50 | 6846.11 | 6832.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 6897.00 | 6901.67 | 6877.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 14:00:00 | 6897.00 | 6901.67 | 6877.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 6900.00 | 6899.47 | 6880.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:15:00 | 6906.50 | 6899.47 | 6880.30 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 14:15:00 | 6905.00 | 6899.59 | 6888.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 6976.50 | 7013.63 | 7016.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2025-09-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 14:15:00 | 6976.50 | 7013.63 | 7016.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 10:15:00 | 6967.50 | 6993.25 | 7005.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 11:15:00 | 7007.00 | 6996.00 | 7005.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 11:15:00 | 7007.00 | 6996.00 | 7005.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 11:15:00 | 7007.00 | 6996.00 | 7005.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 12:00:00 | 7007.00 | 6996.00 | 7005.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 7021.50 | 7001.10 | 7006.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:00:00 | 7021.50 | 7001.10 | 7006.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 6999.00 | 7000.68 | 7006.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:45:00 | 6992.50 | 7001.54 | 7005.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 11:15:00 | 7032.50 | 7011.73 | 7009.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 7032.50 | 7011.73 | 7009.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 12:15:00 | 7042.50 | 7017.88 | 7012.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 15:15:00 | 6988.00 | 7016.85 | 7013.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 15:15:00 | 6988.00 | 7016.85 | 7013.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 6988.00 | 7016.85 | 7013.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 6875.50 | 7016.85 | 7013.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — SELL (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 09:15:00 | 6864.00 | 6986.28 | 7000.26 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 13:15:00 | 6983.00 | 6951.88 | 6949.36 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 6900.00 | 6952.08 | 6953.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 15:15:00 | 6890.50 | 6939.77 | 6947.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 6957.50 | 6929.91 | 6941.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 10:15:00 | 6957.50 | 6929.91 | 6941.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 6957.50 | 6929.91 | 6941.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 6957.50 | 6929.91 | 6941.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 6914.50 | 6926.83 | 6938.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 13:00:00 | 6901.00 | 6921.66 | 6935.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 13:30:00 | 6899.50 | 6915.73 | 6931.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 12:15:00 | 6970.00 | 6934.33 | 6932.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 6970.00 | 6934.33 | 6932.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 09:15:00 | 7005.50 | 6962.52 | 6947.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-13 10:15:00 | 6951.00 | 6960.21 | 6948.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-13 10:15:00 | 6951.00 | 6960.21 | 6948.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 6951.00 | 6960.21 | 6948.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 6951.00 | 6960.21 | 6948.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 11:15:00 | 6950.00 | 6958.17 | 6948.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:15:00 | 6949.50 | 6958.17 | 6948.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 12:15:00 | 6946.50 | 6955.84 | 6948.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 12:45:00 | 6939.00 | 6955.84 | 6948.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 6915.00 | 6947.67 | 6945.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 14:00:00 | 6915.00 | 6947.67 | 6945.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 101 — SELL (started 2025-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 14:15:00 | 6912.00 | 6940.54 | 6942.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 09:15:00 | 6828.00 | 6914.74 | 6929.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 6883.50 | 6882.17 | 6904.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 14:15:00 | 6883.50 | 6882.17 | 6904.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 6883.50 | 6882.17 | 6904.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:30:00 | 6893.50 | 6882.17 | 6904.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 6929.50 | 6890.17 | 6904.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:30:00 | 6924.00 | 6890.17 | 6904.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 6904.00 | 6892.93 | 6904.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 6926.50 | 6892.93 | 6904.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 6903.00 | 6894.95 | 6904.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:45:00 | 6927.00 | 6894.95 | 6904.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 6900.50 | 6896.06 | 6904.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 13:00:00 | 6900.50 | 6896.06 | 6904.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 6886.00 | 6894.05 | 6902.36 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 7008.50 | 6917.02 | 6910.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 7069.50 | 6987.47 | 6954.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 12:15:00 | 7035.50 | 7045.18 | 7015.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 13:00:00 | 7035.50 | 7045.18 | 7015.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 13:15:00 | 7032.00 | 7042.55 | 7016.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 13:45:00 | 7015.50 | 7042.55 | 7016.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 7011.50 | 7036.34 | 7016.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 14:45:00 | 7022.50 | 7036.34 | 7016.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 7012.00 | 7031.47 | 7016.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 7025.00 | 7029.46 | 7017.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 6971.00 | 7017.77 | 7013.53 | EMA400 retest candle locked (from upside) |

### Cycle 103 — SELL (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 10:15:00 | 6964.00 | 7007.01 | 7009.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 11:15:00 | 6957.00 | 6997.01 | 7004.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 6865.00 | 6860.27 | 6903.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 10:15:00 | 6879.50 | 6860.27 | 6903.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 6916.00 | 6876.95 | 6897.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 13:45:00 | 6915.00 | 6876.95 | 6897.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 14:15:00 | 6908.50 | 6883.26 | 6898.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-27 15:15:00 | 6905.00 | 6883.26 | 6898.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 09:15:00 | 6963.00 | 6902.69 | 6905.23 | SL hit (close>static) qty=1.00 sl=6926.50 alert=retest2 |

### Cycle 104 — BUY (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 10:15:00 | 6975.00 | 6917.15 | 6911.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 14:15:00 | 7010.00 | 6953.83 | 6932.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 13:15:00 | 6953.00 | 6969.36 | 6951.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 13:15:00 | 6953.00 | 6969.36 | 6951.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 6953.00 | 6969.36 | 6951.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:00:00 | 6953.00 | 6969.36 | 6951.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 6948.50 | 6965.19 | 6951.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 15:00:00 | 6948.50 | 6965.19 | 6951.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 6978.50 | 6967.85 | 6953.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 6975.00 | 6967.85 | 6953.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 6923.00 | 6958.88 | 6951.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:00:00 | 6923.00 | 6958.88 | 6951.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 6929.50 | 6953.00 | 6949.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:45:00 | 6906.00 | 6953.00 | 6949.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2025-10-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 11:15:00 | 6895.50 | 6941.50 | 6944.26 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 7049.50 | 6940.99 | 6939.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-31 11:15:00 | 7080.00 | 6985.83 | 6961.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-03 11:15:00 | 7009.00 | 7016.94 | 6992.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-03 11:45:00 | 7003.00 | 7016.94 | 6992.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 6971.00 | 7008.92 | 6998.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:30:00 | 6940.50 | 7008.92 | 6998.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 6932.00 | 6993.54 | 6992.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 11:00:00 | 6932.00 | 6993.54 | 6992.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 107 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 6959.00 | 6986.63 | 6989.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 14:15:00 | 6926.50 | 6966.79 | 6978.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 12:15:00 | 6822.00 | 6816.17 | 6867.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 12:30:00 | 6832.50 | 6816.17 | 6867.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 13:15:00 | 6858.00 | 6824.54 | 6866.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 13:30:00 | 6855.00 | 6824.54 | 6866.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 6882.50 | 6836.13 | 6868.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 6882.50 | 6836.13 | 6868.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 6885.00 | 6845.90 | 6869.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-10 09:15:00 | 6820.50 | 6845.90 | 6869.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 6798.00 | 6853.88 | 6861.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 14:30:00 | 6877.00 | 6840.13 | 6846.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 10:00:00 | 6863.00 | 6849.96 | 6850.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-12 10:15:00 | 6880.00 | 6855.97 | 6852.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2025-11-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 10:15:00 | 6880.00 | 6855.97 | 6852.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-13 09:15:00 | 6930.00 | 6877.46 | 6864.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 6886.00 | 6895.45 | 6880.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 14:15:00 | 6886.00 | 6895.45 | 6880.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 14:15:00 | 6886.00 | 6895.45 | 6880.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 15:00:00 | 6886.00 | 6895.45 | 6880.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 6796.50 | 6875.66 | 6872.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 09:15:00 | 6792.00 | 6875.66 | 6872.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — SELL (started 2025-11-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 09:15:00 | 6746.00 | 6849.73 | 6861.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 11:15:00 | 6690.00 | 6799.98 | 6835.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 6739.00 | 6738.86 | 6786.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 09:30:00 | 6733.50 | 6738.86 | 6786.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 6781.00 | 6752.83 | 6785.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:30:00 | 6793.50 | 6752.83 | 6785.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 12:15:00 | 6788.00 | 6759.87 | 6785.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:30:00 | 6784.00 | 6759.87 | 6785.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 13:15:00 | 6792.50 | 6766.39 | 6785.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:00:00 | 6792.50 | 6766.39 | 6785.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 6800.00 | 6773.11 | 6787.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 14:30:00 | 6798.00 | 6773.11 | 6787.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 6809.00 | 6780.29 | 6789.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:15:00 | 6799.00 | 6780.29 | 6789.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 6755.50 | 6775.33 | 6786.14 | EMA400 retest candle locked (from downside) |

### Cycle 110 — BUY (started 2025-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 12:15:00 | 6829.50 | 6795.51 | 6793.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 13:15:00 | 6831.50 | 6802.71 | 6796.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 15:15:00 | 6785.00 | 6803.05 | 6798.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 15:15:00 | 6785.00 | 6803.05 | 6798.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 6785.00 | 6803.05 | 6798.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-19 09:15:00 | 6799.00 | 6803.05 | 6798.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 6827.00 | 6807.84 | 6800.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-19 14:00:00 | 6865.00 | 6822.83 | 6810.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-27 10:15:00 | 7051.50 | 7166.11 | 7180.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 7051.50 | 7166.11 | 7180.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 11:15:00 | 7005.50 | 7133.99 | 7164.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 14:15:00 | 7058.50 | 7047.26 | 7082.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-28 15:00:00 | 7058.50 | 7047.26 | 7082.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 7128.00 | 7063.05 | 7083.47 | EMA400 retest candle locked (from downside) |

### Cycle 112 — BUY (started 2025-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 13:15:00 | 7230.00 | 7112.66 | 7101.23 | EMA200 above EMA400 |

### Cycle 113 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 7069.50 | 7106.24 | 7107.27 | EMA200 below EMA400 |

### Cycle 114 — BUY (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 09:15:00 | 7148.50 | 7104.88 | 7099.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 14:15:00 | 7211.50 | 7151.05 | 7126.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 12:15:00 | 7148.50 | 7168.30 | 7146.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 12:15:00 | 7148.50 | 7168.30 | 7146.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 7148.50 | 7168.30 | 7146.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 7148.50 | 7168.30 | 7146.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 7122.50 | 7159.14 | 7143.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:30:00 | 7123.00 | 7159.14 | 7143.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 7140.50 | 7155.41 | 7143.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 11:45:00 | 7158.00 | 7143.43 | 7140.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-10 09:15:00 | 7163.50 | 7143.72 | 7142.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 7125.50 | 7215.70 | 7221.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 7125.50 | 7215.70 | 7221.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-15 11:15:00 | 7104.00 | 7179.25 | 7202.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 09:15:00 | 7151.50 | 7092.62 | 7123.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 7151.50 | 7092.62 | 7123.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 7151.50 | 7092.62 | 7123.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:30:00 | 7053.50 | 7113.28 | 7125.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 14:00:00 | 7074.50 | 7093.91 | 7110.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 09:15:00 | 7184.00 | 7113.81 | 7115.87 | SL hit (close>static) qty=1.00 sl=7176.50 alert=retest2 |

### Cycle 116 — BUY (started 2025-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 10:15:00 | 7182.50 | 7127.55 | 7121.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 7219.50 | 7175.59 | 7149.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 7290.00 | 7295.66 | 7258.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 09:15:00 | 7261.00 | 7295.66 | 7258.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 7270.00 | 7290.53 | 7259.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 10:45:00 | 7303.00 | 7294.43 | 7264.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 09:15:00 | 7353.50 | 7308.16 | 7284.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 11:45:00 | 7313.00 | 7313.20 | 7292.54 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 11:45:00 | 7306.50 | 7308.22 | 7301.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 7285.00 | 7303.57 | 7300.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 12:45:00 | 7281.50 | 7303.57 | 7300.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-29 13:15:00 | 7261.50 | 7295.16 | 7296.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2025-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 13:15:00 | 7261.50 | 7295.16 | 7296.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 10:15:00 | 7230.50 | 7274.09 | 7285.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 7244.50 | 7223.75 | 7249.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 10:15:00 | 7244.50 | 7223.75 | 7249.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 7244.50 | 7223.75 | 7249.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 7244.50 | 7223.75 | 7249.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 11:15:00 | 7260.50 | 7231.10 | 7250.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:00:00 | 7260.50 | 7231.10 | 7250.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 12:15:00 | 7271.00 | 7239.08 | 7252.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 12:45:00 | 7285.50 | 7239.08 | 7252.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 118 — BUY (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 14:15:00 | 7317.50 | 7263.15 | 7261.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 12:15:00 | 7348.50 | 7301.60 | 7282.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 14:15:00 | 7332.00 | 7341.67 | 7321.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 14:15:00 | 7332.00 | 7341.67 | 7321.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 7332.00 | 7341.67 | 7321.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:45:00 | 7317.50 | 7341.67 | 7321.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 7464.00 | 7366.51 | 7336.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 10:15:00 | 7476.50 | 7366.51 | 7336.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 11:30:00 | 7469.00 | 7403.05 | 7358.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 13:45:00 | 7469.00 | 7425.75 | 7377.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 14:15:00 | 7481.50 | 7425.75 | 7377.49 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 7514.50 | 7555.98 | 7541.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:45:00 | 7524.00 | 7555.98 | 7541.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 7508.00 | 7546.38 | 7538.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 14:00:00 | 7508.00 | 7546.38 | 7538.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-09 15:15:00 | 7510.00 | 7533.29 | 7533.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — SELL (started 2026-01-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 15:15:00 | 7510.00 | 7533.29 | 7533.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 7403.00 | 7507.23 | 7521.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 7439.50 | 7429.19 | 7469.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 15:00:00 | 7439.50 | 7429.19 | 7469.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 7454.00 | 7436.36 | 7466.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 10:30:00 | 7429.00 | 7431.49 | 7461.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 7435.50 | 7432.29 | 7458.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:30:00 | 7402.00 | 7423.93 | 7452.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 7057.55 | 7209.44 | 7270.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 7063.72 | 7209.44 | 7270.18 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 10:15:00 | 7031.90 | 7184.15 | 7253.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 7266.00 | 7163.76 | 7205.95 | SL hit (close>ema200) qty=0.50 sl=7163.76 alert=retest2 |

### Cycle 120 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 7147.50 | 7084.54 | 7084.39 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 09:15:00 | 6880.00 | 7043.63 | 7065.81 | EMA200 below EMA400 |

### Cycle 122 — BUY (started 2026-01-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 15:15:00 | 7071.50 | 7044.23 | 7042.91 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 09:15:00 | 7028.00 | 7040.99 | 7041.55 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 11:15:00 | 7060.00 | 7043.83 | 7042.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 13:15:00 | 7090.00 | 7053.17 | 7047.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 6965.00 | 7083.65 | 7069.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 6965.00 | 7083.65 | 7069.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 6965.00 | 7083.65 | 7069.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 6965.00 | 7083.65 | 7069.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 125 — SELL (started 2026-02-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 12:15:00 | 6930.00 | 7052.92 | 7057.03 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 7180.00 | 7017.83 | 7002.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 7208.50 | 7055.97 | 7021.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 7211.50 | 7245.64 | 7182.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 7211.50 | 7245.64 | 7182.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 7211.50 | 7245.64 | 7182.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 7211.50 | 7245.64 | 7182.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 12:15:00 | 7211.50 | 7229.00 | 7189.90 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2026-02-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 12:15:00 | 7151.00 | 7179.60 | 7180.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-09 09:15:00 | 7135.50 | 7165.21 | 7173.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 7194.50 | 7171.07 | 7175.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 10:15:00 | 7194.50 | 7171.07 | 7175.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 7194.50 | 7171.07 | 7175.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 10:45:00 | 7200.00 | 7171.07 | 7175.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 7200.00 | 7176.85 | 7177.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:45:00 | 7202.00 | 7176.85 | 7177.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — BUY (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 12:15:00 | 7203.00 | 7182.08 | 7179.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 15:15:00 | 7209.00 | 7188.10 | 7182.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-16 11:15:00 | 8009.50 | 8021.67 | 7907.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-16 11:45:00 | 7999.50 | 8021.67 | 7907.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 11:15:00 | 7939.00 | 7985.67 | 7943.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-17 12:00:00 | 7939.00 | 7985.67 | 7943.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 12:15:00 | 8016.00 | 7991.74 | 7949.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 13:30:00 | 8023.50 | 7996.49 | 7955.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-17 14:30:00 | 8019.50 | 8005.79 | 7963.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 15:15:00 | 8020.00 | 7992.09 | 7975.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-19 14:15:00 | 7900.50 | 7965.13 | 7971.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 129 — SELL (started 2026-02-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 14:15:00 | 7900.50 | 7965.13 | 7971.79 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2026-02-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 10:15:00 | 8040.00 | 7974.95 | 7966.95 | EMA200 above EMA400 |

### Cycle 131 — SELL (started 2026-02-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 13:15:00 | 7937.50 | 7971.57 | 7976.15 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2026-02-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 11:15:00 | 8074.00 | 7989.72 | 7980.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-26 12:15:00 | 8102.00 | 8047.92 | 8018.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 8087.00 | 8108.96 | 8061.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 09:15:00 | 8087.00 | 8108.96 | 8061.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 8087.00 | 8108.96 | 8061.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 09:45:00 | 8094.50 | 8108.96 | 8061.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 8083.50 | 8103.87 | 8063.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 11:45:00 | 8101.00 | 8098.10 | 8064.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 13:45:00 | 8101.50 | 8094.02 | 8068.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 14:15:00 | 8005.00 | 8076.22 | 8062.45 | SL hit (close<static) qty=1.00 sl=8056.00 alert=retest2 |

### Cycle 133 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 7930.00 | 8037.02 | 8046.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 7780.00 | 7956.25 | 8006.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 7696.00 | 7683.04 | 7781.61 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 10:15:00 | 7666.50 | 7683.04 | 7781.61 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 11:00:00 | 7657.00 | 7677.83 | 7770.28 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-05 13:45:00 | 7669.50 | 7670.99 | 7743.58 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 7725.00 | 7681.80 | 7741.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 7784.00 | 7681.80 | 7741.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 7742.00 | 7693.84 | 7741.90 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-05 15:15:00 | 7742.00 | 7693.84 | 7741.90 | SL hit (close>ema400) qty=1.00 sl=7741.90 alert=retest1 |

### Cycle 134 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 7019.00 | 6916.24 | 6911.32 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 6801.00 | 6923.79 | 6933.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 6745.00 | 6859.50 | 6879.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 6830.00 | 6757.84 | 6803.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 6830.00 | 6757.84 | 6803.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 6830.00 | 6757.84 | 6803.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 09:30:00 | 6859.50 | 6757.84 | 6803.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 10:15:00 | 6822.00 | 6770.67 | 6805.62 | EMA400 retest candle locked (from downside) |

### Cycle 136 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 6930.00 | 6841.65 | 6832.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 7050.00 | 6890.85 | 6857.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 6828.00 | 6952.10 | 6919.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 6828.00 | 6952.10 | 6919.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 6828.00 | 6952.10 | 6919.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 6828.00 | 6952.10 | 6919.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — SELL (started 2026-03-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 11:15:00 | 6768.00 | 6888.86 | 6894.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 6747.50 | 6827.65 | 6860.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 6805.50 | 6701.23 | 6763.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 6805.50 | 6701.23 | 6763.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 6805.50 | 6701.23 | 6763.16 | EMA400 retest candle locked (from downside) |

### Cycle 138 — BUY (started 2026-04-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 14:15:00 | 6825.00 | 6794.66 | 6791.52 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 6631.00 | 6765.98 | 6779.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-06 10:15:00 | 6599.00 | 6658.43 | 6703.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-07 14:15:00 | 6598.50 | 6558.48 | 6604.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 14:15:00 | 6598.50 | 6558.48 | 6604.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 6598.50 | 6558.48 | 6604.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 15:00:00 | 6598.50 | 6558.48 | 6604.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 6648.00 | 6576.39 | 6608.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-08 09:15:00 | 6931.50 | 6576.39 | 6608.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 7000.00 | 6661.11 | 6644.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 7049.00 | 6738.69 | 6681.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 7142.50 | 7295.36 | 7174.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 7142.50 | 7295.36 | 7174.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 7142.50 | 7295.36 | 7174.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 10:15:00 | 7136.00 | 7295.36 | 7174.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 7115.00 | 7259.28 | 7169.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 11:00:00 | 7115.00 | 7259.28 | 7169.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 141 — SELL (started 2026-04-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 15:15:00 | 7050.00 | 7132.80 | 7133.69 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2026-04-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 10:15:00 | 7165.00 | 7123.85 | 7118.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 14:15:00 | 7192.00 | 7151.89 | 7134.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 14:15:00 | 7239.00 | 7251.92 | 7217.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 15:00:00 | 7239.00 | 7251.92 | 7217.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 15:15:00 | 7220.00 | 7245.54 | 7217.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 09:15:00 | 7184.00 | 7245.54 | 7217.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 7165.00 | 7229.43 | 7212.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:00:00 | 7165.00 | 7229.43 | 7212.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 7140.00 | 7211.54 | 7206.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 11:00:00 | 7140.00 | 7211.54 | 7206.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 143 — SELL (started 2026-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 12:15:00 | 7191.50 | 7202.49 | 7202.78 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2026-04-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 13:15:00 | 7240.50 | 7210.09 | 7206.21 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 7093.50 | 7185.21 | 7196.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 12:15:00 | 7072.00 | 7150.22 | 7177.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 09:15:00 | 7120.00 | 7118.06 | 7151.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 11:15:00 | 7149.00 | 7123.28 | 7147.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 7149.00 | 7123.28 | 7147.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 11:30:00 | 7142.00 | 7123.28 | 7147.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 7105.50 | 7119.72 | 7143.85 | EMA400 retest candle locked (from downside) |

### Cycle 146 — BUY (started 2026-04-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 12:15:00 | 7193.50 | 7146.52 | 7145.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 7225.00 | 7172.73 | 7159.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 11:15:00 | 7065.50 | 7155.73 | 7154.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 11:15:00 | 7065.50 | 7155.73 | 7154.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 7065.50 | 7155.73 | 7154.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:00:00 | 7065.50 | 7155.73 | 7154.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2026-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 12:15:00 | 7088.00 | 7142.18 | 7148.08 | EMA200 below EMA400 |

### Cycle 148 — BUY (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 10:15:00 | 7229.50 | 7153.31 | 7147.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 7269.00 | 7176.45 | 7158.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 14:15:00 | 7188.00 | 7198.05 | 7174.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 14:15:00 | 7188.00 | 7198.05 | 7174.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 7188.00 | 7198.05 | 7174.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 14:30:00 | 7192.00 | 7198.05 | 7174.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 6982.50 | 7153.98 | 7158.85 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2026-05-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 10:15:00 | 7215.00 | 7139.75 | 7136.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 13:15:00 | 7308.50 | 7189.80 | 7161.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 10:15:00 | 7220.00 | 7282.65 | 7248.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 10:15:00 | 7220.00 | 7282.65 | 7248.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 7220.00 | 7282.65 | 7248.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 11:00:00 | 7220.00 | 7282.65 | 7248.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 7247.00 | 7275.52 | 7248.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 13:00:00 | 7268.00 | 7274.01 | 7250.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 14:15:00 | 7279.00 | 7265.81 | 7248.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-22 09:15:00 | 4690.25 | 2024-05-28 14:15:00 | 4769.10 | STOP_HIT | 1.00 | 1.68% |
| BUY | retest2 | 2024-05-22 09:45:00 | 4692.10 | 2024-05-28 14:15:00 | 4769.10 | STOP_HIT | 1.00 | 1.64% |
| BUY | retest2 | 2024-05-22 10:30:00 | 4688.15 | 2024-05-28 14:15:00 | 4769.10 | STOP_HIT | 1.00 | 1.73% |
| BUY | retest2 | 2024-05-22 11:00:00 | 4694.05 | 2024-05-28 14:15:00 | 4769.10 | STOP_HIT | 1.00 | 1.60% |
| BUY | retest2 | 2024-05-23 10:45:00 | 4765.20 | 2024-05-28 14:15:00 | 4769.10 | STOP_HIT | 1.00 | 0.08% |
| BUY | retest2 | 2024-05-23 11:15:00 | 4766.95 | 2024-05-28 14:15:00 | 4769.10 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2024-05-31 09:45:00 | 4719.60 | 2024-06-04 11:15:00 | 4483.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-31 10:45:00 | 4723.50 | 2024-06-04 11:15:00 | 4487.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 09:15:00 | 4709.90 | 2024-06-04 11:15:00 | 4474.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-05-31 09:45:00 | 4719.60 | 2024-06-04 15:15:00 | 4548.95 | STOP_HIT | 0.50 | 3.62% |
| SELL | retest2 | 2024-05-31 10:45:00 | 4723.50 | 2024-06-04 15:15:00 | 4548.95 | STOP_HIT | 0.50 | 3.70% |
| SELL | retest2 | 2024-06-03 09:15:00 | 4709.90 | 2024-06-04 15:15:00 | 4548.95 | STOP_HIT | 0.50 | 3.42% |
| BUY | retest2 | 2024-06-12 10:45:00 | 4785.75 | 2024-06-20 12:15:00 | 4871.60 | STOP_HIT | 1.00 | 1.79% |
| BUY | retest2 | 2024-06-13 13:00:00 | 4792.30 | 2024-06-20 12:15:00 | 4871.60 | STOP_HIT | 1.00 | 1.65% |
| SELL | retest2 | 2024-06-25 11:00:00 | 4827.50 | 2024-07-03 13:15:00 | 4686.00 | STOP_HIT | 1.00 | 2.93% |
| BUY | retest2 | 2024-07-04 11:45:00 | 4706.10 | 2024-07-19 09:15:00 | 4830.05 | STOP_HIT | 1.00 | 2.63% |
| BUY | retest2 | 2024-07-04 12:45:00 | 4698.30 | 2024-07-19 10:15:00 | 4846.00 | STOP_HIT | 1.00 | 3.14% |
| BUY | retest2 | 2024-07-04 14:00:00 | 4704.20 | 2024-07-19 10:15:00 | 4846.00 | STOP_HIT | 1.00 | 3.01% |
| BUY | retest2 | 2024-07-05 09:15:00 | 4710.95 | 2024-07-19 10:15:00 | 4846.00 | STOP_HIT | 1.00 | 2.87% |
| BUY | retest2 | 2024-07-12 12:15:00 | 4855.55 | 2024-07-19 10:15:00 | 4846.00 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2024-07-12 13:00:00 | 4865.00 | 2024-07-19 10:15:00 | 4846.00 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2024-07-15 10:00:00 | 4859.95 | 2024-07-19 10:15:00 | 4846.00 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2024-07-18 09:30:00 | 4865.45 | 2024-07-19 10:15:00 | 4846.00 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2024-07-18 13:00:00 | 4890.95 | 2024-07-19 10:15:00 | 4846.00 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2024-07-29 14:15:00 | 4970.00 | 2024-07-30 09:15:00 | 4935.75 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2024-07-30 11:30:00 | 4971.95 | 2024-07-30 12:15:00 | 4933.95 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2024-08-30 09:15:00 | 4936.35 | 2024-09-03 10:15:00 | 4916.00 | STOP_HIT | 1.00 | -0.41% |
| BUY | retest2 | 2024-09-02 12:15:00 | 4921.95 | 2024-09-03 10:15:00 | 4916.00 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2024-09-02 14:00:00 | 4923.65 | 2024-09-03 10:15:00 | 4916.00 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2024-09-03 10:00:00 | 4925.25 | 2024-09-03 10:15:00 | 4916.00 | STOP_HIT | 1.00 | -0.19% |
| SELL | retest2 | 2024-09-11 13:15:00 | 4740.00 | 2024-09-12 13:15:00 | 4814.60 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2024-09-11 14:00:00 | 4725.95 | 2024-09-12 13:15:00 | 4814.60 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2024-09-12 10:00:00 | 4728.05 | 2024-09-12 13:15:00 | 4814.60 | STOP_HIT | 1.00 | -1.83% |
| BUY | retest2 | 2024-09-23 12:30:00 | 4885.00 | 2024-09-23 14:15:00 | 4879.40 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2024-09-23 14:00:00 | 4886.65 | 2024-09-23 14:15:00 | 4879.40 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2024-10-09 11:45:00 | 4667.70 | 2024-10-10 09:15:00 | 4748.65 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-11-22 14:00:00 | 4963.25 | 2024-11-26 09:15:00 | 4880.95 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2024-12-04 10:00:00 | 4819.95 | 2024-12-05 15:15:00 | 4826.05 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2024-12-05 14:45:00 | 4800.35 | 2024-12-05 15:15:00 | 4826.05 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2024-12-13 10:15:00 | 4773.55 | 2024-12-13 13:15:00 | 4848.95 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2024-12-18 15:00:00 | 4747.60 | 2024-12-20 10:15:00 | 4813.80 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2024-12-19 09:15:00 | 4706.45 | 2024-12-20 10:15:00 | 4813.80 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2024-12-19 12:45:00 | 4745.20 | 2024-12-20 10:15:00 | 4813.80 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-01-13 09:15:00 | 5008.00 | 2025-01-15 12:15:00 | 5065.00 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-01-21 12:15:00 | 4996.70 | 2025-01-23 10:15:00 | 5056.50 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-01-21 13:45:00 | 4988.10 | 2025-01-23 10:15:00 | 5056.50 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-01-22 10:00:00 | 4989.65 | 2025-01-23 10:15:00 | 5056.50 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-01-22 15:15:00 | 4996.50 | 2025-01-23 10:15:00 | 5056.50 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-02-05 12:45:00 | 5429.70 | 2025-02-06 09:15:00 | 5354.20 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-02-05 15:00:00 | 5422.00 | 2025-02-06 09:15:00 | 5354.20 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-02-10 13:00:00 | 5353.70 | 2025-02-11 09:15:00 | 5086.01 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 13:45:00 | 5344.90 | 2025-02-11 09:15:00 | 5077.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 13:00:00 | 5353.70 | 2025-02-13 09:15:00 | 4818.33 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-10 13:45:00 | 5344.90 | 2025-02-13 09:15:00 | 4810.41 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-03-04 09:15:00 | 4815.15 | 2025-03-05 09:15:00 | 4945.45 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2025-03-04 10:00:00 | 4847.00 | 2025-03-05 09:15:00 | 4945.45 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-03-04 12:15:00 | 4837.30 | 2025-03-05 09:15:00 | 4945.45 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-03-04 15:00:00 | 4812.60 | 2025-03-05 09:15:00 | 4945.45 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2025-03-18 09:15:00 | 5079.40 | 2025-03-27 13:15:00 | 5366.85 | STOP_HIT | 1.00 | 5.66% |
| SELL | retest2 | 2025-03-28 13:45:00 | 5334.40 | 2025-04-02 14:15:00 | 5356.80 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-03-28 15:15:00 | 5345.25 | 2025-04-03 09:15:00 | 5350.75 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-04-01 10:45:00 | 5341.85 | 2025-04-03 10:15:00 | 5399.50 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-04-02 11:00:00 | 5341.85 | 2025-04-03 10:15:00 | 5399.50 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-04-02 13:30:00 | 5318.50 | 2025-04-03 10:15:00 | 5399.50 | STOP_HIT | 1.00 | -1.52% |
| SELL | retest2 | 2025-04-02 15:15:00 | 5322.00 | 2025-04-03 10:15:00 | 5399.50 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-04-28 14:30:00 | 5604.00 | 2025-05-06 10:15:00 | 5580.50 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2025-04-29 09:30:00 | 5617.00 | 2025-05-06 10:15:00 | 5580.50 | STOP_HIT | 1.00 | 0.65% |
| SELL | retest2 | 2025-04-29 10:00:00 | 5590.50 | 2025-05-06 10:15:00 | 5580.50 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2025-05-30 12:15:00 | 5344.50 | 2025-06-04 09:15:00 | 5380.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-06-02 10:00:00 | 5337.00 | 2025-06-04 09:15:00 | 5380.00 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-06-02 10:45:00 | 5344.50 | 2025-06-04 09:15:00 | 5380.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-06-02 11:15:00 | 5341.00 | 2025-06-04 09:15:00 | 5380.00 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-06-03 09:45:00 | 5332.00 | 2025-06-04 09:15:00 | 5380.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-06-03 11:00:00 | 5317.50 | 2025-06-04 09:15:00 | 5380.00 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest1 | 2025-06-10 09:30:00 | 5399.50 | 2025-06-10 14:15:00 | 5379.00 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest1 | 2025-06-19 09:15:00 | 5437.00 | 2025-06-26 09:15:00 | 5569.00 | STOP_HIT | 1.00 | 2.43% |
| SELL | retest2 | 2025-07-16 09:15:00 | 5575.00 | 2025-07-16 13:15:00 | 5621.50 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-07-16 11:15:00 | 5573.00 | 2025-07-16 13:15:00 | 5621.50 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-07-28 14:15:00 | 5372.50 | 2025-07-29 14:15:00 | 5474.00 | STOP_HIT | 1.00 | -1.89% |
| SELL | retest2 | 2025-07-28 15:15:00 | 5380.00 | 2025-07-29 14:15:00 | 5474.00 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-07-31 11:00:00 | 5458.00 | 2025-08-18 09:15:00 | 6003.80 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-09-18 09:15:00 | 6906.50 | 2025-09-29 14:15:00 | 6976.50 | STOP_HIT | 1.00 | 1.01% |
| BUY | retest2 | 2025-09-18 14:15:00 | 6905.00 | 2025-09-29 14:15:00 | 6976.50 | STOP_HIT | 1.00 | 1.04% |
| SELL | retest2 | 2025-10-01 09:45:00 | 6992.50 | 2025-10-01 11:15:00 | 7032.50 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-10-09 13:00:00 | 6901.00 | 2025-10-10 12:15:00 | 6970.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-10-09 13:30:00 | 6899.50 | 2025-10-10 12:15:00 | 6970.00 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-10-27 15:15:00 | 6905.00 | 2025-10-28 09:15:00 | 6963.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-11-10 09:15:00 | 6820.50 | 2025-11-12 10:15:00 | 6880.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-11-11 09:15:00 | 6798.00 | 2025-11-12 10:15:00 | 6880.00 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-11-11 14:30:00 | 6877.00 | 2025-11-12 10:15:00 | 6880.00 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-11-12 10:00:00 | 6863.00 | 2025-11-12 10:15:00 | 6880.00 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-11-19 14:00:00 | 6865.00 | 2025-11-27 10:15:00 | 7051.50 | STOP_HIT | 1.00 | 2.72% |
| BUY | retest2 | 2025-12-09 11:45:00 | 7158.00 | 2025-12-15 09:15:00 | 7125.50 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-12-10 09:15:00 | 7163.50 | 2025-12-15 09:15:00 | 7125.50 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-12-18 09:30:00 | 7053.50 | 2025-12-19 09:15:00 | 7184.00 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-12-18 14:00:00 | 7074.50 | 2025-12-19 09:15:00 | 7184.00 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-12-24 10:45:00 | 7303.00 | 2025-12-29 13:15:00 | 7261.50 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-12-26 09:15:00 | 7353.50 | 2025-12-29 13:15:00 | 7261.50 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-12-26 11:45:00 | 7313.00 | 2025-12-29 13:15:00 | 7261.50 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-12-29 11:45:00 | 7306.50 | 2025-12-29 13:15:00 | 7261.50 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2026-01-05 10:15:00 | 7476.50 | 2026-01-09 15:15:00 | 7510.00 | STOP_HIT | 1.00 | 0.45% |
| BUY | retest2 | 2026-01-05 11:30:00 | 7469.00 | 2026-01-09 15:15:00 | 7510.00 | STOP_HIT | 1.00 | 0.55% |
| BUY | retest2 | 2026-01-05 13:45:00 | 7469.00 | 2026-01-09 15:15:00 | 7510.00 | STOP_HIT | 1.00 | 0.55% |
| BUY | retest2 | 2026-01-05 14:15:00 | 7481.50 | 2026-01-09 15:15:00 | 7510.00 | STOP_HIT | 1.00 | 0.38% |
| SELL | retest2 | 2026-01-13 10:30:00 | 7429.00 | 2026-01-21 09:15:00 | 7057.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 7435.50 | 2026-01-21 09:15:00 | 7063.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:30:00 | 7402.00 | 2026-01-21 10:15:00 | 7031.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 10:30:00 | 7429.00 | 2026-01-22 09:15:00 | 7266.00 | STOP_HIT | 0.50 | 2.19% |
| SELL | retest2 | 2026-01-13 12:00:00 | 7435.50 | 2026-01-22 09:15:00 | 7266.00 | STOP_HIT | 0.50 | 2.28% |
| SELL | retest2 | 2026-01-13 12:30:00 | 7402.00 | 2026-01-22 09:15:00 | 7266.00 | STOP_HIT | 0.50 | 1.84% |
| BUY | retest2 | 2026-02-17 13:30:00 | 8023.50 | 2026-02-19 14:15:00 | 7900.50 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2026-02-17 14:30:00 | 8019.50 | 2026-02-19 14:15:00 | 7900.50 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2026-02-18 15:15:00 | 8020.00 | 2026-02-19 14:15:00 | 7900.50 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2026-02-27 11:45:00 | 8101.00 | 2026-02-27 14:15:00 | 8005.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-02-27 13:45:00 | 8101.50 | 2026-02-27 14:15:00 | 8005.00 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2026-02-27 14:30:00 | 8097.50 | 2026-02-27 15:15:00 | 8014.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest1 | 2026-03-05 10:15:00 | 7666.50 | 2026-03-05 15:15:00 | 7742.00 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest1 | 2026-03-05 11:00:00 | 7657.00 | 2026-03-05 15:15:00 | 7742.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest1 | 2026-03-05 13:45:00 | 7669.50 | 2026-03-05 15:15:00 | 7742.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2026-03-06 09:15:00 | 7681.00 | 2026-03-09 10:15:00 | 7296.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 09:15:00 | 7681.00 | 2026-03-10 09:15:00 | 7383.50 | STOP_HIT | 0.50 | 3.87% |
