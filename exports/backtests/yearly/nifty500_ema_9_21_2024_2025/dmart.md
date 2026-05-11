# Avenue Supermarts Ltd. (DMART)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 4396.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 139 |
| ALERT1 | 90 |
| ALERT2 | 89 |
| ALERT2_SKIP | 43 |
| ALERT3 | 218 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 107 |
| PARTIAL | 4 |
| TARGET_HIT | 4 |
| STOP_HIT | 109 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 117 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 26 / 91
- **Target hits / Stop hits / Partials:** 4 / 109 / 4
- **Avg / median % per leg:** -0.10% / -0.86%
- **Sum % (uncompounded):** -12.04%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 49 | 12 | 24.5% | 4 | 44 | 1 | 0.30% | 14.8% |
| BUY @ 2nd Alert (retest1) | 3 | 3 | 100.0% | 0 | 2 | 1 | 3.50% | 10.5% |
| BUY @ 3rd Alert (retest2) | 46 | 9 | 19.6% | 4 | 42 | 0 | 0.09% | 4.3% |
| SELL (all) | 68 | 14 | 20.6% | 0 | 65 | 3 | -0.39% | -26.8% |
| SELL @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.56% | -6.3% |
| SELL @ 3rd Alert (retest2) | 64 | 14 | 21.9% | 0 | 61 | 3 | -0.32% | -20.6% |
| retest1 (combined) | 7 | 3 | 42.9% | 0 | 6 | 1 | 0.61% | 4.2% |
| retest2 (combined) | 110 | 23 | 20.9% | 4 | 103 | 3 | -0.15% | -16.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 11:15:00 | 4712.00 | 4752.24 | 4753.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-13 13:15:00 | 4691.95 | 4732.69 | 4743.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-15 10:15:00 | 4631.00 | 4628.41 | 4665.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-15 11:00:00 | 4631.00 | 4628.41 | 4665.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 11:15:00 | 4626.55 | 4628.04 | 4661.96 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 13:15:00 | 4673.00 | 4666.65 | 4666.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 09:15:00 | 4692.00 | 4674.18 | 4670.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-17 14:15:00 | 4680.00 | 4683.78 | 4677.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-17 14:15:00 | 4680.00 | 4683.78 | 4677.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 14:15:00 | 4680.00 | 4683.78 | 4677.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-17 15:00:00 | 4680.00 | 4683.78 | 4677.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 15:15:00 | 4700.00 | 4687.03 | 4679.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 09:15:00 | 4679.50 | 4687.03 | 4679.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 4656.25 | 4680.87 | 4677.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-21 10:15:00 | 4694.50 | 4681.12 | 4678.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-24 12:15:00 | 4706.25 | 4757.35 | 4757.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2024-05-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 12:15:00 | 4706.25 | 4757.35 | 4757.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-24 14:15:00 | 4680.15 | 4734.35 | 4746.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 11:15:00 | 4505.00 | 4498.59 | 4551.77 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-29 12:00:00 | 4505.00 | 4498.59 | 4551.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 13:15:00 | 4548.00 | 4510.14 | 4547.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 14:00:00 | 4548.00 | 4510.14 | 4547.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 14:15:00 | 4487.20 | 4505.55 | 4542.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 15:15:00 | 4470.00 | 4505.55 | 4542.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 10:15:00 | 4455.00 | 4398.24 | 4396.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2024-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 10:15:00 | 4455.00 | 4398.24 | 4396.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-04 14:15:00 | 4558.00 | 4459.70 | 4427.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 10:15:00 | 4721.65 | 4743.43 | 4634.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 11:00:00 | 4721.65 | 4743.43 | 4634.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-10 10:15:00 | 4748.80 | 4730.67 | 4702.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 11:30:00 | 4781.45 | 4744.94 | 4711.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 10:15:00 | 4699.85 | 4733.89 | 4737.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2024-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 10:15:00 | 4699.85 | 4733.89 | 4737.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-12 13:15:00 | 4681.15 | 4710.34 | 4724.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-13 10:15:00 | 4700.20 | 4692.21 | 4709.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-13 10:15:00 | 4700.20 | 4692.21 | 4709.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 4700.20 | 4692.21 | 4709.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-13 14:00:00 | 4660.55 | 4686.13 | 4702.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-14 09:15:00 | 4738.05 | 4689.24 | 4699.11 | SL hit (close>static) qty=1.00 sl=4713.40 alert=retest2 |

### Cycle 6 — BUY (started 2024-06-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-14 12:15:00 | 4727.30 | 4707.67 | 4706.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-14 14:15:00 | 4741.90 | 4718.58 | 4711.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 14:15:00 | 4940.00 | 4986.20 | 4913.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-19 15:00:00 | 4940.00 | 4986.20 | 4913.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 15:15:00 | 4915.00 | 4971.96 | 4913.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 09:15:00 | 5003.10 | 4971.96 | 4913.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-20 10:15:00 | 4898.00 | 4953.65 | 4915.20 | SL hit (close<static) qty=1.00 sl=4901.35 alert=retest2 |

### Cycle 7 — SELL (started 2024-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-20 15:15:00 | 4866.90 | 4893.22 | 4896.27 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-06-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 13:15:00 | 4959.95 | 4903.10 | 4899.40 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 14:15:00 | 4786.85 | 4879.85 | 4889.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 14:15:00 | 4778.50 | 4813.39 | 4835.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 14:15:00 | 4798.90 | 4753.70 | 4787.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 14:15:00 | 4798.90 | 4753.70 | 4787.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 14:15:00 | 4798.90 | 4753.70 | 4787.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-26 14:45:00 | 4785.00 | 4753.70 | 4787.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 15:15:00 | 4812.00 | 4765.36 | 4789.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 09:15:00 | 4800.30 | 4765.36 | 4789.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 10:15:00 | 4810.60 | 4782.51 | 4793.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 11:15:00 | 4785.00 | 4782.51 | 4793.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-27 14:15:00 | 4913.30 | 4800.53 | 4797.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — BUY (started 2024-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-27 14:15:00 | 4913.30 | 4800.53 | 4797.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-27 15:15:00 | 4927.00 | 4825.82 | 4809.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-28 10:15:00 | 4836.30 | 4838.03 | 4818.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-28 11:00:00 | 4836.30 | 4838.03 | 4818.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 11:15:00 | 4822.10 | 4834.85 | 4818.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 11:30:00 | 4815.00 | 4834.85 | 4818.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 12:15:00 | 4790.85 | 4826.05 | 4816.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-28 12:45:00 | 4787.50 | 4826.05 | 4816.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 13:15:00 | 4786.15 | 4818.07 | 4813.48 | EMA400 retest candle locked (from upside) |

### Cycle 11 — SELL (started 2024-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-28 14:15:00 | 4701.85 | 4794.82 | 4803.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-01 09:15:00 | 4673.45 | 4757.47 | 4784.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 14:15:00 | 4736.20 | 4733.73 | 4760.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-01 15:00:00 | 4736.20 | 4733.73 | 4760.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 09:15:00 | 4750.00 | 4736.06 | 4756.64 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2024-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 10:15:00 | 4781.15 | 4764.60 | 4762.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 14:15:00 | 4800.65 | 4777.87 | 4769.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 09:15:00 | 4772.35 | 4779.97 | 4772.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-04 09:15:00 | 4772.35 | 4779.97 | 4772.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 4772.35 | 4779.97 | 4772.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-04 14:45:00 | 4808.45 | 4779.44 | 4774.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 14:45:00 | 4840.90 | 4796.88 | 4784.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 13:45:00 | 4808.00 | 4823.12 | 4821.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 15:00:00 | 4815.00 | 4821.50 | 4820.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-09 15:15:00 | 4800.15 | 4817.23 | 4818.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2024-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-09 15:15:00 | 4800.15 | 4817.23 | 4818.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-10 09:15:00 | 4792.25 | 4812.23 | 4816.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-10 11:15:00 | 4814.75 | 4803.26 | 4811.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-10 11:15:00 | 4814.75 | 4803.26 | 4811.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 4814.75 | 4803.26 | 4811.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 12:00:00 | 4814.75 | 4803.26 | 4811.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 12:15:00 | 4814.45 | 4805.50 | 4811.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 13:00:00 | 4814.45 | 4805.50 | 4811.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 13:15:00 | 4807.65 | 4805.93 | 4811.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 13:45:00 | 4824.85 | 4805.93 | 4811.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 14:15:00 | 4830.00 | 4810.74 | 4812.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-10 15:00:00 | 4830.00 | 4810.74 | 4812.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 15:15:00 | 4817.95 | 4812.18 | 4813.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:15:00 | 4897.20 | 4812.18 | 4813.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — BUY (started 2024-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-11 09:15:00 | 4910.10 | 4831.77 | 4822.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 09:15:00 | 4986.30 | 4901.30 | 4866.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 14:15:00 | 4929.95 | 4931.76 | 4898.32 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-12 15:15:00 | 4950.00 | 4931.76 | 4898.32 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 12:15:00 | 4976.70 | 5035.58 | 5024.59 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-07-19 12:15:00 | 4976.70 | 5035.58 | 5024.59 | SL hit (close<ema400) qty=1.00 sl=5024.59 alert=retest1 |

### Cycle 15 — SELL (started 2024-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 10:15:00 | 4994.70 | 5017.17 | 5018.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-22 11:15:00 | 4982.00 | 5010.14 | 5015.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 13:15:00 | 5009.70 | 5005.52 | 5012.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 13:15:00 | 5009.70 | 5005.52 | 5012.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 5009.70 | 5005.52 | 5012.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 5020.00 | 5005.52 | 5012.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 5011.00 | 5006.61 | 5012.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 15:00:00 | 5011.00 | 5006.61 | 5012.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 5010.50 | 5007.39 | 5011.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 5040.80 | 5007.39 | 5011.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2024-07-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 09:15:00 | 5050.00 | 5015.91 | 5015.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-23 14:15:00 | 5065.60 | 5035.32 | 5026.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-25 09:15:00 | 5118.25 | 5123.49 | 5089.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 09:15:00 | 5118.25 | 5123.49 | 5089.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 09:15:00 | 5118.25 | 5123.49 | 5089.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-25 14:45:00 | 5160.45 | 5134.66 | 5107.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-26 15:15:00 | 5075.65 | 5108.06 | 5109.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — SELL (started 2024-07-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-26 15:15:00 | 5075.65 | 5108.06 | 5109.53 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2024-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 10:15:00 | 5110.65 | 5103.27 | 5102.39 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2024-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 12:15:00 | 5050.05 | 5092.11 | 5097.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 14:15:00 | 5030.15 | 5072.73 | 5087.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 13:15:00 | 4971.30 | 4956.34 | 4993.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-01 14:00:00 | 4971.30 | 4956.34 | 4993.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 4999.95 | 4965.07 | 4993.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 15:00:00 | 4999.95 | 4965.07 | 4993.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 15:15:00 | 4990.00 | 4970.05 | 4993.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 09:15:00 | 4959.00 | 4970.05 | 4993.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 13:00:00 | 4966.95 | 4973.72 | 4987.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 10:00:00 | 4944.20 | 4948.27 | 4969.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-06 11:15:00 | 5027.40 | 4934.43 | 4936.31 | SL hit (close>static) qty=1.00 sl=5008.00 alert=retest2 |

### Cycle 20 — BUY (started 2024-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 12:15:00 | 5008.35 | 4949.21 | 4942.86 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2024-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 14:15:00 | 4902.80 | 4939.48 | 4939.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 15:15:00 | 4900.00 | 4931.58 | 4935.92 | Break + close below crossover candle low |

### Cycle 22 — BUY (started 2024-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 09:15:00 | 4996.05 | 4944.48 | 4941.38 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2024-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 10:15:00 | 4980.70 | 4992.66 | 4993.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 13:15:00 | 4957.35 | 4980.18 | 4987.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-14 10:15:00 | 4957.25 | 4955.04 | 4970.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-14 11:00:00 | 4957.25 | 4955.04 | 4970.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 11:15:00 | 4970.00 | 4958.03 | 4970.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-14 12:00:00 | 4970.00 | 4958.03 | 4970.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 12:15:00 | 4959.40 | 4958.31 | 4969.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 09:30:00 | 4949.30 | 4952.23 | 4963.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 13:15:00 | 4952.10 | 4945.88 | 4956.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-16 13:15:00 | 5008.85 | 4958.47 | 4961.54 | SL hit (close>static) qty=1.00 sl=4970.00 alert=retest2 |

### Cycle 24 — BUY (started 2024-08-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 14:15:00 | 5024.00 | 4971.58 | 4967.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 5067.25 | 5008.59 | 4991.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 11:15:00 | 5002.55 | 5008.87 | 4994.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 12:00:00 | 5002.55 | 5008.87 | 4994.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 5071.45 | 5090.53 | 5071.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 15:00:00 | 5071.45 | 5090.53 | 5071.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 15:15:00 | 5036.00 | 5079.62 | 5068.72 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2024-08-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 10:15:00 | 4939.75 | 5044.84 | 5054.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 11:15:00 | 4921.80 | 5020.23 | 5042.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 12:15:00 | 4969.85 | 4954.93 | 4985.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-26 12:45:00 | 4970.75 | 4954.93 | 4985.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 4958.95 | 4957.67 | 4977.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 10:15:00 | 4950.55 | 4957.67 | 4977.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-27 11:45:00 | 4947.95 | 4956.68 | 4973.60 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 09:30:00 | 4935.95 | 4947.05 | 4962.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-29 09:15:00 | 5084.25 | 4975.28 | 4966.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2024-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 09:15:00 | 5084.25 | 4975.28 | 4966.46 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2024-08-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-30 14:15:00 | 4923.50 | 4982.23 | 4984.67 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2024-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-02 15:15:00 | 5000.00 | 4980.82 | 4980.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-03 09:15:00 | 5021.20 | 4988.90 | 4983.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-03 14:15:00 | 4967.50 | 5003.38 | 4995.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-03 14:15:00 | 4967.50 | 5003.38 | 4995.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 14:15:00 | 4967.50 | 5003.38 | 4995.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-03 15:00:00 | 4967.50 | 5003.38 | 4995.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 15:15:00 | 4970.00 | 4996.70 | 4993.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-04 09:15:00 | 4991.00 | 4996.70 | 4993.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 4992.00 | 4995.76 | 4992.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 10:45:00 | 5003.00 | 5001.01 | 4995.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-12 10:15:00 | 5286.00 | 5324.95 | 5325.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2024-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-12 10:15:00 | 5286.00 | 5324.95 | 5325.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-12 14:15:00 | 5231.45 | 5294.48 | 5310.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-16 09:15:00 | 5227.65 | 5223.51 | 5255.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 09:15:00 | 5227.65 | 5223.51 | 5255.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 5227.65 | 5223.51 | 5255.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-16 09:45:00 | 5248.90 | 5223.51 | 5255.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 09:15:00 | 5213.40 | 5210.36 | 5231.53 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2024-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 13:15:00 | 5275.25 | 5238.10 | 5233.32 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2024-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 12:15:00 | 5201.95 | 5235.98 | 5236.75 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2024-09-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 14:15:00 | 5304.75 | 5246.09 | 5240.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 15:15:00 | 5335.00 | 5263.87 | 5249.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-20 12:15:00 | 5281.45 | 5290.34 | 5269.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-20 13:00:00 | 5281.45 | 5290.34 | 5269.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 5320.00 | 5295.58 | 5275.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 14:45:00 | 5228.00 | 5295.58 | 5275.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 11:15:00 | 5272.25 | 5295.10 | 5282.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 13:45:00 | 5294.95 | 5291.77 | 5283.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-23 15:15:00 | 5305.50 | 5291.18 | 5283.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-24 09:15:00 | 5247.10 | 5284.65 | 5282.19 | SL hit (close<static) qty=1.00 sl=5260.25 alert=retest2 |

### Cycle 33 — SELL (started 2024-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 14:15:00 | 5260.10 | 5320.66 | 5322.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-26 10:15:00 | 5215.70 | 5283.46 | 5303.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 12:15:00 | 5140.00 | 5137.05 | 5175.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-30 12:45:00 | 5145.10 | 5137.05 | 5175.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 4653.00 | 4546.62 | 4620.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 09:15:00 | 4231.50 | 4575.72 | 4594.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 11:15:00 | 4019.92 | 4086.23 | 4146.92 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-22 09:15:00 | 4019.00 | 3997.33 | 4040.65 | SL hit (close>ema200) qty=0.50 sl=3997.33 alert=retest2 |

### Cycle 34 — BUY (started 2024-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-23 12:15:00 | 4123.00 | 4054.71 | 4045.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-23 13:15:00 | 4161.75 | 4076.12 | 4056.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-24 11:15:00 | 4074.90 | 4108.02 | 4083.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-24 11:15:00 | 4074.90 | 4108.02 | 4083.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 11:15:00 | 4074.90 | 4108.02 | 4083.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-24 12:00:00 | 4074.90 | 4108.02 | 4083.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 12:15:00 | 4066.95 | 4099.81 | 4082.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-24 13:00:00 | 4066.95 | 4099.81 | 4082.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-24 13:15:00 | 4068.75 | 4093.60 | 4080.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-24 13:45:00 | 4074.00 | 4093.60 | 4080.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2024-10-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 09:15:00 | 3991.20 | 4062.78 | 4069.10 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2024-10-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-25 13:15:00 | 4085.05 | 4070.82 | 4070.25 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-10-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 14:15:00 | 4055.00 | 4067.65 | 4068.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-25 15:15:00 | 4040.00 | 4062.12 | 4066.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 09:15:00 | 4062.85 | 4062.27 | 4065.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-28 09:15:00 | 4062.85 | 4062.27 | 4065.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 4062.85 | 4062.27 | 4065.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-28 14:30:00 | 4009.05 | 4035.17 | 4050.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-12 13:15:00 | 3808.60 | 3861.06 | 3876.71 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-13 10:15:00 | 3834.95 | 3825.65 | 3851.72 | SL hit (close>ema200) qty=0.50 sl=3825.65 alert=retest2 |

### Cycle 38 — BUY (started 2024-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-28 09:15:00 | 3708.85 | 3673.69 | 3670.11 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2024-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-02 11:15:00 | 3642.90 | 3683.23 | 3688.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-02 12:15:00 | 3624.15 | 3671.42 | 3682.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 14:15:00 | 3678.00 | 3669.27 | 3679.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 14:15:00 | 3678.00 | 3669.27 | 3679.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 14:15:00 | 3678.00 | 3669.27 | 3679.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 15:00:00 | 3678.00 | 3669.27 | 3679.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 15:15:00 | 3687.00 | 3672.82 | 3680.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:15:00 | 3701.15 | 3672.82 | 3680.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 3679.95 | 3674.25 | 3680.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 09:30:00 | 3707.00 | 3674.25 | 3680.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 10:15:00 | 3692.95 | 3677.99 | 3681.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 11:00:00 | 3692.95 | 3677.99 | 3681.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 11:15:00 | 3703.35 | 3683.06 | 3683.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-03 11:45:00 | 3705.10 | 3683.06 | 3683.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — BUY (started 2024-12-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-03 12:15:00 | 3728.85 | 3692.22 | 3687.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-03 13:15:00 | 3828.75 | 3719.52 | 3700.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-06 09:15:00 | 3807.10 | 3842.44 | 3816.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-06 09:15:00 | 3807.10 | 3842.44 | 3816.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 3807.10 | 3842.44 | 3816.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:00:00 | 3807.10 | 3842.44 | 3816.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 3836.00 | 3841.15 | 3817.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-06 12:00:00 | 3843.75 | 3841.67 | 3820.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-06 14:15:00 | 3803.40 | 3827.28 | 3818.50 | SL hit (close<static) qty=1.00 sl=3805.60 alert=retest2 |

### Cycle 41 — SELL (started 2024-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 09:15:00 | 3773.45 | 3813.11 | 3813.37 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2024-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 14:15:00 | 3826.75 | 3811.87 | 3811.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 15:15:00 | 3835.00 | 3816.49 | 3813.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 11:15:00 | 3810.00 | 3819.10 | 3816.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-10 11:15:00 | 3810.00 | 3819.10 | 3816.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-10 11:15:00 | 3810.00 | 3819.10 | 3816.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-10 12:00:00 | 3810.00 | 3819.10 | 3816.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2024-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 12:15:00 | 3792.05 | 3813.69 | 3813.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-11 09:15:00 | 3722.50 | 3793.16 | 3803.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-12 12:15:00 | 3718.00 | 3717.23 | 3745.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-12 13:00:00 | 3718.00 | 3717.23 | 3745.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 15:15:00 | 3474.00 | 3456.76 | 3474.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 09:15:00 | 3501.15 | 3456.76 | 3474.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-24 09:15:00 | 3480.65 | 3461.54 | 3475.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-24 15:00:00 | 3462.50 | 3475.56 | 3478.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 12:00:00 | 3467.30 | 3466.96 | 3472.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-26 14:15:00 | 3512.45 | 3482.09 | 3478.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2024-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 14:15:00 | 3512.45 | 3482.09 | 3478.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 09:15:00 | 3567.60 | 3500.80 | 3487.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 09:15:00 | 3500.75 | 3538.85 | 3519.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-30 09:15:00 | 3500.75 | 3538.85 | 3519.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 3500.75 | 3538.85 | 3519.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 10:00:00 | 3500.75 | 3538.85 | 3519.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 10:15:00 | 3528.85 | 3536.85 | 3520.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 15:00:00 | 3615.75 | 3544.20 | 3527.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 10:30:00 | 3540.65 | 3540.91 | 3530.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 12:30:00 | 3545.45 | 3540.46 | 3531.75 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-01-03 09:15:00 | 3977.33 | 3705.95 | 3628.76 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2025-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 09:15:00 | 3783.00 | 3865.16 | 3870.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-10 09:15:00 | 3757.15 | 3805.90 | 3825.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 09:15:00 | 3509.85 | 3498.54 | 3566.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 10:00:00 | 3509.85 | 3498.54 | 3566.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 14:15:00 | 3573.00 | 3529.76 | 3557.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 15:00:00 | 3573.00 | 3529.76 | 3557.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 15:15:00 | 3570.00 | 3537.80 | 3558.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 09:15:00 | 3599.60 | 3537.80 | 3558.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 3600.35 | 3550.31 | 3562.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 09:30:00 | 3602.00 | 3550.31 | 3562.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2025-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-16 11:15:00 | 3602.30 | 3574.43 | 3572.12 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 3556.10 | 3599.34 | 3601.14 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2025-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 13:15:00 | 3600.70 | 3592.01 | 3590.87 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2025-01-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 09:15:00 | 3546.30 | 3584.81 | 3588.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 09:15:00 | 3531.60 | 3570.05 | 3578.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 11:15:00 | 3544.75 | 3533.60 | 3548.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:00:00 | 3544.75 | 3533.60 | 3548.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 3552.85 | 3537.45 | 3549.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 13:00:00 | 3552.85 | 3537.45 | 3549.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 13:15:00 | 3589.10 | 3547.78 | 3552.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:00:00 | 3589.10 | 3547.78 | 3552.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 14:15:00 | 3559.65 | 3550.15 | 3553.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 14:30:00 | 3581.20 | 3550.15 | 3553.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2025-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-28 15:15:00 | 3595.00 | 3559.12 | 3557.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 09:15:00 | 3613.50 | 3580.33 | 3571.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 13:15:00 | 3950.00 | 3957.65 | 3844.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-03 14:00:00 | 3950.00 | 3957.65 | 3844.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 10:15:00 | 3902.65 | 3924.72 | 3895.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 11:00:00 | 3902.65 | 3924.72 | 3895.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 11:15:00 | 3863.95 | 3912.57 | 3893.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-05 12:00:00 | 3863.95 | 3912.57 | 3893.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 12:15:00 | 3898.45 | 3909.75 | 3893.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-05 13:15:00 | 3905.75 | 3909.75 | 3893.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-05 15:15:00 | 3850.00 | 3890.64 | 3888.32 | SL hit (close<static) qty=1.00 sl=3860.25 alert=retest2 |

### Cycle 51 — SELL (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 09:15:00 | 3844.00 | 3881.31 | 3884.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 10:15:00 | 3831.00 | 3871.25 | 3879.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 14:15:00 | 3748.80 | 3746.72 | 3786.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 15:00:00 | 3748.80 | 3746.72 | 3786.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 09:15:00 | 3777.05 | 3753.04 | 3782.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-10 09:45:00 | 3799.80 | 3753.04 | 3782.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 10:15:00 | 3776.30 | 3757.69 | 3782.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 11:30:00 | 3758.20 | 3755.68 | 3778.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-19 09:15:00 | 3695.00 | 3664.91 | 3660.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2025-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 09:15:00 | 3695.00 | 3664.91 | 3660.88 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 11:15:00 | 3633.65 | 3661.55 | 3664.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-20 12:15:00 | 3621.20 | 3653.48 | 3660.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-03 12:15:00 | 3427.95 | 3415.48 | 3459.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 13:00:00 | 3427.95 | 3415.48 | 3459.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 15:15:00 | 3469.90 | 3433.60 | 3457.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-04 09:15:00 | 3439.35 | 3433.60 | 3457.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-05 12:15:00 | 3485.00 | 3447.21 | 3446.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 3485.00 | 3447.21 | 3446.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 3512.80 | 3474.23 | 3461.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-11 09:15:00 | 3578.50 | 3594.72 | 3573.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 09:15:00 | 3578.50 | 3594.72 | 3573.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 3578.50 | 3594.72 | 3573.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-11 10:45:00 | 3593.00 | 3595.56 | 3575.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-25 11:15:00 | 3952.30 | 3927.52 | 3907.96 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2025-04-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-01 15:15:00 | 4016.00 | 4039.62 | 4042.16 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 10:15:00 | 4061.75 | 4044.05 | 4043.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 12:15:00 | 4071.00 | 4051.22 | 4047.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-04 09:15:00 | 4012.60 | 4112.09 | 4097.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-04 09:15:00 | 4012.60 | 4112.09 | 4097.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 4012.60 | 4112.09 | 4097.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 10:00:00 | 4012.60 | 4112.09 | 4097.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 4037.90 | 4097.25 | 4092.48 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 4041.20 | 4086.04 | 4087.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 4009.30 | 4062.89 | 4076.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 14:15:00 | 4021.65 | 3999.18 | 4028.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-07 14:15:00 | 4021.65 | 3999.18 | 4028.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 14:15:00 | 4021.65 | 3999.18 | 4028.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-07 15:00:00 | 4021.65 | 3999.18 | 4028.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 4033.00 | 4005.94 | 4028.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 09:15:00 | 4042.40 | 4005.94 | 4028.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 4053.45 | 4015.45 | 4031.02 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2025-04-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 12:15:00 | 4058.10 | 4043.20 | 4041.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-09 10:15:00 | 4100.20 | 4056.35 | 4048.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 13:15:00 | 4115.50 | 4135.81 | 4107.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-11 14:00:00 | 4115.50 | 4135.81 | 4107.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 10:15:00 | 4089.70 | 4126.43 | 4112.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-15 11:00:00 | 4089.70 | 4126.43 | 4112.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 11:15:00 | 4150.90 | 4131.33 | 4115.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 12:15:00 | 4153.90 | 4131.33 | 4115.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 12:15:00 | 4375.00 | 4430.00 | 4436.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2025-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 12:15:00 | 4375.00 | 4430.00 | 4436.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-28 15:15:00 | 4360.00 | 4386.02 | 4404.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-06 09:15:00 | 4040.50 | 4033.73 | 4097.28 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-06 14:15:00 | 3999.10 | 4021.80 | 4070.96 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-07 09:30:00 | 3995.00 | 4006.73 | 4051.39 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-07 10:00:00 | 3973.60 | 4006.73 | 4051.39 | SELL ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-05-07 15:00:00 | 3997.90 | 4005.93 | 4034.36 | SELL ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 4003.40 | 4004.07 | 4028.50 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-05-08 11:15:00 | 4053.80 | 4015.94 | 4029.77 | SL hit (close>ema400) qty=1.00 sl=4029.77 alert=retest1 |

### Cycle 60 — BUY (started 2025-05-12 13:15:00)

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

### Cycle 61 — SELL (started 2025-05-20 13:15:00)

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

### Cycle 62 — BUY (started 2025-05-23 11:15:00)

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

### Cycle 63 — SELL (started 2025-05-27 11:15:00)

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

### Cycle 64 — BUY (started 2025-06-03 13:15:00)

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

### Cycle 65 — SELL (started 2025-06-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 12:15:00 | 4090.90 | 4149.44 | 4156.18 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 4087.00 | 4066.61 | 4064.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 09:15:00 | 4179.00 | 4091.27 | 4077.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 4185.40 | 4195.15 | 4158.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 12:45:00 | 4181.70 | 4195.15 | 4158.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 4185.70 | 4195.17 | 4170.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 11:30:00 | 4246.60 | 4206.26 | 4179.74 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 09:15:00 | 4263.30 | 4370.40 | 4374.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2025-07-03 09:15:00)

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

### Cycle 68 — BUY (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 13:15:00 | 4100.90 | 4071.71 | 4069.77 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2025-07-17 12:15:00)

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

### Cycle 70 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 4046.30 | 4038.55 | 4037.87 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 4029.50 | 4036.74 | 4037.11 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2025-07-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 13:15:00 | 4049.10 | 4039.58 | 4038.36 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2025-07-24 14:15:00)

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

### Cycle 74 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 4019.00 | 3987.80 | 3986.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 11:15:00 | 4206.10 | 4038.71 | 4010.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 10:15:00 | 4242.80 | 4258.48 | 4199.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 11:00:00 | 4242.80 | 4258.48 | 4199.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 14:15:00 | 4186.00 | 4228.71 | 4202.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 15:00:00 | 4186.00 | 4228.71 | 4202.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 4216.60 | 4226.29 | 4203.89 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2025-08-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 13:15:00 | 4163.80 | 4191.89 | 4193.82 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2025-08-04 15:15:00)

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

### Cycle 77 — SELL (started 2025-08-08 11:15:00)

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

### Cycle 78 — BUY (started 2025-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 11:15:00 | 4257.60 | 4217.01 | 4215.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 4276.70 | 4245.95 | 4232.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 15:15:00 | 4347.80 | 4351.40 | 4316.09 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-08-18 09:15:00 | 4474.00 | 4351.40 | 4316.09 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-20 09:15:00 | 4697.70 | 4629.06 | 4547.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-08-21 11:15:00 | 4695.90 | 4708.06 | 4648.23 | SL hit (close<ema200) qty=0.50 sl=4708.06 alert=retest1 |

### Cycle 79 — SELL (started 2025-08-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 09:15:00 | 4680.00 | 4708.87 | 4709.19 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2025-08-29 10:15:00)

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

### Cycle 81 — SELL (started 2025-09-02 10:15:00)

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

### Cycle 82 — BUY (started 2025-09-03 14:15:00)

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

### Cycle 83 — SELL (started 2025-09-08 11:15:00)

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

### Cycle 84 — BUY (started 2025-09-15 13:15:00)

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

### Cycle 85 — SELL (started 2025-09-22 13:15:00)

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

### Cycle 86 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 4353.40 | 4320.86 | 4317.10 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-10-13 09:15:00)

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

### Cycle 88 — BUY (started 2025-10-16 09:15:00)

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

### Cycle 89 — SELL (started 2025-10-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 13:15:00 | 4271.00 | 4282.61 | 4282.63 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 4291.50 | 4282.12 | 4282.12 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 10:15:00 | 4280.30 | 4281.76 | 4281.95 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2025-10-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 11:15:00 | 4290.10 | 4283.42 | 4282.69 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-10-23 14:15:00)

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

### Cycle 94 — BUY (started 2025-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 12:15:00 | 4185.00 | 4176.15 | 4176.14 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2025-11-06 09:15:00)

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

### Cycle 96 — BUY (started 2025-11-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 11:15:00 | 4070.10 | 4060.56 | 4060.05 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 15:15:00 | 4054.00 | 4059.47 | 4059.90 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2025-11-13 09:15:00)

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

### Cycle 99 — SELL (started 2025-11-14 10:15:00)

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

### Cycle 100 — BUY (started 2025-11-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 09:15:00 | 4063.40 | 4031.86 | 4029.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 13:15:00 | 4088.70 | 4060.11 | 4044.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 4050.00 | 4065.11 | 4051.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 4050.00 | 4065.11 | 4051.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 4050.00 | 4065.11 | 4051.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:00:00 | 4050.00 | 4065.11 | 4051.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 4058.20 | 4063.73 | 4052.23 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2025-11-24 09:15:00)

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

### Cycle 102 — BUY (started 2025-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 15:15:00 | 4015.00 | 4011.45 | 4011.34 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 12:15:00 | 3987.30 | 4007.11 | 4009.54 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2025-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 10:15:00 | 4020.60 | 4011.82 | 4010.69 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2025-11-28 13:15:00)

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

### Cycle 106 — BUY (started 2025-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 14:15:00 | 3954.70 | 3935.36 | 3933.38 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-12-08 10:15:00)

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

### Cycle 108 — BUY (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 09:15:00 | 3892.70 | 3842.46 | 3840.63 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2025-12-17 11:15:00)

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

### Cycle 110 — BUY (started 2025-12-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 09:15:00 | 3863.90 | 3811.27 | 3806.97 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2025-12-24 12:15:00)

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

### Cycle 112 — BUY (started 2026-01-07 10:15:00)

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

### Cycle 113 — SELL (started 2026-01-16 11:15:00)

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

### Cycle 114 — BUY (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 10:15:00 | 3727.30 | 3700.98 | 3698.94 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2026-01-23 14:15:00)

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

### Cycle 116 — BUY (started 2026-01-28 11:15:00)

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

### Cycle 117 — SELL (started 2026-01-29 12:15:00)

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

### Cycle 118 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 3720.00 | 3667.05 | 3666.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 11:15:00 | 3778.30 | 3700.32 | 3682.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 3873.10 | 3886.23 | 3838.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 10:00:00 | 3873.10 | 3886.23 | 3838.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 3972.80 | 3986.04 | 3963.22 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 3890.10 | 3950.66 | 3954.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 10:15:00 | 3880.30 | 3906.96 | 3926.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 3916.00 | 3889.53 | 3906.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 3916.00 | 3889.53 | 3906.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 3916.00 | 3889.53 | 3906.22 | EMA400 retest candle locked (from downside) |

### Cycle 120 — BUY (started 2026-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 14:15:00 | 3920.20 | 3905.74 | 3904.55 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2026-02-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 10:15:00 | 3881.00 | 3901.63 | 3903.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 11:15:00 | 3861.70 | 3893.65 | 3899.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 10:15:00 | 3881.90 | 3863.43 | 3878.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 10:15:00 | 3881.90 | 3863.43 | 3878.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 3881.90 | 3863.43 | 3878.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:00:00 | 3881.90 | 3863.43 | 3878.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 11:15:00 | 3869.70 | 3864.68 | 3877.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 11:30:00 | 3886.60 | 3864.68 | 3877.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 13:15:00 | 3881.70 | 3867.88 | 3876.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 14:00:00 | 3881.70 | 3867.88 | 3876.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 14:15:00 | 3874.60 | 3869.22 | 3876.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 15:15:00 | 3850.00 | 3869.22 | 3876.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:45:00 | 3846.20 | 3862.50 | 3871.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-25 09:15:00 | 3903.00 | 3847.34 | 3848.62 | SL hit (close>static) qty=1.00 sl=3884.40 alert=retest2 |

### Cycle 122 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 3940.90 | 3866.05 | 3857.01 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2026-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 12:15:00 | 3838.40 | 3861.81 | 3863.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 3811.80 | 3852.59 | 3858.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-27 10:15:00 | 3871.90 | 3856.46 | 3859.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 10:15:00 | 3871.90 | 3856.46 | 3859.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 10:15:00 | 3871.90 | 3856.46 | 3859.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-27 11:00:00 | 3871.90 | 3856.46 | 3859.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 11:15:00 | 3860.80 | 3857.32 | 3859.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 12:30:00 | 3851.70 | 3858.58 | 3860.20 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 15:00:00 | 3841.20 | 3856.03 | 3858.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-02 09:15:00 | 3797.10 | 3855.83 | 3858.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 15:15:00 | 3850.00 | 3805.43 | 3800.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2026-03-05 15:15:00)

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

### Cycle 125 — SELL (started 2026-03-13 13:15:00)

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

### Cycle 126 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 3854.40 | 3838.23 | 3836.31 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2026-03-19 13:15:00)

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

### Cycle 128 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 3869.40 | 3769.29 | 3759.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 3910.80 | 3797.59 | 3773.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 14:15:00 | 3896.20 | 3901.23 | 3863.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 15:00:00 | 3896.20 | 3901.23 | 3863.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 4418.10 | 4468.36 | 4415.06 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2026-04-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-10 10:15:00 | 4357.90 | 4395.45 | 4398.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 09:15:00 | 4343.10 | 4378.29 | 4387.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 10:15:00 | 4383.10 | 4379.25 | 4387.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 10:15:00 | 4383.10 | 4379.25 | 4387.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 4383.10 | 4379.25 | 4387.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 10:45:00 | 4381.90 | 4379.25 | 4387.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 11:15:00 | 4440.00 | 4391.40 | 4391.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 12:00:00 | 4440.00 | 4391.40 | 4391.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2026-04-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 12:15:00 | 4449.10 | 4402.94 | 4397.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 14:15:00 | 4478.70 | 4451.92 | 4432.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 09:15:00 | 4450.10 | 4454.21 | 4437.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-16 10:00:00 | 4450.10 | 4454.21 | 4437.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 4447.90 | 4452.95 | 4438.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:45:00 | 4439.00 | 4452.95 | 4438.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 11:15:00 | 4425.80 | 4447.52 | 4437.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 12:00:00 | 4425.80 | 4447.52 | 4437.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 12:15:00 | 4390.30 | 4436.07 | 4432.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 13:00:00 | 4390.30 | 4436.07 | 4432.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2026-04-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 14:15:00 | 4427.60 | 4430.29 | 4430.50 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2026-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-16 15:15:00 | 4436.50 | 4431.53 | 4431.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 09:15:00 | 4521.10 | 4449.44 | 4439.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 11:15:00 | 4579.00 | 4581.32 | 4535.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 12:00:00 | 4579.00 | 4581.32 | 4535.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 4554.00 | 4587.07 | 4568.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 13:30:00 | 4552.10 | 4587.07 | 4568.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 4590.90 | 4587.83 | 4570.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 12:00:00 | 4606.50 | 4584.59 | 4573.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 11:15:00 | 4541.60 | 4571.95 | 4573.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2026-04-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 11:15:00 | 4541.60 | 4571.95 | 4573.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 12:15:00 | 4538.40 | 4565.24 | 4570.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 4495.40 | 4466.70 | 4499.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 4495.40 | 4466.70 | 4499.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 4495.40 | 4466.70 | 4499.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:30:00 | 4520.00 | 4466.70 | 4499.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 4514.00 | 4476.16 | 4501.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:30:00 | 4527.20 | 4476.16 | 4501.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 4485.80 | 4478.09 | 4499.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 12:15:00 | 4521.00 | 4478.09 | 4499.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 12:15:00 | 4514.00 | 4485.27 | 4500.97 | EMA400 retest candle locked (from downside) |

### Cycle 134 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 4544.00 | 4508.75 | 4507.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 11:15:00 | 4570.10 | 4521.02 | 4513.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 14:15:00 | 4537.10 | 4541.27 | 4526.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 15:00:00 | 4537.10 | 4541.27 | 4526.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 4524.70 | 4578.98 | 4562.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 14:45:00 | 4605.80 | 4563.42 | 4558.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 15:15:00 | 4614.00 | 4563.42 | 4558.39 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 09:15:00 | 4387.00 | 4536.23 | 4547.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 09:15:00 | 4387.00 | 4536.23 | 4547.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 11:15:00 | 4361.00 | 4485.90 | 4521.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 10:15:00 | 4370.00 | 4365.96 | 4405.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-06 10:45:00 | 4373.60 | 4365.96 | 4405.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 4418.00 | 4376.37 | 4406.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 12:00:00 | 4418.00 | 4376.37 | 4406.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 12:15:00 | 4420.80 | 4385.26 | 4408.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 12:45:00 | 4422.00 | 4385.26 | 4408.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 13:15:00 | 4422.70 | 4392.75 | 4409.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 13:30:00 | 4421.90 | 4392.75 | 4409.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 4405.90 | 4404.45 | 4411.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 11:15:00 | 4373.60 | 4399.76 | 4408.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 09:15:00 | 4439.00 | 4400.49 | 4403.81 | SL hit (close>static) qty=1.00 sl=4437.20 alert=retest2 |

### Cycle 136 — BUY (started 2026-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 11:15:00 | 4409.70 | 4405.98 | 4405.96 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 4401.80 | 4405.15 | 4405.58 | EMA200 below EMA400 |

### Cycle 138 — BUY (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 13:15:00 | 4419.10 | 4407.94 | 4406.81 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2026-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 15:15:00 | 4396.10 | 4404.12 | 4405.18 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-21 10:15:00 | 4694.50 | 2024-05-24 12:15:00 | 4706.25 | STOP_HIT | 1.00 | 0.25% |
| SELL | retest2 | 2024-05-29 15:15:00 | 4470.00 | 2024-06-04 10:15:00 | 4455.00 | STOP_HIT | 1.00 | 0.34% |
| BUY | retest2 | 2024-06-10 11:30:00 | 4781.45 | 2024-06-12 10:15:00 | 4699.85 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-06-13 14:00:00 | 4660.55 | 2024-06-14 09:15:00 | 4738.05 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2024-06-20 09:15:00 | 5003.10 | 2024-06-20 10:15:00 | 4898.00 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2024-06-27 11:15:00 | 4785.00 | 2024-06-27 14:15:00 | 4913.30 | STOP_HIT | 1.00 | -2.68% |
| BUY | retest2 | 2024-07-04 14:45:00 | 4808.45 | 2024-07-09 15:15:00 | 4800.15 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2024-07-05 14:45:00 | 4840.90 | 2024-07-09 15:15:00 | 4800.15 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-07-09 13:45:00 | 4808.00 | 2024-07-09 15:15:00 | 4800.15 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2024-07-09 15:00:00 | 4815.00 | 2024-07-09 15:15:00 | 4800.15 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest1 | 2024-07-12 15:15:00 | 4950.00 | 2024-07-19 12:15:00 | 4976.70 | STOP_HIT | 1.00 | 0.54% |
| BUY | retest2 | 2024-07-19 15:00:00 | 5024.15 | 2024-07-22 10:15:00 | 4994.70 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2024-07-22 10:00:00 | 5019.00 | 2024-07-22 10:15:00 | 4994.70 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2024-07-25 14:45:00 | 5160.45 | 2024-07-26 15:15:00 | 5075.65 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-08-02 09:15:00 | 4959.00 | 2024-08-06 11:15:00 | 5027.40 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-08-02 13:00:00 | 4966.95 | 2024-08-06 11:15:00 | 5027.40 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2024-08-05 10:00:00 | 4944.20 | 2024-08-06 11:15:00 | 5027.40 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2024-08-16 09:30:00 | 4949.30 | 2024-08-16 13:15:00 | 5008.85 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2024-08-16 13:15:00 | 4952.10 | 2024-08-16 13:15:00 | 5008.85 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2024-08-27 10:15:00 | 4950.55 | 2024-08-29 09:15:00 | 5084.25 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2024-08-27 11:45:00 | 4947.95 | 2024-08-29 09:15:00 | 5084.25 | STOP_HIT | 1.00 | -2.75% |
| SELL | retest2 | 2024-08-28 09:30:00 | 4935.95 | 2024-08-29 09:15:00 | 5084.25 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2024-09-04 10:45:00 | 5003.00 | 2024-09-12 10:15:00 | 5286.00 | STOP_HIT | 1.00 | 5.66% |
| BUY | retest2 | 2024-09-23 13:45:00 | 5294.95 | 2024-09-24 09:15:00 | 5247.10 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2024-09-23 15:15:00 | 5305.50 | 2024-09-24 09:15:00 | 5247.10 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2024-09-24 10:45:00 | 5365.10 | 2024-09-25 14:15:00 | 5260.10 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2024-10-14 09:15:00 | 4231.50 | 2024-10-18 11:15:00 | 4019.92 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-14 09:15:00 | 4231.50 | 2024-10-22 09:15:00 | 4019.00 | STOP_HIT | 0.50 | 5.02% |
| SELL | retest2 | 2024-10-28 14:30:00 | 4009.05 | 2024-11-12 13:15:00 | 3808.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-28 14:30:00 | 4009.05 | 2024-11-13 10:15:00 | 3834.95 | STOP_HIT | 0.50 | 4.34% |
| BUY | retest2 | 2024-12-06 12:00:00 | 3843.75 | 2024-12-06 14:15:00 | 3803.40 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-12-24 15:00:00 | 3462.50 | 2024-12-26 14:15:00 | 3512.45 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2024-12-26 12:00:00 | 3467.30 | 2024-12-26 14:15:00 | 3512.45 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-12-30 15:00:00 | 3615.75 | 2025-01-03 09:15:00 | 3977.33 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-31 10:30:00 | 3540.65 | 2025-01-03 09:15:00 | 3894.72 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-31 12:30:00 | 3545.45 | 2025-01-03 09:15:00 | 3899.99 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-02-05 13:15:00 | 3905.75 | 2025-02-05 15:15:00 | 3850.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-02-10 11:30:00 | 3758.20 | 2025-02-19 09:15:00 | 3695.00 | STOP_HIT | 1.00 | 1.68% |
| SELL | retest2 | 2025-03-04 09:15:00 | 3439.35 | 2025-03-05 12:15:00 | 3485.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2025-03-11 10:45:00 | 3593.00 | 2025-03-25 11:15:00 | 3952.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-04-15 12:15:00 | 4153.90 | 2025-04-25 12:15:00 | 4375.00 | STOP_HIT | 1.00 | 5.32% |
| SELL | retest1 | 2025-05-06 14:15:00 | 3999.10 | 2025-05-08 11:15:00 | 4053.80 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest1 | 2025-05-07 09:30:00 | 3995.00 | 2025-05-08 11:15:00 | 4053.80 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest1 | 2025-05-07 10:00:00 | 3973.60 | 2025-05-08 11:15:00 | 4053.80 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest1 | 2025-05-07 15:00:00 | 3997.90 | 2025-05-08 11:15:00 | 4053.80 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-05-08 14:45:00 | 3980.20 | 2025-05-12 13:15:00 | 4021.80 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-05-09 09:45:00 | 3955.70 | 2025-05-12 13:15:00 | 4021.80 | STOP_HIT | 1.00 | -1.67% |
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
| SELL | retest2 | 2026-02-20 15:15:00 | 3850.00 | 2026-02-25 09:15:00 | 3903.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2026-02-23 09:45:00 | 3846.20 | 2026-02-25 09:15:00 | 3903.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2026-02-27 12:30:00 | 3851.70 | 2026-03-05 15:15:00 | 3850.00 | STOP_HIT | 1.00 | 0.04% |
| SELL | retest2 | 2026-02-27 15:00:00 | 3841.20 | 2026-03-05 15:15:00 | 3850.00 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2026-03-02 09:15:00 | 3797.10 | 2026-03-05 15:15:00 | 3850.00 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2026-03-11 13:15:00 | 3938.70 | 2026-03-12 09:15:00 | 3902.60 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2026-03-12 11:30:00 | 3941.00 | 2026-03-13 12:15:00 | 3872.60 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2026-04-22 12:00:00 | 4606.50 | 2026-04-23 11:15:00 | 4541.60 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2026-04-30 14:45:00 | 4605.80 | 2026-05-04 09:15:00 | 4387.00 | STOP_HIT | 1.00 | -4.75% |
| BUY | retest2 | 2026-04-30 15:15:00 | 4614.00 | 2026-05-04 09:15:00 | 4387.00 | STOP_HIT | 1.00 | -4.92% |
| SELL | retest2 | 2026-05-07 11:15:00 | 4373.60 | 2026-05-08 09:15:00 | 4439.00 | STOP_HIT | 1.00 | -1.50% |
