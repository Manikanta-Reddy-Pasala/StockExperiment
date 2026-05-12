# Alkem Laboratories Ltd. (ALKEM)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 5560.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 69 |
| ALERT1 | 49 |
| ALERT2 | 48 |
| ALERT2_SKIP | 22 |
| ALERT3 | 131 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 85 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 90 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 91 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 20 / 71
- **Target hits / Stop hits / Partials:** 0 / 90 / 1
- **Avg / median % per leg:** -0.49% / -0.80%
- **Sum % (uncompounded):** -44.28%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 42 | 16 | 38.1% | 0 | 42 | 0 | 0.03% | 1.3% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.00% | -4.0% |
| BUY @ 3rd Alert (retest2) | 38 | 16 | 42.1% | 0 | 38 | 0 | 0.14% | 5.2% |
| SELL (all) | 49 | 4 | 8.2% | 0 | 48 | 1 | -0.93% | -45.5% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.87% | -0.9% |
| SELL @ 3rd Alert (retest2) | 48 | 4 | 8.3% | 0 | 47 | 1 | -0.93% | -44.7% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.97% | -4.9% |
| retest2 (combined) | 86 | 20 | 23.3% | 0 | 85 | 1 | -0.46% | -39.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 5019.00 | 4975.20 | 4974.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 5115.00 | 5017.31 | 4995.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 09:15:00 | 5086.00 | 5088.28 | 5049.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 10:00:00 | 5086.00 | 5088.28 | 5049.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 10:15:00 | 5071.50 | 5084.92 | 5051.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:00:00 | 5071.50 | 5084.92 | 5051.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 5325.50 | 5248.18 | 5204.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 10:30:00 | 5391.50 | 5270.64 | 5218.98 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 13:00:00 | 5388.50 | 5307.79 | 5245.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:30:00 | 5377.00 | 5314.11 | 5291.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:00:00 | 5379.00 | 5314.11 | 5291.55 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 5321.00 | 5321.97 | 5301.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-21 12:45:00 | 5325.00 | 5321.97 | 5301.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 5376.50 | 5351.89 | 5323.74 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-22 13:15:00 | 5294.00 | 5308.53 | 5310.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 13:15:00 | 5294.00 | 5308.53 | 5310.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 13:15:00 | 5172.00 | 5229.31 | 5244.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 14:15:00 | 5293.50 | 5242.15 | 5249.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 14:15:00 | 5293.50 | 5242.15 | 5249.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 5293.50 | 5242.15 | 5249.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 5293.50 | 5242.15 | 5249.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — BUY (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 15:15:00 | 5325.00 | 5258.72 | 5255.93 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 5098.50 | 5226.67 | 5241.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 5049.50 | 5114.56 | 5166.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 14:15:00 | 5096.00 | 5084.52 | 5128.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 15:00:00 | 5096.00 | 5084.52 | 5128.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 4995.00 | 5068.37 | 5114.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 11:30:00 | 4973.50 | 5041.70 | 5093.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 15:15:00 | 4980.00 | 5019.76 | 5069.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 10:15:00 | 4942.00 | 4894.09 | 4889.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 4942.00 | 4894.09 | 4889.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 12:15:00 | 4966.50 | 4917.68 | 4901.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 11:15:00 | 4918.50 | 4940.36 | 4923.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 11:15:00 | 4918.50 | 4940.36 | 4923.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 4918.50 | 4940.36 | 4923.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:00:00 | 4918.50 | 4940.36 | 4923.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 4910.00 | 4934.29 | 4921.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:30:00 | 4918.50 | 4934.29 | 4921.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 4898.50 | 4913.13 | 4914.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 4870.50 | 4904.15 | 4909.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 13:15:00 | 4823.50 | 4815.04 | 4849.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 14:00:00 | 4823.50 | 4815.04 | 4849.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 4853.00 | 4822.63 | 4849.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 4853.00 | 4822.63 | 4849.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 15:15:00 | 4850.50 | 4828.21 | 4849.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 09:15:00 | 4832.00 | 4828.21 | 4849.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 4810.00 | 4824.57 | 4846.25 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 4867.00 | 4853.84 | 4852.17 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 4840.00 | 4850.85 | 4851.09 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 13:15:00 | 4855.00 | 4851.68 | 4851.44 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-06-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 14:15:00 | 4821.00 | 4845.55 | 4848.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 10:15:00 | 4791.00 | 4823.04 | 4836.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 14:15:00 | 4810.00 | 4807.77 | 4823.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-18 14:45:00 | 4814.00 | 4807.77 | 4823.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 4780.00 | 4800.97 | 4817.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:30:00 | 4763.00 | 4794.28 | 4813.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 12:15:00 | 4761.00 | 4781.39 | 4795.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 13:15:00 | 4763.00 | 4778.31 | 4792.44 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 14:15:00 | 4765.00 | 4775.95 | 4790.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 4774.50 | 4775.66 | 4788.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 4774.50 | 4775.66 | 4788.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 4774.50 | 4775.43 | 4787.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:15:00 | 4776.00 | 4775.43 | 4787.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 4767.50 | 4773.84 | 4785.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 15:15:00 | 4757.00 | 4776.41 | 4783.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 4800.00 | 4778.02 | 4782.44 | SL hit (close>static) qty=1.00 sl=4798.50 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 4823.50 | 4789.19 | 4786.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 09:15:00 | 4847.00 | 4810.68 | 4799.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 13:15:00 | 4836.50 | 4840.81 | 4828.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 14:00:00 | 4836.50 | 4840.81 | 4828.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 14:15:00 | 4837.00 | 4840.05 | 4829.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 14:45:00 | 4829.00 | 4840.05 | 4829.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 4884.10 | 4917.15 | 4896.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:00:00 | 4884.10 | 4917.15 | 4896.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 4863.70 | 4906.46 | 4893.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 4863.70 | 4906.46 | 4893.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 12 — SELL (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 12:15:00 | 4825.00 | 4877.63 | 4882.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 12:15:00 | 4819.00 | 4847.49 | 4859.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 4873.90 | 4843.46 | 4852.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 4873.90 | 4843.46 | 4852.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 4873.90 | 4843.46 | 4852.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 10:00:00 | 4873.90 | 4843.46 | 4852.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 10:15:00 | 4866.70 | 4848.11 | 4853.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 11:30:00 | 4852.90 | 4851.01 | 4854.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-04 12:15:00 | 4858.70 | 4851.01 | 4854.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 12:15:00 | 4884.10 | 4857.62 | 4857.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 12:15:00 | 4884.10 | 4857.62 | 4857.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-04 13:15:00 | 4889.00 | 4863.90 | 4860.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 09:15:00 | 4851.60 | 4866.84 | 4862.98 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 09:15:00 | 4851.60 | 4866.84 | 4862.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 4851.60 | 4866.84 | 4862.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 4851.60 | 4866.84 | 4862.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 4860.10 | 4865.50 | 4862.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-07 12:15:00 | 4865.20 | 4864.40 | 4862.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-08 10:15:00 | 4813.80 | 4859.01 | 4861.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 4813.80 | 4859.01 | 4861.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 11:15:00 | 4801.40 | 4847.49 | 4856.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 10:15:00 | 4819.10 | 4816.19 | 4832.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 10:45:00 | 4822.10 | 4816.19 | 4832.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 4797.10 | 4761.70 | 4783.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:45:00 | 4804.50 | 4761.70 | 4783.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 4777.10 | 4764.78 | 4783.22 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 13:15:00 | 4839.80 | 4796.59 | 4794.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 09:15:00 | 4843.30 | 4807.68 | 4800.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 4990.10 | 5005.13 | 4973.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 10:00:00 | 4990.10 | 5005.13 | 4973.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 5006.80 | 5002.42 | 4977.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:30:00 | 4991.70 | 5002.42 | 4977.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 4987.40 | 4997.96 | 4984.81 | EMA400 retest candle locked (from upside) |

### Cycle 16 — SELL (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 10:15:00 | 4951.00 | 4985.28 | 4986.18 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 12:15:00 | 4991.90 | 4982.18 | 4982.07 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 13:15:00 | 4978.90 | 4981.53 | 4981.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 15:15:00 | 4967.60 | 4978.48 | 4980.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 4980.40 | 4978.87 | 4980.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 4980.40 | 4978.87 | 4980.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 4980.40 | 4978.87 | 4980.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:30:00 | 5016.10 | 4978.87 | 4980.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 10:15:00 | 4993.70 | 4981.83 | 4981.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-24 12:15:00 | 5003.80 | 4988.30 | 4984.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-25 15:15:00 | 4994.90 | 5007.24 | 4999.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-25 15:15:00 | 4994.90 | 5007.24 | 4999.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 15:15:00 | 4994.90 | 5007.24 | 4999.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:15:00 | 5013.20 | 5007.24 | 4999.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 5054.90 | 5016.77 | 5004.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 10:15:00 | 5071.00 | 5016.77 | 5004.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 09:30:00 | 5055.00 | 5039.80 | 5023.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-29 11:30:00 | 5055.90 | 5045.70 | 5029.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-30 09:30:00 | 5058.10 | 5061.47 | 5045.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 5053.50 | 5059.87 | 5045.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:45:00 | 5040.60 | 5059.87 | 5045.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 5029.60 | 5072.32 | 5060.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 10:00:00 | 5029.60 | 5072.32 | 5060.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 10:15:00 | 5021.20 | 5062.10 | 5057.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 11:30:00 | 5038.00 | 5057.26 | 5055.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 14:15:00 | 5033.90 | 5050.30 | 5052.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 5033.90 | 5050.30 | 5052.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 15:15:00 | 5015.00 | 5043.24 | 5049.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 13:15:00 | 4895.00 | 4883.17 | 4929.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 13:45:00 | 4899.00 | 4883.17 | 4929.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 4857.00 | 4882.99 | 4918.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:30:00 | 4841.50 | 4890.49 | 4907.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 10:30:00 | 4850.00 | 4879.19 | 4901.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 13:45:00 | 4846.00 | 4859.02 | 4885.02 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 15:15:00 | 4840.00 | 4863.12 | 4884.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 4891.00 | 4865.00 | 4881.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:30:00 | 4912.00 | 4865.00 | 4881.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 4855.50 | 4863.10 | 4879.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-07 12:00:00 | 4849.50 | 4860.38 | 4876.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 4844.00 | 4856.83 | 4868.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 14:45:00 | 4849.00 | 4814.91 | 4824.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 09:15:00 | 4928.50 | 4842.44 | 4835.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 09:15:00 | 4928.50 | 4842.44 | 4835.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 10:15:00 | 4959.50 | 4865.85 | 4846.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 14:15:00 | 5341.50 | 5349.13 | 5292.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 15:00:00 | 5341.50 | 5349.13 | 5292.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 5404.50 | 5361.95 | 5308.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 10:30:00 | 5439.00 | 5401.41 | 5377.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 15:15:00 | 5443.00 | 5417.34 | 5394.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 10:15:00 | 5405.00 | 5420.34 | 5421.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 5405.00 | 5420.34 | 5421.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 09:15:00 | 5378.00 | 5403.29 | 5411.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 15:15:00 | 5327.50 | 5311.48 | 5333.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-02 09:15:00 | 5251.00 | 5311.48 | 5333.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 5306.50 | 5310.49 | 5331.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 11:45:00 | 5245.50 | 5290.21 | 5318.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:00:00 | 5240.50 | 5280.27 | 5310.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-04 10:15:00 | 5329.50 | 5299.85 | 5298.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 10:15:00 | 5329.50 | 5299.85 | 5298.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-08 09:15:00 | 5375.50 | 5331.48 | 5317.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 14:15:00 | 5331.50 | 5338.53 | 5327.38 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 14:15:00 | 5331.50 | 5338.53 | 5327.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 14:15:00 | 5331.50 | 5338.53 | 5327.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 15:00:00 | 5331.50 | 5338.53 | 5327.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 5315.00 | 5333.83 | 5326.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 5357.50 | 5333.83 | 5326.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 10:00:00 | 5345.50 | 5336.16 | 5328.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 5365.00 | 5341.05 | 5335.04 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 5314.50 | 5337.04 | 5334.89 | SL hit (close<static) qty=1.00 sl=5315.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 12:15:00 | 5313.00 | 5332.23 | 5332.90 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 09:15:00 | 5369.50 | 5334.58 | 5333.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 10:15:00 | 5377.00 | 5343.06 | 5337.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 09:15:00 | 5418.00 | 5427.52 | 5401.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-15 09:15:00 | 5418.00 | 5427.52 | 5401.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 5418.00 | 5427.52 | 5401.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-15 12:00:00 | 5472.50 | 5440.03 | 5412.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 11:15:00 | 5477.00 | 5490.86 | 5472.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 11:45:00 | 5489.50 | 5491.79 | 5474.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 14:00:00 | 5471.50 | 5485.36 | 5474.27 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 5477.50 | 5483.79 | 5474.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:45:00 | 5474.00 | 5483.79 | 5474.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 15:15:00 | 5459.50 | 5478.93 | 5473.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:15:00 | 5472.00 | 5478.93 | 5473.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 5461.50 | 5475.45 | 5472.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:15:00 | 5526.00 | 5477.07 | 5474.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 10:45:00 | 5504.00 | 5515.28 | 5502.94 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 11:30:00 | 5510.00 | 5512.62 | 5502.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-22 12:45:00 | 5504.50 | 5509.40 | 5502.28 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 5495.50 | 5506.62 | 5501.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 13:45:00 | 5494.00 | 5506.62 | 5501.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 14:15:00 | 5499.50 | 5505.19 | 5501.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:30:00 | 5493.00 | 5505.19 | 5501.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 15:15:00 | 5486.00 | 5501.36 | 5500.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 09:15:00 | 5525.50 | 5501.36 | 5500.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 13:15:00 | 5520.00 | 5537.59 | 5539.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 13:15:00 | 5520.00 | 5537.59 | 5539.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 5500.50 | 5530.17 | 5536.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 15:15:00 | 5443.50 | 5429.52 | 5455.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 09:15:00 | 5434.50 | 5429.52 | 5455.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 5432.50 | 5430.12 | 5453.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 11:30:00 | 5420.00 | 5430.00 | 5449.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 12:15:00 | 5409.00 | 5430.00 | 5449.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 10:00:00 | 5414.50 | 5419.14 | 5435.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 5461.00 | 5432.99 | 5437.00 | SL hit (close>static) qty=1.00 sl=5460.50 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 5469.00 | 5442.91 | 5440.98 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 09:15:00 | 5401.50 | 5435.09 | 5439.53 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-10-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 13:15:00 | 5496.50 | 5449.42 | 5444.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 10:15:00 | 5501.00 | 5473.06 | 5458.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 11:15:00 | 5472.00 | 5472.85 | 5459.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 12:00:00 | 5472.00 | 5472.85 | 5459.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 11:15:00 | 5470.00 | 5483.45 | 5472.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 11:45:00 | 5473.50 | 5483.45 | 5472.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 5494.00 | 5485.56 | 5474.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 13:30:00 | 5497.00 | 5486.55 | 5476.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 14:30:00 | 5497.50 | 5486.44 | 5477.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 15:15:00 | 5499.00 | 5486.44 | 5477.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 10:00:00 | 5499.00 | 5490.96 | 5480.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 5488.50 | 5490.47 | 5481.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 11:00:00 | 5488.50 | 5490.47 | 5481.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 5487.00 | 5489.77 | 5482.10 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-10 09:15:00 | 5455.00 | 5478.77 | 5479.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 09:15:00 | 5455.00 | 5478.77 | 5479.63 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 14:15:00 | 5488.00 | 5474.94 | 5473.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 15:15:00 | 5509.00 | 5481.75 | 5476.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 09:15:00 | 5463.00 | 5478.00 | 5475.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 5463.00 | 5478.00 | 5475.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 5463.00 | 5478.00 | 5475.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:30:00 | 5460.00 | 5478.00 | 5475.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 5483.00 | 5479.00 | 5476.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 12:45:00 | 5492.00 | 5481.58 | 5477.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 14:30:00 | 5492.50 | 5485.15 | 5480.23 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 14:15:00 | 5521.00 | 5554.66 | 5555.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 5521.00 | 5554.66 | 5555.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 09:15:00 | 5509.50 | 5540.89 | 5548.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 14:15:00 | 5543.00 | 5528.57 | 5538.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 14:15:00 | 5543.00 | 5528.57 | 5538.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 5543.00 | 5528.57 | 5538.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 15:00:00 | 5543.00 | 5528.57 | 5538.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 15:15:00 | 5545.00 | 5531.86 | 5538.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 09:15:00 | 5554.00 | 5531.86 | 5538.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 5452.00 | 5431.84 | 5460.98 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 14:15:00 | 5527.00 | 5480.09 | 5475.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 15:15:00 | 5544.00 | 5492.87 | 5482.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 5476.50 | 5489.60 | 5481.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 5476.50 | 5489.60 | 5481.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 5476.50 | 5489.60 | 5481.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:45:00 | 5459.50 | 5489.60 | 5481.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 5506.00 | 5492.88 | 5483.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-30 15:00:00 | 5530.50 | 5507.95 | 5494.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 5530.00 | 5502.43 | 5498.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 11:45:00 | 5529.00 | 5515.25 | 5505.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 10:15:00 | 5655.00 | 5721.64 | 5728.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 5655.00 | 5721.64 | 5728.13 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 13:15:00 | 5729.50 | 5707.38 | 5706.01 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 09:15:00 | 5669.00 | 5701.00 | 5703.56 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 13:15:00 | 5737.00 | 5707.82 | 5705.27 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 15:15:00 | 5686.00 | 5708.95 | 5709.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 09:15:00 | 5664.00 | 5699.96 | 5705.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 13:15:00 | 5691.50 | 5685.61 | 5695.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 14:00:00 | 5691.50 | 5685.61 | 5695.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 5701.50 | 5688.79 | 5695.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 5701.50 | 5688.79 | 5695.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 5709.00 | 5692.83 | 5697.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 09:15:00 | 5667.00 | 5692.83 | 5697.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 5665.50 | 5687.36 | 5694.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 12:00:00 | 5636.00 | 5673.35 | 5686.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 15:00:00 | 5629.00 | 5657.24 | 5675.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-26 09:15:00 | 5723.50 | 5677.62 | 5671.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — BUY (started 2025-11-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 09:15:00 | 5723.50 | 5677.62 | 5671.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 10:15:00 | 5732.50 | 5688.60 | 5677.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 5729.00 | 5733.01 | 5708.78 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 11:00:00 | 5729.00 | 5733.01 | 5708.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 5691.50 | 5724.71 | 5707.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 12:00:00 | 5691.50 | 5724.71 | 5707.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 12:15:00 | 5706.50 | 5721.07 | 5707.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 10:45:00 | 5716.50 | 5708.00 | 5703.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 14:15:00 | 5683.00 | 5699.41 | 5700.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 14:15:00 | 5683.00 | 5699.41 | 5700.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 15:15:00 | 5677.50 | 5695.03 | 5698.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 13:15:00 | 5631.00 | 5625.71 | 5647.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-02 13:45:00 | 5632.50 | 5625.71 | 5647.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 5625.00 | 5628.10 | 5644.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 5646.00 | 5628.10 | 5644.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 5629.00 | 5628.28 | 5643.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:45:00 | 5615.00 | 5625.62 | 5640.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 5680.00 | 5647.87 | 5646.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 5680.00 | 5647.87 | 5646.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 10:15:00 | 5709.50 | 5669.18 | 5660.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 13:15:00 | 5681.00 | 5684.01 | 5670.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-05 13:30:00 | 5680.00 | 5684.01 | 5670.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 14:15:00 | 5670.00 | 5681.21 | 5670.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 14:45:00 | 5660.00 | 5681.21 | 5670.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 5685.00 | 5681.96 | 5671.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 5681.50 | 5681.96 | 5671.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 5675.50 | 5680.67 | 5671.95 | EMA400 retest candle locked (from upside) |

### Cycle 42 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 5611.00 | 5660.00 | 5664.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 15:15:00 | 5595.00 | 5637.40 | 5653.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 10:15:00 | 5634.00 | 5628.18 | 5645.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 11:00:00 | 5634.00 | 5628.18 | 5645.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 11:15:00 | 5659.00 | 5634.34 | 5646.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 12:00:00 | 5659.00 | 5634.34 | 5646.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 5656.50 | 5638.77 | 5647.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:15:00 | 5669.00 | 5638.77 | 5647.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 5670.50 | 5645.12 | 5649.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 5666.50 | 5645.12 | 5649.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 5672.00 | 5654.24 | 5653.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 5717.50 | 5666.89 | 5659.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 10:15:00 | 5663.50 | 5666.21 | 5659.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 10:15:00 | 5663.50 | 5666.21 | 5659.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 5663.50 | 5666.21 | 5659.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:00:00 | 5663.50 | 5666.21 | 5659.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 5643.00 | 5661.57 | 5658.15 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 12:15:00 | 5629.50 | 5655.16 | 5655.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 13:15:00 | 5611.00 | 5646.32 | 5651.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 5648.00 | 5622.74 | 5635.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 11:15:00 | 5648.00 | 5622.74 | 5635.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 5648.00 | 5622.74 | 5635.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 5648.00 | 5622.74 | 5635.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 5638.00 | 5625.79 | 5635.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:45:00 | 5660.50 | 5625.79 | 5635.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 13:15:00 | 5621.50 | 5624.94 | 5634.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:30:00 | 5597.00 | 5625.45 | 5633.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 10:45:00 | 5602.00 | 5620.32 | 5629.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 12:30:00 | 5608.50 | 5616.72 | 5625.68 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 13:15:00 | 5650.00 | 5623.38 | 5627.89 | SL hit (close>static) qty=1.00 sl=5640.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-16 11:15:00 | 5638.00 | 5622.40 | 5621.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 12:15:00 | 5654.00 | 5628.72 | 5624.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 10:15:00 | 5643.00 | 5649.14 | 5638.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 10:15:00 | 5643.00 | 5649.14 | 5638.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 5643.00 | 5649.14 | 5638.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:00:00 | 5643.00 | 5649.14 | 5638.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 5639.00 | 5647.11 | 5638.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:30:00 | 5640.50 | 5647.11 | 5638.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 5608.00 | 5639.29 | 5635.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 13:00:00 | 5608.00 | 5639.29 | 5635.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 13:15:00 | 5600.50 | 5631.53 | 5632.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 12:15:00 | 5557.50 | 5598.21 | 5614.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 14:15:00 | 5551.00 | 5540.46 | 5565.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 14:15:00 | 5551.00 | 5540.46 | 5565.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 5551.00 | 5540.46 | 5565.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 5555.50 | 5540.46 | 5565.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 5582.50 | 5548.87 | 5566.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 5613.50 | 5548.87 | 5566.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 5593.50 | 5557.80 | 5569.16 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 5640.00 | 5580.27 | 5577.77 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 09:15:00 | 5554.00 | 5589.25 | 5592.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 11:15:00 | 5538.00 | 5573.52 | 5584.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 13:15:00 | 5572.50 | 5571.87 | 5581.71 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-24 13:45:00 | 5567.00 | 5571.87 | 5581.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 5514.50 | 5556.85 | 5572.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 09:30:00 | 5536.50 | 5556.85 | 5572.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 5518.50 | 5534.77 | 5551.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 14:30:00 | 5508.50 | 5525.55 | 5539.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 10:45:00 | 5506.00 | 5474.02 | 5492.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 15:00:00 | 5510.00 | 5492.28 | 5496.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-02 11:15:00 | 5557.50 | 5488.80 | 5483.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 5557.50 | 5488.80 | 5483.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 5597.50 | 5525.20 | 5502.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 5546.50 | 5567.98 | 5538.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 13:15:00 | 5546.50 | 5567.98 | 5538.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 5546.50 | 5567.98 | 5538.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 5546.50 | 5567.98 | 5538.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 5563.00 | 5566.99 | 5540.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:30:00 | 5537.00 | 5566.99 | 5540.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 15:15:00 | 5569.50 | 5567.49 | 5543.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:15:00 | 5590.50 | 5567.49 | 5543.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 10:30:00 | 5580.00 | 5572.73 | 5550.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 5770.50 | 5823.62 | 5824.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 5770.50 | 5823.62 | 5824.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 10:15:00 | 5727.00 | 5804.29 | 5815.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-20 10:15:00 | 5753.50 | 5750.40 | 5775.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 10:15:00 | 5753.50 | 5750.40 | 5775.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 5753.50 | 5750.40 | 5775.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:45:00 | 5780.00 | 5750.40 | 5775.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 09:15:00 | 5722.00 | 5722.82 | 5749.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 10:30:00 | 5647.50 | 5706.46 | 5739.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 15:15:00 | 5744.50 | 5720.04 | 5716.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 5744.50 | 5720.04 | 5716.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 5794.50 | 5734.93 | 5723.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 5744.00 | 5756.45 | 5741.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 14:15:00 | 5744.00 | 5756.45 | 5741.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 5744.00 | 5756.45 | 5741.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 5744.00 | 5756.45 | 5741.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 5749.00 | 5754.96 | 5741.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 5767.50 | 5754.96 | 5741.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 5774.00 | 5758.77 | 5744.86 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-28 11:15:00 | 5700.00 | 5736.94 | 5741.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 12:15:00 | 5669.50 | 5723.45 | 5734.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 14:15:00 | 5724.00 | 5716.29 | 5729.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 15:00:00 | 5724.00 | 5716.29 | 5729.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 15:15:00 | 5720.00 | 5717.03 | 5728.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:15:00 | 5686.50 | 5717.03 | 5728.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 5633.00 | 5700.23 | 5719.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 10:30:00 | 5625.00 | 5668.25 | 5684.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:00:00 | 5602.00 | 5655.00 | 5676.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 11:15:00 | 5703.00 | 5601.48 | 5598.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 5703.00 | 5601.48 | 5598.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 12:15:00 | 5735.00 | 5628.18 | 5611.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 10:15:00 | 5656.00 | 5664.59 | 5639.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 10:15:00 | 5656.00 | 5664.59 | 5639.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 10:15:00 | 5656.00 | 5664.59 | 5639.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 10:30:00 | 5644.50 | 5664.59 | 5639.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 11:15:00 | 5674.00 | 5666.48 | 5642.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 11:30:00 | 5677.00 | 5666.48 | 5642.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 15:15:00 | 5660.00 | 5663.44 | 5648.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:15:00 | 5613.00 | 5663.44 | 5648.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 5607.00 | 5652.15 | 5644.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:45:00 | 5594.50 | 5652.15 | 5644.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 5650.00 | 5651.72 | 5645.42 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 5580.50 | 5640.70 | 5643.60 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 5674.00 | 5645.03 | 5644.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 5693.00 | 5654.62 | 5648.56 | Break + close above crossover candle high |

### Cycle 56 — SELL (started 2026-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 15:15:00 | 5587.50 | 5641.20 | 5643.01 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 5746.50 | 5662.26 | 5652.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 11:15:00 | 5771.00 | 5700.37 | 5672.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-12 10:15:00 | 5863.50 | 5865.86 | 5820.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-12 10:45:00 | 5855.50 | 5865.86 | 5820.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 14:15:00 | 5876.00 | 5859.20 | 5831.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-12 14:30:00 | 5846.00 | 5859.20 | 5831.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 15:15:00 | 5857.00 | 5858.76 | 5833.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:15:00 | 5798.00 | 5858.76 | 5833.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 5773.00 | 5841.61 | 5827.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 5799.00 | 5841.61 | 5827.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 5770.50 | 5827.39 | 5822.69 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 11:15:00 | 5782.00 | 5818.31 | 5818.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 12:15:00 | 5735.00 | 5801.65 | 5811.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 13:15:00 | 5461.50 | 5446.36 | 5493.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-18 14:00:00 | 5461.50 | 5446.36 | 5493.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 5448.00 | 5451.40 | 5484.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 12:00:00 | 5408.50 | 5439.00 | 5472.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:45:00 | 5418.50 | 5390.10 | 5411.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-23 13:15:00 | 5490.00 | 5431.77 | 5425.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-02-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 13:15:00 | 5490.00 | 5431.77 | 5425.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 09:15:00 | 5500.50 | 5477.65 | 5459.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 11:15:00 | 5672.50 | 5690.08 | 5633.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 11:30:00 | 5673.00 | 5690.08 | 5633.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 5637.00 | 5673.04 | 5639.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 5637.00 | 5673.04 | 5639.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 5614.00 | 5661.23 | 5636.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 5572.00 | 5661.23 | 5636.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 5560.00 | 5640.99 | 5629.79 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 5540.50 | 5620.89 | 5621.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 5499.50 | 5596.61 | 5610.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-02 15:15:00 | 5582.50 | 5570.39 | 5590.65 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-04 09:15:00 | 5489.00 | 5570.39 | 5590.65 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 5506.50 | 5494.45 | 5529.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:30:00 | 5522.50 | 5494.45 | 5529.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 5500.50 | 5491.21 | 5516.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:00:00 | 5500.50 | 5491.21 | 5516.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 5536.50 | 5500.27 | 5518.44 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 5536.50 | 5500.27 | 5518.44 | SL hit (close>ema400) qty=1.00 sl=5518.44 alert=retest1 |

### Cycle 61 — BUY (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 13:15:00 | 5555.50 | 5525.67 | 5524.97 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 5461.00 | 5513.51 | 5519.72 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 5620.00 | 5520.94 | 5512.71 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 5467.50 | 5547.94 | 5550.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 13:15:00 | 5407.00 | 5486.23 | 5518.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 10:15:00 | 5322.00 | 5315.30 | 5361.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 5385.00 | 5340.27 | 5354.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 5385.00 | 5340.27 | 5354.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 5253.50 | 5358.86 | 5359.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 14:15:00 | 5294.50 | 5230.04 | 5228.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 5294.50 | 5230.04 | 5228.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 5377.00 | 5270.79 | 5248.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 5392.50 | 5394.65 | 5336.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 09:30:00 | 5382.00 | 5394.65 | 5336.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 5353.00 | 5375.01 | 5345.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 5353.00 | 5375.01 | 5345.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 5333.50 | 5366.70 | 5343.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 5333.50 | 5366.70 | 5343.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 5271.50 | 5347.66 | 5337.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 5268.00 | 5347.66 | 5337.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 5279.00 | 5333.93 | 5332.07 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 5301.00 | 5327.34 | 5329.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 13:15:00 | 5228.00 | 5285.50 | 5304.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 5250.00 | 5209.50 | 5247.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 13:15:00 | 5250.00 | 5209.50 | 5247.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 5250.00 | 5209.50 | 5247.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:00:00 | 5250.00 | 5209.50 | 5247.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 5264.00 | 5220.40 | 5248.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 15:00:00 | 5264.00 | 5220.40 | 5248.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 15:15:00 | 5271.00 | 5230.52 | 5250.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 09:15:00 | 5221.50 | 5230.52 | 5250.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 5211.00 | 5226.61 | 5247.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 11:00:00 | 5180.00 | 5217.29 | 5240.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 11:30:00 | 5174.50 | 5205.03 | 5233.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 13:00:00 | 5180.00 | 5200.03 | 5228.42 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:30:00 | 5157.50 | 5209.56 | 5224.68 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 10:15:00 | 5200.00 | 5207.65 | 5222.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-07 10:30:00 | 5217.50 | 5207.65 | 5222.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 09:15:00 | 5237.50 | 5194.05 | 5206.12 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-08 11:15:00 | 5254.50 | 5215.25 | 5214.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 11:15:00 | 5254.50 | 5215.25 | 5214.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 11:15:00 | 5336.50 | 5257.96 | 5238.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 5396.00 | 5403.75 | 5355.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 5396.00 | 5403.75 | 5355.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 5396.00 | 5403.75 | 5355.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 5408.50 | 5403.75 | 5355.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:30:00 | 5432.50 | 5395.23 | 5371.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 5539.50 | 5632.68 | 5640.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 5539.50 | 5632.68 | 5640.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 11:15:00 | 5443.00 | 5576.96 | 5613.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-23 15:15:00 | 5539.00 | 5538.60 | 5580.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-24 09:15:00 | 5559.50 | 5538.60 | 5580.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 5542.00 | 5539.28 | 5576.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 10:30:00 | 5503.50 | 5530.53 | 5569.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 14:15:00 | 5228.32 | 5441.76 | 5513.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 5419.50 | 5405.03 | 5482.09 | SL hit (close>ema200) qty=0.50 sl=5405.03 alert=retest2 |

### Cycle 69 — BUY (started 2026-05-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 14:15:00 | 5404.50 | 5378.54 | 5375.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 5509.50 | 5408.89 | 5389.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 15:15:00 | 5552.50 | 5569.90 | 5525.80 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 09:15:00 | 5628.00 | 5569.90 | 5525.80 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 10:00:00 | 5609.00 | 5577.72 | 5533.36 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 12:30:00 | 5610.50 | 5594.69 | 5553.65 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-05-08 14:00:00 | 5616.50 | 5599.05 | 5559.36 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 5560.00 | 5589.39 | 5561.75 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-05-08 15:15:00 | 5560.00 | 5589.39 | 5561.75 | SL hit (close<ema400) qty=1.00 sl=5561.75 alert=retest1 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 11:15:00 | 5006.00 | 2025-05-12 11:15:00 | 5019.00 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest2 | 2025-05-19 10:30:00 | 5391.50 | 2025-05-22 13:15:00 | 5294.00 | STOP_HIT | 1.00 | -1.81% |
| BUY | retest2 | 2025-05-19 13:00:00 | 5388.50 | 2025-05-22 13:15:00 | 5294.00 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2025-05-21 09:30:00 | 5377.00 | 2025-05-22 13:15:00 | 5294.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2025-05-21 10:00:00 | 5379.00 | 2025-05-22 13:15:00 | 5294.00 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-06-03 11:30:00 | 4973.50 | 2025-06-10 10:15:00 | 4942.00 | STOP_HIT | 1.00 | 0.63% |
| SELL | retest2 | 2025-06-03 15:15:00 | 4980.00 | 2025-06-10 10:15:00 | 4942.00 | STOP_HIT | 1.00 | 0.76% |
| SELL | retest2 | 2025-06-19 10:30:00 | 4763.00 | 2025-06-24 09:15:00 | 4800.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-06-20 12:15:00 | 4761.00 | 2025-06-24 11:15:00 | 4823.50 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-06-20 13:15:00 | 4763.00 | 2025-06-24 11:15:00 | 4823.50 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2025-06-20 14:15:00 | 4765.00 | 2025-06-24 11:15:00 | 4823.50 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-06-23 15:15:00 | 4757.00 | 2025-06-24 11:15:00 | 4823.50 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-07-04 11:30:00 | 4852.90 | 2025-07-04 12:15:00 | 4884.10 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-07-04 12:15:00 | 4858.70 | 2025-07-04 12:15:00 | 4884.10 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-07-07 12:15:00 | 4865.20 | 2025-07-08 10:15:00 | 4813.80 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-07-28 10:15:00 | 5071.00 | 2025-07-31 14:15:00 | 5033.90 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-07-29 09:30:00 | 5055.00 | 2025-07-31 14:15:00 | 5033.90 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-07-29 11:30:00 | 5055.90 | 2025-07-31 14:15:00 | 5033.90 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-07-30 09:30:00 | 5058.10 | 2025-07-31 14:15:00 | 5033.90 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-07-31 11:30:00 | 5038.00 | 2025-07-31 14:15:00 | 5033.90 | STOP_HIT | 1.00 | -0.08% |
| SELL | retest2 | 2025-08-06 09:30:00 | 4841.50 | 2025-08-12 09:15:00 | 4928.50 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2025-08-06 10:30:00 | 4850.00 | 2025-08-12 09:15:00 | 4928.50 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-08-06 13:45:00 | 4846.00 | 2025-08-12 09:15:00 | 4928.50 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-08-06 15:15:00 | 4840.00 | 2025-08-12 09:15:00 | 4928.50 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-08-07 12:00:00 | 4849.50 | 2025-08-12 09:15:00 | 4928.50 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2025-08-08 09:15:00 | 4844.00 | 2025-08-12 09:15:00 | 4928.50 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-08-11 14:45:00 | 4849.00 | 2025-08-12 09:15:00 | 4928.50 | STOP_HIT | 1.00 | -1.64% |
| BUY | retest2 | 2025-08-21 10:30:00 | 5439.00 | 2025-08-26 10:15:00 | 5405.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-08-21 15:15:00 | 5443.00 | 2025-08-26 10:15:00 | 5405.00 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-09-02 11:45:00 | 5245.50 | 2025-09-04 10:15:00 | 5329.50 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-09-02 13:00:00 | 5240.50 | 2025-09-04 10:15:00 | 5329.50 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-09-09 09:15:00 | 5357.50 | 2025-09-10 11:15:00 | 5314.50 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-09-09 10:00:00 | 5345.50 | 2025-09-10 11:15:00 | 5314.50 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-09-10 09:15:00 | 5365.00 | 2025-09-10 11:15:00 | 5314.50 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-09-15 12:00:00 | 5472.50 | 2025-09-25 13:15:00 | 5520.00 | STOP_HIT | 1.00 | 0.87% |
| BUY | retest2 | 2025-09-17 11:15:00 | 5477.00 | 2025-09-25 13:15:00 | 5520.00 | STOP_HIT | 1.00 | 0.79% |
| BUY | retest2 | 2025-09-17 11:45:00 | 5489.50 | 2025-09-25 13:15:00 | 5520.00 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2025-09-17 14:00:00 | 5471.50 | 2025-09-25 13:15:00 | 5520.00 | STOP_HIT | 1.00 | 0.89% |
| BUY | retest2 | 2025-09-19 09:15:00 | 5526.00 | 2025-09-25 13:15:00 | 5520.00 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2025-09-22 10:45:00 | 5504.00 | 2025-09-25 13:15:00 | 5520.00 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2025-09-22 11:30:00 | 5510.00 | 2025-09-25 13:15:00 | 5520.00 | STOP_HIT | 1.00 | 0.18% |
| BUY | retest2 | 2025-09-22 12:45:00 | 5504.50 | 2025-09-25 13:15:00 | 5520.00 | STOP_HIT | 1.00 | 0.28% |
| BUY | retest2 | 2025-09-23 09:15:00 | 5525.50 | 2025-09-25 13:15:00 | 5520.00 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-09-30 11:30:00 | 5420.00 | 2025-10-01 13:15:00 | 5461.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-09-30 12:15:00 | 5409.00 | 2025-10-01 13:15:00 | 5461.00 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-10-01 10:00:00 | 5414.50 | 2025-10-01 13:15:00 | 5461.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-10-08 13:30:00 | 5497.00 | 2025-10-10 09:15:00 | 5455.00 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-10-08 14:30:00 | 5497.50 | 2025-10-10 09:15:00 | 5455.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-10-08 15:15:00 | 5499.00 | 2025-10-10 09:15:00 | 5455.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-10-09 10:00:00 | 5499.00 | 2025-10-10 09:15:00 | 5455.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-10-14 12:45:00 | 5492.00 | 2025-10-23 14:15:00 | 5521.00 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2025-10-14 14:30:00 | 5492.50 | 2025-10-23 14:15:00 | 5521.00 | STOP_HIT | 1.00 | 0.52% |
| BUY | retest2 | 2025-10-30 15:00:00 | 5530.50 | 2025-11-14 10:15:00 | 5655.00 | STOP_HIT | 1.00 | 2.25% |
| BUY | retest2 | 2025-11-03 09:15:00 | 5530.00 | 2025-11-14 10:15:00 | 5655.00 | STOP_HIT | 1.00 | 2.26% |
| BUY | retest2 | 2025-11-03 11:45:00 | 5529.00 | 2025-11-14 10:15:00 | 5655.00 | STOP_HIT | 1.00 | 2.28% |
| SELL | retest2 | 2025-11-24 12:00:00 | 5636.00 | 2025-11-26 09:15:00 | 5723.50 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-11-24 15:00:00 | 5629.00 | 2025-11-26 09:15:00 | 5723.50 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-11-28 10:45:00 | 5716.50 | 2025-11-28 14:15:00 | 5683.00 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-12-03 10:45:00 | 5615.00 | 2025-12-04 09:15:00 | 5680.00 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2025-12-11 14:30:00 | 5597.00 | 2025-12-12 13:15:00 | 5650.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-12-12 10:45:00 | 5602.00 | 2025-12-12 13:15:00 | 5650.00 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-12-12 12:30:00 | 5608.50 | 2025-12-12 13:15:00 | 5650.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-12-12 15:15:00 | 5614.50 | 2025-12-16 11:15:00 | 5638.00 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-12-15 09:15:00 | 5587.00 | 2025-12-16 11:15:00 | 5638.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-12-15 11:15:00 | 5607.50 | 2025-12-16 11:15:00 | 5638.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-12-15 14:30:00 | 5607.00 | 2025-12-16 11:15:00 | 5638.00 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-12-15 15:15:00 | 5611.00 | 2025-12-16 11:15:00 | 5638.00 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-12-29 14:30:00 | 5508.50 | 2026-01-02 11:15:00 | 5557.50 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-12-31 10:45:00 | 5506.00 | 2026-01-02 11:15:00 | 5557.50 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-12-31 15:00:00 | 5510.00 | 2026-01-02 11:15:00 | 5557.50 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-01-06 09:15:00 | 5590.50 | 2026-01-19 09:15:00 | 5770.50 | STOP_HIT | 1.00 | 3.22% |
| BUY | retest2 | 2026-01-06 10:30:00 | 5580.00 | 2026-01-19 09:15:00 | 5770.50 | STOP_HIT | 1.00 | 3.41% |
| SELL | retest2 | 2026-01-21 10:30:00 | 5647.50 | 2026-01-22 15:15:00 | 5744.50 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2026-02-01 10:30:00 | 5625.00 | 2026-02-03 11:15:00 | 5703.00 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-02-01 12:00:00 | 5602.00 | 2026-02-03 11:15:00 | 5703.00 | STOP_HIT | 1.00 | -1.80% |
| SELL | retest2 | 2026-02-19 12:00:00 | 5408.50 | 2026-02-23 13:15:00 | 5490.00 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2026-02-23 09:45:00 | 5418.50 | 2026-02-23 13:15:00 | 5490.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest1 | 2026-03-04 09:15:00 | 5489.00 | 2026-03-05 14:15:00 | 5536.50 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2026-03-19 09:15:00 | 5253.50 | 2026-03-24 14:15:00 | 5294.50 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2026-04-06 11:00:00 | 5180.00 | 2026-04-08 11:15:00 | 5254.50 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2026-04-06 11:30:00 | 5174.50 | 2026-04-08 11:15:00 | 5254.50 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-04-06 13:00:00 | 5180.00 | 2026-04-08 11:15:00 | 5254.50 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2026-04-07 09:30:00 | 5157.50 | 2026-04-08 11:15:00 | 5254.50 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2026-04-13 10:15:00 | 5408.50 | 2026-04-23 09:15:00 | 5539.50 | STOP_HIT | 1.00 | 2.42% |
| BUY | retest2 | 2026-04-15 09:30:00 | 5432.50 | 2026-04-23 09:15:00 | 5539.50 | STOP_HIT | 1.00 | 1.97% |
| SELL | retest2 | 2026-04-24 10:30:00 | 5503.50 | 2026-04-24 14:15:00 | 5228.32 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-24 10:30:00 | 5503.50 | 2026-04-27 09:15:00 | 5419.50 | STOP_HIT | 0.50 | 1.53% |
| BUY | retest1 | 2026-05-08 09:15:00 | 5628.00 | 2026-05-08 15:15:00 | 5560.00 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest1 | 2026-05-08 10:00:00 | 5609.00 | 2026-05-08 15:15:00 | 5560.00 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest1 | 2026-05-08 12:30:00 | 5610.50 | 2026-05-08 15:15:00 | 5560.00 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest1 | 2026-05-08 14:00:00 | 5616.50 | 2026-05-08 15:15:00 | 5560.00 | STOP_HIT | 1.00 | -1.01% |
