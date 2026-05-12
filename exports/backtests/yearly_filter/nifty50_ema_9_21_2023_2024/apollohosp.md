# APOLLOHOSP (APOLLOHOSP)

## Backtest Summary

- **Window:** 2023-03-14 09:15:00 → 2026-05-08 15:15:00 (5436 bars)
- **Last close:** 8100.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 230 |
| ALERT1 | 156 |
| ALERT2 | 156 |
| ALERT2_SKIP | 79 |
| ALERT3 | 421 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 145 |
| PARTIAL | 7 |
| TARGET_HIT | 1 |
| STOP_HIT | 150 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 158 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 50 / 108
- **Target hits / Stop hits / Partials:** 1 / 150 / 7
- **Avg / median % per leg:** 0.03% / -0.62%
- **Sum % (uncompounded):** 4.92%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 88 | 18 | 20.5% | 1 | 87 | 0 | -0.40% | -35.0% |
| BUY @ 2nd Alert (retest1) | 6 | 1 | 16.7% | 0 | 6 | 0 | -1.45% | -8.7% |
| BUY @ 3rd Alert (retest2) | 82 | 17 | 20.7% | 1 | 81 | 0 | -0.32% | -26.3% |
| SELL (all) | 70 | 32 | 45.7% | 0 | 63 | 7 | 0.57% | 39.9% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 70 | 32 | 45.7% | 0 | 63 | 7 | 0.57% | 39.9% |
| retest1 (combined) | 6 | 1 | 16.7% | 0 | 6 | 0 | -1.45% | -8.7% |
| retest2 (combined) | 152 | 49 | 32.2% | 1 | 144 | 7 | 0.09% | 13.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2023-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-16 11:15:00 | 4588.00 | 4618.24 | 4620.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-05-16 14:15:00 | 4555.80 | 4596.08 | 4608.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-05-22 09:15:00 | 4502.60 | 4459.95 | 4479.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-22 09:15:00 | 4502.60 | 4459.95 | 4479.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 09:15:00 | 4502.60 | 4459.95 | 4479.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-05-22 10:00:00 | 4502.60 | 4459.95 | 4479.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-22 10:15:00 | 4508.00 | 4469.56 | 4481.81 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2023-05-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-05-22 12:15:00 | 4580.10 | 4505.98 | 4497.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-05-22 14:15:00 | 4614.80 | 4539.91 | 4514.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-05-23 14:15:00 | 4552.30 | 4570.85 | 4547.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-05-23 14:15:00 | 4552.30 | 4570.85 | 4547.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 14:15:00 | 4552.30 | 4570.85 | 4547.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-05-23 14:45:00 | 4557.80 | 4570.85 | 4547.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-05-23 15:15:00 | 4541.00 | 4564.88 | 4546.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-05-24 09:15:00 | 4562.55 | 4564.88 | 4546.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-05-24 14:15:00 | 4534.95 | 4564.83 | 4556.84 | SL hit (close<static) qty=1.00 sl=4538.00 alert=retest2 |

### Cycle 3 — SELL (started 2023-05-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-05-31 13:15:00 | 4561.65 | 4629.09 | 4631.58 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2023-06-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-01 09:15:00 | 4815.00 | 4664.38 | 4646.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-02 09:15:00 | 4841.60 | 4788.58 | 4730.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-05 14:15:00 | 4945.00 | 4948.79 | 4883.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-05 15:00:00 | 4945.00 | 4948.79 | 4883.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-06 11:15:00 | 4915.00 | 4936.02 | 4897.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-06 11:45:00 | 4901.70 | 4936.02 | 4897.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 09:15:00 | 4966.40 | 4996.48 | 4964.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 09:30:00 | 4985.60 | 4996.48 | 4964.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 10:15:00 | 4942.10 | 4985.60 | 4962.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 11:00:00 | 4942.10 | 4985.60 | 4962.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 11:15:00 | 4944.15 | 4977.31 | 4960.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 11:45:00 | 4940.00 | 4977.31 | 4960.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-08 12:15:00 | 4939.90 | 4969.83 | 4958.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-08 12:30:00 | 4941.45 | 4969.83 | 4958.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — SELL (started 2023-06-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-08 15:15:00 | 4930.10 | 4949.36 | 4950.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-09 15:15:00 | 4920.00 | 4932.76 | 4940.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-12 09:15:00 | 4965.85 | 4939.38 | 4943.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-06-12 09:15:00 | 4965.85 | 4939.38 | 4943.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 09:15:00 | 4965.85 | 4939.38 | 4943.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 09:30:00 | 4982.35 | 4939.38 | 4943.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 10:15:00 | 4970.00 | 4945.50 | 4945.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 10:45:00 | 4978.00 | 4945.50 | 4945.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 12:15:00 | 4948.55 | 4944.41 | 4944.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-12 13:00:00 | 4948.55 | 4944.41 | 4944.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-12 13:15:00 | 4938.20 | 4943.17 | 4944.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-06-12 14:15:00 | 4924.00 | 4943.17 | 4944.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-06-13 09:15:00 | 4969.00 | 4943.50 | 4943.64 | SL hit (close>static) qty=1.00 sl=4954.95 alert=retest2 |

### Cycle 6 — BUY (started 2023-06-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-13 10:15:00 | 4982.00 | 4951.20 | 4947.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-14 09:15:00 | 4993.00 | 4960.58 | 4953.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-19 12:15:00 | 5183.60 | 5190.65 | 5149.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-19 12:45:00 | 5171.10 | 5190.65 | 5149.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 14:15:00 | 5144.05 | 5176.67 | 5150.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-19 15:00:00 | 5144.05 | 5176.67 | 5150.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-19 15:15:00 | 5150.00 | 5171.34 | 5150.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 09:15:00 | 5132.15 | 5171.34 | 5150.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 10:15:00 | 5173.00 | 5173.05 | 5154.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 10:30:00 | 5165.80 | 5173.05 | 5154.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 13:15:00 | 5141.30 | 5164.99 | 5155.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-06-20 13:45:00 | 5134.40 | 5164.99 | 5155.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-20 14:15:00 | 5161.90 | 5164.37 | 5156.09 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2023-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-21 10:15:00 | 5093.95 | 5140.58 | 5146.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-06-22 12:15:00 | 5078.45 | 5114.19 | 5128.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-06-26 15:15:00 | 5040.00 | 5037.42 | 5055.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-06-27 09:15:00 | 5088.65 | 5037.42 | 5055.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 09:15:00 | 5118.80 | 5053.70 | 5061.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-06-27 10:00:00 | 5118.80 | 5053.70 | 5061.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-27 10:15:00 | 5109.35 | 5064.83 | 5065.80 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2023-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-06-27 11:15:00 | 5109.30 | 5073.72 | 5069.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-06-27 13:15:00 | 5122.90 | 5087.24 | 5076.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-06-28 14:15:00 | 5126.35 | 5128.37 | 5109.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-06-28 14:45:00 | 5120.10 | 5128.37 | 5109.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-06-30 09:15:00 | 5108.15 | 5124.92 | 5111.04 | EMA400 retest candle locked (from upside) |

### Cycle 9 — SELL (started 2023-06-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-06-30 12:15:00 | 5081.05 | 5102.07 | 5102.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-03 12:15:00 | 5068.00 | 5091.32 | 5096.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-04 12:15:00 | 5075.75 | 5074.41 | 5083.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-04 13:00:00 | 5075.75 | 5074.41 | 5083.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 13:15:00 | 5110.00 | 5081.52 | 5086.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-04 13:45:00 | 5103.50 | 5081.52 | 5086.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-04 14:15:00 | 5109.00 | 5087.02 | 5088.27 | EMA400 retest candle locked (from downside) |

### Cycle 10 — BUY (started 2023-07-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-04 15:15:00 | 5114.00 | 5092.42 | 5090.61 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2023-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-05 11:15:00 | 5077.80 | 5089.22 | 5089.71 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2023-07-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-06 09:15:00 | 5205.00 | 5107.37 | 5096.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-06 10:15:00 | 5225.00 | 5130.90 | 5108.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-07 10:15:00 | 5218.25 | 5226.39 | 5179.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-07-07 10:45:00 | 5205.25 | 5226.39 | 5179.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 14:15:00 | 5135.65 | 5195.95 | 5179.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-07 15:00:00 | 5135.65 | 5195.95 | 5179.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-07 15:15:00 | 5151.50 | 5187.06 | 5176.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 09:15:00 | 5135.90 | 5187.06 | 5176.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 10:15:00 | 5179.45 | 5180.74 | 5175.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-10 11:15:00 | 5165.35 | 5180.74 | 5175.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-10 11:15:00 | 5170.55 | 5178.70 | 5174.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-10 12:15:00 | 5185.00 | 5178.70 | 5174.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-10 15:15:00 | 5160.00 | 5172.98 | 5173.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — SELL (started 2023-07-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-10 15:15:00 | 5160.00 | 5172.98 | 5173.64 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2023-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-11 09:15:00 | 5246.65 | 5187.71 | 5180.28 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2023-07-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-13 14:15:00 | 5184.00 | 5216.71 | 5219.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-14 09:15:00 | 5114.55 | 5192.00 | 5207.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-14 14:15:00 | 5178.60 | 5165.82 | 5185.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-07-14 15:00:00 | 5178.60 | 5165.82 | 5185.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-17 09:15:00 | 5299.90 | 5195.66 | 5196.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-07-17 10:00:00 | 5299.90 | 5195.66 | 5196.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2023-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-17 10:15:00 | 5232.30 | 5202.99 | 5199.44 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2023-07-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-18 11:15:00 | 5116.10 | 5193.17 | 5200.91 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2023-07-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-20 15:15:00 | 5197.00 | 5187.74 | 5187.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-21 09:15:00 | 5233.80 | 5196.95 | 5191.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-21 11:15:00 | 5195.55 | 5202.48 | 5195.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-21 11:15:00 | 5195.55 | 5202.48 | 5195.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 11:15:00 | 5195.55 | 5202.48 | 5195.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-21 11:30:00 | 5200.45 | 5202.48 | 5195.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-21 12:15:00 | 5220.30 | 5206.04 | 5197.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-07-21 13:30:00 | 5225.95 | 5206.92 | 5198.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-07-24 14:15:00 | 5187.15 | 5201.47 | 5200.44 | SL hit (close<static) qty=1.00 sl=5193.20 alert=retest2 |

### Cycle 19 — SELL (started 2023-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-24 15:15:00 | 5165.00 | 5194.18 | 5197.22 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2023-07-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-25 14:15:00 | 5214.95 | 5196.97 | 5196.70 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2023-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-26 11:15:00 | 5186.55 | 5196.72 | 5196.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-07-26 12:15:00 | 5172.85 | 5191.94 | 5194.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-07-27 09:15:00 | 5208.50 | 5180.74 | 5187.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-27 09:15:00 | 5208.50 | 5180.74 | 5187.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-27 09:15:00 | 5208.50 | 5180.74 | 5187.22 | EMA400 retest candle locked (from downside) |

### Cycle 22 — BUY (started 2023-07-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-07-27 11:15:00 | 5205.60 | 5191.96 | 5191.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-07-27 14:15:00 | 5220.65 | 5204.59 | 5198.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-07-31 09:15:00 | 5220.00 | 5271.94 | 5244.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-07-31 09:15:00 | 5220.00 | 5271.94 | 5244.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 09:15:00 | 5220.00 | 5271.94 | 5244.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-31 09:45:00 | 5207.00 | 5271.94 | 5244.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-07-31 10:15:00 | 5211.05 | 5259.76 | 5241.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-07-31 10:30:00 | 5208.55 | 5259.76 | 5241.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2023-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-07-31 14:15:00 | 5170.20 | 5222.31 | 5227.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-01 09:15:00 | 5100.15 | 5191.91 | 5212.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-07 09:15:00 | 4996.45 | 4988.26 | 5008.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-08-07 10:00:00 | 4996.45 | 4988.26 | 5008.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 10:15:00 | 5006.55 | 4991.92 | 5008.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-07 10:30:00 | 5003.00 | 4991.92 | 5008.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 11:15:00 | 5002.95 | 4994.13 | 5007.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-07 11:30:00 | 5017.25 | 4994.13 | 5007.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 12:15:00 | 5010.25 | 4997.35 | 5007.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-07 12:45:00 | 5011.10 | 4997.35 | 5007.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-07 13:15:00 | 5002.45 | 4998.37 | 5007.45 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2023-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-08 12:15:00 | 5042.70 | 5015.42 | 5012.71 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2023-08-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-09 10:15:00 | 4988.40 | 5010.94 | 5012.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-09 11:15:00 | 4960.70 | 5000.89 | 5007.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-11 09:15:00 | 4994.50 | 4948.65 | 4965.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-11 09:15:00 | 4994.50 | 4948.65 | 4965.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 09:15:00 | 4994.50 | 4948.65 | 4965.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-11 09:45:00 | 4997.00 | 4948.65 | 4965.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-11 10:15:00 | 4969.90 | 4952.90 | 4965.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-11 11:15:00 | 4957.45 | 4952.90 | 4965.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-16 15:15:00 | 4946.00 | 4904.96 | 4899.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2023-08-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-16 15:15:00 | 4946.00 | 4904.96 | 4899.49 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2023-08-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-17 12:15:00 | 4878.45 | 4895.80 | 4896.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-18 09:15:00 | 4839.40 | 4880.35 | 4888.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-18 11:15:00 | 4880.05 | 4875.40 | 4884.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-18 11:15:00 | 4880.05 | 4875.40 | 4884.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 11:15:00 | 4880.05 | 4875.40 | 4884.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-18 11:45:00 | 4880.40 | 4875.40 | 4884.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 13:15:00 | 4894.25 | 4877.91 | 4884.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-18 14:00:00 | 4894.25 | 4877.91 | 4884.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-18 14:15:00 | 4847.00 | 4871.73 | 4880.88 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2023-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-21 14:15:00 | 4931.85 | 4888.87 | 4883.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-08-22 11:15:00 | 4949.00 | 4916.12 | 4899.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-08-22 14:15:00 | 4919.90 | 4924.65 | 4907.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-08-22 14:45:00 | 4914.05 | 4924.65 | 4907.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-23 09:15:00 | 4976.95 | 4935.08 | 4915.52 | EMA400 retest candle locked (from upside) |

### Cycle 29 — SELL (started 2023-08-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-24 15:15:00 | 4907.00 | 4929.66 | 4930.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-25 09:15:00 | 4872.95 | 4918.32 | 4925.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-28 10:15:00 | 4896.65 | 4882.75 | 4897.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-28 10:15:00 | 4896.65 | 4882.75 | 4897.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 10:15:00 | 4896.65 | 4882.75 | 4897.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-28 11:00:00 | 4896.65 | 4882.75 | 4897.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-28 11:15:00 | 4884.00 | 4883.00 | 4896.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-29 09:30:00 | 4872.00 | 4892.47 | 4897.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-08-29 10:15:00 | 4906.00 | 4895.18 | 4897.83 | SL hit (close>static) qty=1.00 sl=4897.90 alert=retest2 |

### Cycle 30 — BUY (started 2023-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-08-29 11:15:00 | 4924.10 | 4900.96 | 4900.22 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2023-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-08-29 14:15:00 | 4880.80 | 4896.26 | 4898.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-08-29 15:15:00 | 4874.90 | 4891.99 | 4896.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-08-30 13:15:00 | 4880.40 | 4879.90 | 4887.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-08-30 13:15:00 | 4880.40 | 4879.90 | 4887.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-30 13:15:00 | 4880.40 | 4879.90 | 4887.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-30 13:30:00 | 4884.90 | 4879.90 | 4887.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 09:15:00 | 4887.05 | 4881.81 | 4886.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-31 09:45:00 | 4895.00 | 4881.81 | 4886.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 10:15:00 | 4884.00 | 4882.25 | 4886.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-31 10:30:00 | 4879.00 | 4882.25 | 4886.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 11:15:00 | 4876.00 | 4881.00 | 4885.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-08-31 11:30:00 | 4889.30 | 4881.00 | 4885.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-08-31 12:15:00 | 4858.45 | 4876.49 | 4882.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-08-31 13:45:00 | 4850.10 | 4869.82 | 4879.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-05 09:15:00 | 4950.00 | 4854.64 | 4849.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2023-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-05 09:15:00 | 4950.00 | 4854.64 | 4849.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-05 10:15:00 | 4963.50 | 4876.41 | 4860.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-08 09:15:00 | 5012.00 | 5012.89 | 4986.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-08 09:15:00 | 5012.00 | 5012.89 | 4986.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-08 09:15:00 | 5012.00 | 5012.89 | 4986.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-11 09:15:00 | 5078.00 | 5000.51 | 4991.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-12 12:15:00 | 4983.05 | 5030.37 | 5028.13 | SL hit (close<static) qty=1.00 sl=4985.00 alert=retest2 |

### Cycle 33 — SELL (started 2023-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-12 13:15:00 | 4980.00 | 5020.30 | 5023.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-09-13 09:15:00 | 4968.35 | 4998.46 | 5011.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-09-13 11:15:00 | 5010.00 | 5000.01 | 5010.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-13 11:15:00 | 5010.00 | 5000.01 | 5010.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-13 11:15:00 | 5010.00 | 5000.01 | 5010.23 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2023-09-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-14 09:15:00 | 5053.80 | 5016.86 | 5014.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-15 09:15:00 | 5093.70 | 5068.06 | 5046.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-15 14:15:00 | 5082.65 | 5098.14 | 5072.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-09-15 15:00:00 | 5082.65 | 5098.14 | 5072.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-18 09:15:00 | 5152.85 | 5109.38 | 5082.21 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2023-09-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-20 09:15:00 | 4960.80 | 5066.60 | 5074.00 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2023-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-25 11:15:00 | 5090.00 | 5019.54 | 5017.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-25 12:15:00 | 5091.90 | 5034.01 | 5024.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-09-26 09:15:00 | 5061.25 | 5062.18 | 5043.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-09-26 09:15:00 | 5061.25 | 5062.18 | 5043.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 09:15:00 | 5061.25 | 5062.18 | 5043.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 09:30:00 | 5046.20 | 5062.18 | 5043.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 10:15:00 | 5081.15 | 5065.98 | 5046.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 10:45:00 | 5058.25 | 5065.98 | 5046.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 14:15:00 | 5054.00 | 5066.93 | 5053.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-09-26 15:00:00 | 5054.00 | 5066.93 | 5053.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-09-26 15:15:00 | 5050.00 | 5063.55 | 5053.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-09-27 09:15:00 | 5064.55 | 5063.55 | 5053.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-09-27 09:15:00 | 5038.00 | 5058.44 | 5051.74 | SL hit (close<static) qty=1.00 sl=5050.00 alert=retest2 |

### Cycle 37 — SELL (started 2023-09-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-09-28 14:15:00 | 5021.00 | 5070.48 | 5072.21 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2023-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-09-29 10:15:00 | 5122.50 | 5081.26 | 5076.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-09-29 11:15:00 | 5141.85 | 5093.38 | 5082.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-03 09:15:00 | 5105.00 | 5122.70 | 5104.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-03 09:15:00 | 5105.00 | 5122.70 | 5104.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-03 09:15:00 | 5105.00 | 5122.70 | 5104.58 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2023-10-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-04 09:15:00 | 5035.95 | 5103.79 | 5105.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-04 11:15:00 | 4986.25 | 5071.96 | 5089.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-05 09:15:00 | 5064.15 | 5055.19 | 5072.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-05 09:15:00 | 5064.15 | 5055.19 | 5072.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-05 09:15:00 | 5064.15 | 5055.19 | 5072.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-09 12:45:00 | 5028.60 | 5047.10 | 5054.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-10 12:15:00 | 5063.25 | 5054.64 | 5054.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2023-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-10 12:15:00 | 5063.25 | 5054.64 | 5054.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-10-11 09:15:00 | 5136.00 | 5074.85 | 5063.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-10-11 14:15:00 | 5096.90 | 5102.87 | 5084.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-10-11 15:00:00 | 5096.90 | 5102.87 | 5084.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-11 15:15:00 | 5083.00 | 5098.90 | 5084.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 09:15:00 | 5082.65 | 5098.90 | 5084.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 09:15:00 | 5051.10 | 5089.34 | 5081.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-10-12 10:00:00 | 5051.10 | 5089.34 | 5081.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-12 10:15:00 | 5070.50 | 5085.57 | 5080.50 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2023-10-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-12 12:15:00 | 5025.00 | 5067.46 | 5072.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-12 13:15:00 | 4937.50 | 5041.47 | 5060.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-13 12:15:00 | 5009.15 | 5001.46 | 5028.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-10-13 13:00:00 | 5009.15 | 5001.46 | 5028.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 13:15:00 | 5012.65 | 5003.70 | 5026.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-13 14:00:00 | 5012.65 | 5003.70 | 5026.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-13 14:15:00 | 5014.80 | 5005.92 | 5025.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-13 14:30:00 | 5018.05 | 5005.92 | 5025.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-16 09:15:00 | 5011.50 | 5008.81 | 5023.66 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2023-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-17 14:15:00 | 5029.80 | 5027.35 | 5027.28 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2023-10-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-18 09:15:00 | 4998.40 | 5021.50 | 5024.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-18 11:15:00 | 4963.00 | 5004.89 | 5016.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-19 11:15:00 | 5016.90 | 4987.45 | 4998.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-10-19 11:15:00 | 5016.90 | 4987.45 | 4998.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 11:15:00 | 5016.90 | 4987.45 | 4998.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 12:00:00 | 5016.90 | 4987.45 | 4998.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 12:15:00 | 5017.05 | 4993.37 | 5000.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-19 13:00:00 | 5017.05 | 4993.37 | 5000.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-19 15:15:00 | 4968.00 | 4990.52 | 4997.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 09:15:00 | 4959.60 | 4990.52 | 4997.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 12:30:00 | 4967.00 | 4979.62 | 4989.05 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-20 15:15:00 | 4952.00 | 4979.95 | 4987.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-23 15:00:00 | 4966.45 | 4981.27 | 4984.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 09:15:00 | 4855.30 | 4800.99 | 4847.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 10:00:00 | 4855.30 | 4800.99 | 4847.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 10:15:00 | 4828.05 | 4806.40 | 4845.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 10:30:00 | 4837.85 | 4806.40 | 4845.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 11:15:00 | 4851.50 | 4815.42 | 4846.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-10-27 12:00:00 | 4851.50 | 4815.42 | 4846.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-10-27 12:15:00 | 4847.75 | 4821.89 | 4846.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-27 14:45:00 | 4835.15 | 4827.65 | 4845.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-10-30 11:00:00 | 4827.70 | 4829.28 | 4841.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-10-31 09:15:00 | 4879.60 | 4848.13 | 4846.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2023-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-10-31 09:15:00 | 4879.60 | 4848.13 | 4846.70 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2023-10-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-10-31 12:15:00 | 4832.65 | 4845.95 | 4846.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-10-31 13:15:00 | 4826.20 | 4842.00 | 4844.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-10-31 15:15:00 | 4842.00 | 4839.16 | 4842.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-01 09:15:00 | 4834.55 | 4839.16 | 4842.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 09:15:00 | 4844.30 | 4840.19 | 4842.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-01 09:30:00 | 4836.90 | 4840.19 | 4842.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-01 10:15:00 | 4833.15 | 4838.78 | 4841.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2023-11-01 13:00:00 | 4817.50 | 4834.12 | 4839.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-02 10:15:00 | 4868.20 | 4834.12 | 4835.62 | SL hit (close>static) qty=1.00 sl=4856.00 alert=retest2 |

### Cycle 46 — BUY (started 2023-11-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-02 11:15:00 | 4874.90 | 4842.28 | 4839.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-02 15:15:00 | 4890.00 | 4865.17 | 4852.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-11-07 10:15:00 | 5139.25 | 5145.27 | 5087.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-11-07 11:00:00 | 5139.25 | 5145.27 | 5087.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 11:15:00 | 5119.35 | 5131.30 | 5110.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 12:00:00 | 5119.35 | 5131.30 | 5110.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 12:15:00 | 5098.70 | 5124.78 | 5109.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-11-08 12:45:00 | 5099.70 | 5124.78 | 5109.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-08 13:15:00 | 5099.35 | 5119.69 | 5108.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-11-08 15:00:00 | 5110.00 | 5117.75 | 5108.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-11-24 10:15:00 | 5447.50 | 5478.27 | 5479.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2023-11-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-11-24 10:15:00 | 5447.50 | 5478.27 | 5479.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-11-24 11:15:00 | 5439.10 | 5470.44 | 5475.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-11-28 15:15:00 | 5392.55 | 5379.15 | 5409.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-11-29 09:15:00 | 5388.15 | 5379.15 | 5409.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 09:15:00 | 5404.35 | 5384.19 | 5408.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 09:45:00 | 5405.60 | 5384.19 | 5408.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 10:15:00 | 5412.25 | 5389.80 | 5408.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-11-29 11:00:00 | 5412.25 | 5389.80 | 5408.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-11-29 11:15:00 | 5413.75 | 5394.59 | 5409.39 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2023-11-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-11-30 09:15:00 | 5466.30 | 5422.80 | 5418.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-11-30 10:15:00 | 5506.00 | 5439.44 | 5426.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-05 09:15:00 | 5579.30 | 5604.04 | 5570.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-05 09:15:00 | 5579.30 | 5604.04 | 5570.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 09:15:00 | 5579.30 | 5604.04 | 5570.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-05 10:00:00 | 5579.30 | 5604.04 | 5570.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-05 10:15:00 | 5584.30 | 5600.09 | 5571.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 12:45:00 | 5604.60 | 5599.50 | 5576.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-05 14:00:00 | 5596.95 | 5598.99 | 5578.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2023-12-06 09:15:00 | 5621.50 | 5594.12 | 5579.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2023-12-06 12:15:00 | 5548.35 | 5567.80 | 5570.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2023-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-06 12:15:00 | 5548.35 | 5567.80 | 5570.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-07 09:15:00 | 5489.70 | 5545.94 | 5558.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-08 11:15:00 | 5492.10 | 5478.61 | 5505.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-08 11:30:00 | 5487.70 | 5478.61 | 5505.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 12:15:00 | 5503.80 | 5483.64 | 5505.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-08 12:30:00 | 5502.35 | 5483.64 | 5505.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-08 13:15:00 | 5501.60 | 5487.24 | 5504.77 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2023-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-11 11:15:00 | 5528.25 | 5515.31 | 5514.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-11 14:15:00 | 5545.95 | 5527.62 | 5520.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-12 09:15:00 | 5509.50 | 5527.74 | 5522.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-12 09:15:00 | 5509.50 | 5527.74 | 5522.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 09:15:00 | 5509.50 | 5527.74 | 5522.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 09:30:00 | 5501.05 | 5527.74 | 5522.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-12 10:15:00 | 5498.80 | 5521.95 | 5520.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-12 10:45:00 | 5497.50 | 5521.95 | 5520.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2023-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-12 11:15:00 | 5500.00 | 5517.56 | 5518.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-12 12:15:00 | 5446.80 | 5503.41 | 5511.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-13 14:15:00 | 5434.10 | 5420.38 | 5451.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2023-12-13 15:00:00 | 5434.10 | 5420.38 | 5451.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-13 15:15:00 | 5444.50 | 5425.20 | 5451.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-14 09:15:00 | 5482.95 | 5425.20 | 5451.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-14 09:15:00 | 5480.95 | 5436.35 | 5453.91 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2023-12-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-14 13:15:00 | 5493.25 | 5469.19 | 5466.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-14 14:15:00 | 5511.55 | 5477.67 | 5470.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-18 10:15:00 | 5520.95 | 5537.51 | 5516.53 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2023-12-18 11:00:00 | 5520.95 | 5537.51 | 5516.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 11:15:00 | 5509.15 | 5531.84 | 5515.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 11:30:00 | 5505.70 | 5531.84 | 5515.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-18 12:15:00 | 5500.55 | 5525.58 | 5514.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-18 13:00:00 | 5500.55 | 5525.58 | 5514.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-20 13:15:00 | 5456.00 | 5537.96 | 5534.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2023-12-20 14:00:00 | 5456.00 | 5537.96 | 5534.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — SELL (started 2023-12-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2023-12-20 14:15:00 | 5401.15 | 5510.60 | 5522.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2023-12-20 15:15:00 | 5394.00 | 5487.28 | 5510.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2023-12-21 11:15:00 | 5497.90 | 5476.60 | 5498.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-21 11:15:00 | 5497.90 | 5476.60 | 5498.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 11:15:00 | 5497.90 | 5476.60 | 5498.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 11:45:00 | 5522.00 | 5476.60 | 5498.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 12:15:00 | 5485.00 | 5478.28 | 5497.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2023-12-21 12:30:00 | 5506.45 | 5478.28 | 5497.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-21 15:15:00 | 5503.90 | 5480.63 | 5493.65 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2023-12-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2023-12-22 12:15:00 | 5522.00 | 5500.75 | 5499.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2023-12-22 14:15:00 | 5541.95 | 5512.77 | 5505.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2023-12-29 09:15:00 | 5685.00 | 5715.41 | 5677.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2023-12-29 09:15:00 | 5685.00 | 5715.41 | 5677.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2023-12-29 09:15:00 | 5685.00 | 5715.41 | 5677.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-01 14:00:00 | 5764.45 | 5723.41 | 5703.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-03 14:00:00 | 5771.50 | 5738.88 | 5727.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-04 10:30:00 | 5770.05 | 5754.37 | 5739.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-04 11:30:00 | 5759.75 | 5752.91 | 5740.21 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-05 09:15:00 | 5779.35 | 5761.76 | 5749.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-05 12:00:00 | 5796.15 | 5771.08 | 5756.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-01-08 09:15:00 | 5694.00 | 5748.07 | 5750.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2024-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-01-08 09:15:00 | 5694.00 | 5748.07 | 5750.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-01-08 10:15:00 | 5655.85 | 5729.63 | 5741.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-01-09 09:15:00 | 5739.15 | 5705.33 | 5720.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-01-09 09:15:00 | 5739.15 | 5705.33 | 5720.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 09:15:00 | 5739.15 | 5705.33 | 5720.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-09 10:00:00 | 5739.15 | 5705.33 | 5720.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-09 10:15:00 | 5783.80 | 5721.02 | 5725.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-01-09 11:00:00 | 5783.80 | 5721.02 | 5725.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — BUY (started 2024-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-01-09 11:15:00 | 5786.00 | 5734.02 | 5731.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-01-09 12:15:00 | 5817.00 | 5750.62 | 5739.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-01-10 09:15:00 | 5772.80 | 5774.44 | 5756.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-01-10 10:00:00 | 5772.80 | 5774.44 | 5756.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 10:15:00 | 5746.65 | 5768.88 | 5755.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 11:00:00 | 5746.65 | 5768.88 | 5755.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 11:15:00 | 5740.00 | 5763.10 | 5753.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-01-10 12:00:00 | 5740.00 | 5763.10 | 5753.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-01-10 12:15:00 | 5748.45 | 5760.17 | 5753.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-01-10 14:30:00 | 5763.95 | 5767.62 | 5757.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-01-23 09:15:00 | 6340.35 | 6154.74 | 6088.65 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2024-02-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-05 11:15:00 | 6275.00 | 6307.37 | 6310.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-05 12:15:00 | 6248.05 | 6295.51 | 6304.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-08 11:15:00 | 6184.70 | 6179.38 | 6201.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-08 11:45:00 | 6180.95 | 6179.38 | 6201.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 14:15:00 | 6234.90 | 6188.90 | 6200.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-08 14:45:00 | 6199.55 | 6188.90 | 6200.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-08 15:15:00 | 6216.70 | 6194.46 | 6201.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-02-09 09:15:00 | 6235.55 | 6194.46 | 6201.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2024-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-09 10:15:00 | 6295.00 | 6222.09 | 6213.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-09 11:15:00 | 6320.35 | 6241.75 | 6223.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-15 09:15:00 | 6684.00 | 6723.05 | 6660.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-02-15 10:00:00 | 6684.00 | 6723.05 | 6660.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 10:15:00 | 6685.95 | 6715.63 | 6663.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 10:45:00 | 6665.80 | 6715.63 | 6663.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 11:15:00 | 6650.05 | 6702.51 | 6662.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 12:00:00 | 6650.05 | 6702.51 | 6662.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 12:15:00 | 6652.15 | 6692.44 | 6661.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-15 12:30:00 | 6652.00 | 6692.44 | 6661.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-15 14:15:00 | 6625.00 | 6663.36 | 6652.36 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2024-02-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-16 09:15:00 | 6550.00 | 6635.35 | 6641.21 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-02-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-02-19 13:15:00 | 6658.20 | 6619.24 | 6618.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-02-20 09:15:00 | 6708.20 | 6646.72 | 6632.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-02-21 15:15:00 | 6759.65 | 6765.03 | 6728.33 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-02-22 09:15:00 | 6838.80 | 6765.03 | 6728.33 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 09:15:00 | 6739.25 | 6759.87 | 6729.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 10:00:00 | 6739.25 | 6759.87 | 6729.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-22 10:15:00 | 6763.45 | 6760.59 | 6732.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-02-22 10:45:00 | 6708.90 | 6760.59 | 6732.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-02-23 09:15:00 | 6779.15 | 6767.50 | 6748.83 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-02-23 11:15:00 | 6727.05 | 6759.11 | 6748.23 | SL hit (close<ema400) qty=1.00 sl=6748.23 alert=retest1 |

### Cycle 61 — SELL (started 2024-02-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-02-26 10:15:00 | 6690.45 | 6740.64 | 6744.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-02-26 11:15:00 | 6660.40 | 6724.59 | 6736.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-02-27 12:15:00 | 6653.00 | 6643.76 | 6675.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-02-27 12:45:00 | 6652.95 | 6643.76 | 6675.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-06 15:15:00 | 6064.05 | 6046.71 | 6065.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 09:15:00 | 6103.85 | 6046.71 | 6065.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 09:15:00 | 6105.35 | 6058.44 | 6069.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 10:00:00 | 6105.35 | 6058.44 | 6069.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 10:15:00 | 6133.90 | 6073.53 | 6075.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 11:00:00 | 6133.90 | 6073.53 | 6075.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 12:15:00 | 6085.00 | 6074.62 | 6075.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-07 12:45:00 | 6087.60 | 6074.62 | 6075.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-07 13:15:00 | 6058.35 | 6071.37 | 6073.83 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2024-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-11 11:15:00 | 6115.20 | 6072.73 | 6072.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-11 12:15:00 | 6139.00 | 6085.98 | 6078.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-12 10:15:00 | 6147.20 | 6148.30 | 6117.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-12 10:45:00 | 6155.00 | 6148.30 | 6117.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 11:15:00 | 6151.20 | 6148.88 | 6120.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 11:30:00 | 6121.60 | 6148.88 | 6120.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 13:15:00 | 6134.55 | 6146.99 | 6124.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 14:00:00 | 6134.55 | 6146.99 | 6124.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-12 14:15:00 | 6147.25 | 6147.04 | 6126.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-12 14:30:00 | 6139.50 | 6147.04 | 6126.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 09:15:00 | 6123.00 | 6141.05 | 6127.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 10:00:00 | 6123.00 | 6141.05 | 6127.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-13 10:15:00 | 6090.45 | 6130.93 | 6124.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-13 11:15:00 | 6077.95 | 6130.93 | 6124.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 63 — SELL (started 2024-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-13 11:15:00 | 6004.40 | 6105.63 | 6113.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-13 13:15:00 | 5988.20 | 6065.96 | 6092.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-14 09:15:00 | 6048.50 | 6047.23 | 6076.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-14 09:15:00 | 6048.50 | 6047.23 | 6076.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 09:15:00 | 6048.50 | 6047.23 | 6076.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 09:30:00 | 6070.00 | 6047.23 | 6076.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-14 10:15:00 | 6038.70 | 6045.52 | 6072.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-14 10:45:00 | 6079.15 | 6045.52 | 6072.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 09:15:00 | 6007.20 | 6003.52 | 6024.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 09:30:00 | 6039.05 | 6003.52 | 6024.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 11:15:00 | 6061.20 | 6016.23 | 6026.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 12:00:00 | 6061.20 | 6016.23 | 6026.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-18 12:15:00 | 6057.30 | 6024.45 | 6029.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-18 13:15:00 | 6074.50 | 6024.45 | 6029.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2024-03-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-18 13:15:00 | 6105.55 | 6040.67 | 6036.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-18 14:15:00 | 6138.95 | 6060.32 | 6045.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-19 10:15:00 | 6060.00 | 6074.07 | 6057.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-19 10:15:00 | 6060.00 | 6074.07 | 6057.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 10:15:00 | 6060.00 | 6074.07 | 6057.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 10:45:00 | 6050.95 | 6074.07 | 6057.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 11:15:00 | 6073.05 | 6073.87 | 6058.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 11:30:00 | 6051.45 | 6073.87 | 6058.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-19 12:15:00 | 6096.00 | 6078.30 | 6062.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-03-19 12:30:00 | 6069.25 | 6078.30 | 6062.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-20 10:15:00 | 6134.20 | 6101.92 | 6080.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 11:30:00 | 6148.45 | 6110.02 | 6086.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-20 12:30:00 | 6149.00 | 6122.78 | 6094.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-03-21 09:45:00 | 6151.50 | 6156.62 | 6121.58 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-03-27 14:15:00 | 6237.95 | 6273.06 | 6275.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2024-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-03-27 14:15:00 | 6237.95 | 6273.06 | 6275.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-03-27 15:15:00 | 6225.00 | 6263.45 | 6271.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-03-28 09:15:00 | 6279.00 | 6266.56 | 6271.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-03-28 09:15:00 | 6279.00 | 6266.56 | 6271.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-03-28 09:15:00 | 6279.00 | 6266.56 | 6271.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-03-28 10:00:00 | 6279.00 | 6266.56 | 6271.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2024-03-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-03-28 10:15:00 | 6357.70 | 6284.79 | 6279.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-03-28 11:15:00 | 6417.35 | 6311.30 | 6292.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-03-28 14:15:00 | 6330.50 | 6333.54 | 6308.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-03-28 15:00:00 | 6330.50 | 6333.54 | 6308.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-03 09:15:00 | 6362.50 | 6414.20 | 6403.09 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2024-04-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-03 11:15:00 | 6366.65 | 6393.08 | 6394.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-04 09:15:00 | 6331.85 | 6372.89 | 6383.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-04 11:15:00 | 6395.80 | 6375.11 | 6382.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-04 11:15:00 | 6395.80 | 6375.11 | 6382.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 11:15:00 | 6395.80 | 6375.11 | 6382.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-04 12:00:00 | 6395.80 | 6375.11 | 6382.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-04 12:15:00 | 6415.15 | 6383.12 | 6385.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-04 12:45:00 | 6411.75 | 6383.12 | 6385.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2024-04-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-04 13:15:00 | 6450.90 | 6396.68 | 6391.77 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2024-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-08 09:15:00 | 6306.50 | 6383.77 | 6392.62 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2024-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-09 11:15:00 | 6493.20 | 6375.33 | 6370.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-09 12:15:00 | 6509.25 | 6402.12 | 6382.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-12 09:15:00 | 6439.50 | 6469.12 | 6446.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-12 09:15:00 | 6439.50 | 6469.12 | 6446.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 09:15:00 | 6439.50 | 6469.12 | 6446.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 09:45:00 | 6436.55 | 6469.12 | 6446.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 10:15:00 | 6425.50 | 6460.40 | 6444.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 11:00:00 | 6425.50 | 6460.40 | 6444.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-12 11:15:00 | 6388.20 | 6445.96 | 6439.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-12 12:00:00 | 6388.20 | 6445.96 | 6439.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2024-04-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-12 13:15:00 | 6416.80 | 6432.41 | 6433.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-15 09:15:00 | 6375.00 | 6415.24 | 6425.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-16 13:15:00 | 6316.70 | 6296.35 | 6335.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-04-16 14:00:00 | 6316.70 | 6296.35 | 6335.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 14:15:00 | 6343.95 | 6305.87 | 6336.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-16 15:00:00 | 6343.95 | 6305.87 | 6336.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-16 15:15:00 | 6325.00 | 6309.70 | 6335.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-04-18 09:15:00 | 6305.00 | 6309.70 | 6335.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-23 09:15:00 | 6252.85 | 6224.92 | 6221.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2024-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-23 09:15:00 | 6252.85 | 6224.92 | 6221.53 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2024-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-23 12:15:00 | 6204.25 | 6220.20 | 6220.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-04-23 14:15:00 | 6195.20 | 6212.95 | 6216.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-04-24 09:15:00 | 6244.70 | 6214.83 | 6216.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-24 09:15:00 | 6244.70 | 6214.83 | 6216.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-24 09:15:00 | 6244.70 | 6214.83 | 6216.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-04-24 10:00:00 | 6244.70 | 6214.83 | 6216.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2024-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-04-24 10:15:00 | 6248.10 | 6221.48 | 6219.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-04-24 11:15:00 | 6275.80 | 6232.34 | 6224.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-04-25 09:15:00 | 6219.40 | 6254.95 | 6242.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-04-25 09:15:00 | 6219.40 | 6254.95 | 6242.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 09:15:00 | 6219.40 | 6254.95 | 6242.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:00:00 | 6219.40 | 6254.95 | 6242.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 10:15:00 | 6225.70 | 6249.10 | 6240.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-04-25 10:45:00 | 6206.40 | 6249.10 | 6240.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-04-25 12:15:00 | 6242.65 | 6243.88 | 6239.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-04-25 13:15:00 | 6252.35 | 6243.88 | 6239.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-04-29 09:15:00 | 5814.85 | 6189.44 | 6234.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2024-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-04-29 09:15:00 | 5814.85 | 6189.44 | 6234.99 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2024-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-06 11:15:00 | 6037.60 | 6011.74 | 6008.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-06 14:15:00 | 6055.10 | 6028.63 | 6017.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-07 09:15:00 | 5999.55 | 6028.87 | 6020.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-07 09:15:00 | 5999.55 | 6028.87 | 6020.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 09:15:00 | 5999.55 | 6028.87 | 6020.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 10:00:00 | 5999.55 | 6028.87 | 6020.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-07 10:15:00 | 5983.20 | 6019.74 | 6016.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-07 10:30:00 | 5979.90 | 6019.74 | 6016.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2024-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-07 11:15:00 | 5893.40 | 5994.47 | 6005.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-07 12:15:00 | 5886.10 | 5972.80 | 5994.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-10 09:15:00 | 5833.75 | 5831.98 | 5866.28 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-10 10:00:00 | 5833.75 | 5831.98 | 5866.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 10:15:00 | 5864.45 | 5838.47 | 5866.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-10 10:30:00 | 5877.05 | 5838.47 | 5866.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-10 11:15:00 | 5844.70 | 5839.72 | 5864.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-10 13:00:00 | 5835.75 | 5838.92 | 5861.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-13 09:15:00 | 5817.80 | 5840.24 | 5856.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-14 09:15:00 | 5790.35 | 5828.43 | 5840.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-14 14:15:00 | 5865.20 | 5842.65 | 5841.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 78 — BUY (started 2024-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 14:15:00 | 5865.20 | 5842.65 | 5841.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-16 09:15:00 | 5900.00 | 5863.96 | 5855.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 11:15:00 | 5958.15 | 5959.37 | 5934.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-21 11:30:00 | 5960.95 | 5959.37 | 5934.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 13:15:00 | 5932.15 | 5952.43 | 5935.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 14:00:00 | 5932.15 | 5952.43 | 5935.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 5950.00 | 5951.94 | 5937.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 14:30:00 | 5930.00 | 5951.94 | 5937.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 5932.30 | 5948.01 | 5936.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:15:00 | 5908.25 | 5948.01 | 5936.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 5890.00 | 5936.41 | 5932.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 10:00:00 | 5890.00 | 5936.41 | 5932.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — SELL (started 2024-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 10:15:00 | 5867.45 | 5922.62 | 5926.48 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2024-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 12:15:00 | 5952.05 | 5919.63 | 5916.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 14:15:00 | 5975.10 | 5938.01 | 5925.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 09:15:00 | 5878.80 | 5928.09 | 5923.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 5878.80 | 5928.09 | 5923.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 5878.80 | 5928.09 | 5923.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 10:00:00 | 5878.80 | 5928.09 | 5923.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2024-05-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 10:15:00 | 5880.85 | 5918.64 | 5919.60 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2024-05-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-24 15:15:00 | 5970.00 | 5924.09 | 5920.71 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2024-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-28 11:15:00 | 5898.35 | 5922.74 | 5924.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-28 12:15:00 | 5888.65 | 5915.92 | 5921.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-29 09:15:00 | 5934.75 | 5910.37 | 5915.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 5934.75 | 5910.37 | 5915.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 5934.75 | 5910.37 | 5915.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-29 10:00:00 | 5934.75 | 5910.37 | 5915.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 10:15:00 | 5915.50 | 5911.40 | 5915.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-29 15:15:00 | 5898.00 | 5916.11 | 5916.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 10:00:00 | 5890.55 | 5826.22 | 5835.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 12:15:00 | 5871.55 | 5846.13 | 5842.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2024-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 12:15:00 | 5871.55 | 5846.13 | 5842.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-04 09:15:00 | 5931.55 | 5869.10 | 5855.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 5802.95 | 5855.87 | 5850.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 5802.95 | 5855.87 | 5850.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 5802.95 | 5855.87 | 5850.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 5802.95 | 5855.87 | 5850.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 5763.70 | 5837.44 | 5842.58 | EMA200 below EMA400 |

### Cycle 86 — BUY (started 2024-06-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 14:15:00 | 5874.95 | 5847.27 | 5845.75 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2024-06-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 15:15:00 | 5826.00 | 5843.02 | 5843.95 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2024-06-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 10:15:00 | 5885.00 | 5852.10 | 5847.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-05 14:15:00 | 5944.25 | 5874.28 | 5859.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-12 09:15:00 | 6090.05 | 6099.46 | 6061.44 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 11:30:00 | 6138.00 | 6109.94 | 6072.97 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-12 14:00:00 | 6142.10 | 6117.46 | 6082.87 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 6197.75 | 6196.68 | 6174.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 15:00:00 | 6242.90 | 6204.26 | 6185.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-19 09:15:00 | 6140.85 | 6197.29 | 6186.18 | SL hit (close<ema400) qty=1.00 sl=6186.18 alert=retest1 |

### Cycle 89 — SELL (started 2024-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-19 15:15:00 | 6159.95 | 6179.61 | 6180.66 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-21 09:15:00 | 6209.00 | 6176.44 | 6176.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-24 10:15:00 | 6242.75 | 6196.13 | 6186.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-26 09:15:00 | 6209.10 | 6264.91 | 6248.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 09:15:00 | 6209.10 | 6264.91 | 6248.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 09:15:00 | 6209.10 | 6264.91 | 6248.79 | EMA400 retest candle locked (from upside) |

### Cycle 91 — SELL (started 2024-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-26 11:15:00 | 6187.30 | 6237.26 | 6238.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 14:15:00 | 6136.85 | 6201.26 | 6220.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-27 14:15:00 | 6197.45 | 6162.46 | 6184.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 14:15:00 | 6197.45 | 6162.46 | 6184.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 14:15:00 | 6197.45 | 6162.46 | 6184.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-27 15:00:00 | 6197.45 | 6162.46 | 6184.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 15:15:00 | 6180.00 | 6165.97 | 6183.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-28 09:15:00 | 6202.80 | 6165.97 | 6183.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-28 09:15:00 | 6202.65 | 6173.30 | 6185.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-28 12:30:00 | 6157.10 | 6174.22 | 6183.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 09:15:00 | 6113.00 | 6177.46 | 6182.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-03 12:15:00 | 6183.75 | 6139.76 | 6139.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — BUY (started 2024-07-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 12:15:00 | 6183.75 | 6139.76 | 6139.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-03 13:15:00 | 6198.00 | 6151.40 | 6144.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 11:15:00 | 6185.90 | 6187.28 | 6167.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 12:00:00 | 6185.90 | 6187.28 | 6167.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 6227.60 | 6209.64 | 6187.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-05 14:00:00 | 6287.10 | 6242.29 | 6211.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-09 11:30:00 | 6281.55 | 6293.05 | 6273.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-19 13:15:00 | 6401.00 | 6439.59 | 6443.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2024-07-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-19 13:15:00 | 6401.00 | 6439.59 | 6443.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 14:15:00 | 6386.00 | 6428.87 | 6438.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 13:15:00 | 6411.00 | 6403.80 | 6418.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 13:15:00 | 6411.00 | 6403.80 | 6418.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 6411.00 | 6403.80 | 6418.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 6422.05 | 6403.80 | 6418.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 14:15:00 | 6399.95 | 6403.03 | 6416.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 14:30:00 | 6419.90 | 6403.03 | 6416.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 6414.00 | 6405.22 | 6416.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 6436.80 | 6405.22 | 6416.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 6413.00 | 6406.78 | 6416.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:45:00 | 6444.20 | 6406.78 | 6416.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 6402.65 | 6405.95 | 6415.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 11:15:00 | 6417.55 | 6405.95 | 6415.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 6415.00 | 6407.76 | 6415.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:00:00 | 6415.00 | 6407.76 | 6415.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 12:15:00 | 6419.90 | 6410.19 | 6415.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:30:00 | 6408.85 | 6410.19 | 6415.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 13:15:00 | 6432.30 | 6414.61 | 6416.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 14:00:00 | 6432.30 | 6414.61 | 6416.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 14:15:00 | 6375.10 | 6406.71 | 6413.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 15:00:00 | 6375.10 | 6406.71 | 6413.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 11:15:00 | 6402.45 | 6399.85 | 6407.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-24 12:15:00 | 6391.05 | 6399.85 | 6407.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 15:15:00 | 6422.65 | 6410.93 | 6410.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 94 — BUY (started 2024-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 15:15:00 | 6422.65 | 6410.93 | 6410.46 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2024-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-25 09:15:00 | 6402.75 | 6409.30 | 6409.76 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2024-07-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 10:15:00 | 6595.20 | 6444.73 | 6424.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 11:15:00 | 6602.65 | 6476.31 | 6440.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 09:15:00 | 6635.60 | 6652.84 | 6594.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-30 10:00:00 | 6635.60 | 6652.84 | 6594.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 13:15:00 | 6630.00 | 6644.18 | 6608.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 14:30:00 | 6637.80 | 6643.34 | 6611.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 09:30:00 | 6649.15 | 6643.60 | 6617.16 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 10:00:00 | 6641.20 | 6643.60 | 6617.16 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-31 11:45:00 | 6637.00 | 6638.51 | 6619.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 12:15:00 | 6621.00 | 6635.01 | 6619.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 13:00:00 | 6621.00 | 6635.01 | 6619.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 13:15:00 | 6612.50 | 6630.50 | 6618.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 14:00:00 | 6612.50 | 6630.50 | 6618.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 14:15:00 | 6600.00 | 6624.40 | 6617.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-31 15:00:00 | 6600.00 | 6624.40 | 6617.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 15:15:00 | 6598.00 | 6619.12 | 6615.34 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-01 10:00:00 | 6670.40 | 6629.38 | 6620.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-05 09:45:00 | 6707.00 | 6716.44 | 6694.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-06 14:15:00 | 6654.05 | 6687.16 | 6691.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 97 — SELL (started 2024-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 14:15:00 | 6654.05 | 6687.16 | 6691.36 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2024-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 12:15:00 | 6746.25 | 6703.03 | 6697.34 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2024-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 11:15:00 | 6641.35 | 6687.84 | 6693.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 12:15:00 | 6598.45 | 6669.96 | 6685.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 14:15:00 | 6593.00 | 6573.95 | 6609.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-09 14:45:00 | 6582.55 | 6573.95 | 6609.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-12 15:15:00 | 6555.00 | 6517.99 | 6554.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 09:15:00 | 6611.90 | 6517.99 | 6554.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 09:15:00 | 6591.00 | 6532.59 | 6557.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 09:30:00 | 6549.85 | 6532.59 | 6557.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 10:15:00 | 6610.00 | 6548.07 | 6562.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 10:45:00 | 6614.95 | 6548.07 | 6562.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — BUY (started 2024-08-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 12:15:00 | 6606.60 | 6573.37 | 6572.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-14 09:15:00 | 6742.50 | 6613.27 | 6591.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-14 14:15:00 | 6516.00 | 6651.27 | 6627.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 14:15:00 | 6516.00 | 6651.27 | 6627.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 14:15:00 | 6516.00 | 6651.27 | 6627.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-14 15:00:00 | 6516.00 | 6651.27 | 6627.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 15:15:00 | 6538.60 | 6628.73 | 6619.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-16 09:15:00 | 6650.00 | 6628.73 | 6619.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 15:15:00 | 6780.00 | 6806.63 | 6807.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2024-08-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 15:15:00 | 6780.00 | 6806.63 | 6807.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 09:15:00 | 6772.65 | 6799.83 | 6803.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-27 14:15:00 | 6783.90 | 6778.88 | 6790.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-27 14:15:00 | 6783.90 | 6778.88 | 6790.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 14:15:00 | 6783.90 | 6778.88 | 6790.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-27 15:00:00 | 6783.90 | 6778.88 | 6790.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 6786.00 | 6780.31 | 6789.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:15:00 | 6792.05 | 6780.31 | 6789.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 6795.15 | 6783.27 | 6790.33 | EMA400 retest candle locked (from downside) |

### Cycle 102 — BUY (started 2024-08-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-28 12:15:00 | 6804.00 | 6794.89 | 6794.48 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2024-08-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 14:15:00 | 6772.90 | 6793.71 | 6794.22 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2024-08-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 11:15:00 | 6812.00 | 6793.66 | 6793.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-29 12:15:00 | 6892.00 | 6813.32 | 6802.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-02 10:15:00 | 6908.45 | 6913.36 | 6879.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-02 11:00:00 | 6908.45 | 6913.36 | 6879.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 11:15:00 | 6886.85 | 6908.06 | 6880.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 11:30:00 | 6882.55 | 6908.06 | 6880.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 12:15:00 | 6884.45 | 6903.34 | 6880.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-02 13:00:00 | 6884.45 | 6903.34 | 6880.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-02 13:15:00 | 6895.00 | 6901.67 | 6882.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-03 09:30:00 | 6903.25 | 6898.53 | 6885.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-03 11:15:00 | 6866.50 | 6890.27 | 6883.59 | SL hit (close<static) qty=1.00 sl=6880.10 alert=retest2 |

### Cycle 105 — SELL (started 2024-09-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-03 13:15:00 | 6862.10 | 6879.28 | 6879.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-03 14:15:00 | 6838.05 | 6871.03 | 6875.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-04 12:15:00 | 6876.90 | 6855.10 | 6863.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 12:15:00 | 6876.90 | 6855.10 | 6863.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 12:15:00 | 6876.90 | 6855.10 | 6863.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 12:30:00 | 6861.25 | 6855.10 | 6863.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 13:15:00 | 6918.30 | 6867.74 | 6868.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 13:45:00 | 6917.85 | 6867.74 | 6868.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 106 — BUY (started 2024-09-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-04 14:15:00 | 6931.90 | 6880.57 | 6874.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-05 09:15:00 | 6991.25 | 6909.82 | 6889.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-05 13:15:00 | 6940.40 | 6947.99 | 6917.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-05 14:00:00 | 6940.40 | 6947.99 | 6917.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-05 14:15:00 | 6935.60 | 6945.51 | 6919.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-05 14:45:00 | 6940.00 | 6945.51 | 6919.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 09:15:00 | 6903.25 | 6934.42 | 6918.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:00:00 | 6903.25 | 6934.42 | 6918.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 6876.20 | 6922.77 | 6914.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 10:30:00 | 6896.00 | 6922.77 | 6914.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 13:15:00 | 6912.40 | 6915.57 | 6912.91 | EMA400 retest candle locked (from upside) |

### Cycle 107 — SELL (started 2024-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-09 10:15:00 | 6880.00 | 6905.58 | 6908.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-09 11:15:00 | 6850.40 | 6894.54 | 6903.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-10 09:15:00 | 6877.60 | 6871.47 | 6886.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-10 09:15:00 | 6877.60 | 6871.47 | 6886.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-10 09:15:00 | 6877.60 | 6871.47 | 6886.38 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2024-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-10 13:15:00 | 6933.40 | 6894.38 | 6893.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 09:15:00 | 6996.90 | 6923.40 | 6912.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 14:15:00 | 7010.20 | 7024.16 | 6992.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 15:00:00 | 7010.20 | 7024.16 | 6992.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 14:15:00 | 7014.65 | 7036.53 | 7025.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-17 15:00:00 | 7014.65 | 7036.53 | 7025.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-17 15:15:00 | 7015.00 | 7032.23 | 7024.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 09:30:00 | 7046.00 | 7031.67 | 7024.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-18 12:15:00 | 6995.15 | 7031.13 | 7026.77 | SL hit (close<static) qty=1.00 sl=6999.90 alert=retest2 |

### Cycle 109 — SELL (started 2024-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 13:15:00 | 6985.20 | 7021.94 | 7022.99 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2024-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-18 14:15:00 | 7038.95 | 7025.35 | 7024.44 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 15:15:00 | 7010.10 | 7022.30 | 7023.14 | EMA200 below EMA400 |

### Cycle 112 — BUY (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 09:15:00 | 7067.95 | 7031.43 | 7027.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-19 10:15:00 | 7112.50 | 7047.64 | 7034.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-19 12:15:00 | 7035.00 | 7047.89 | 7037.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 12:15:00 | 7035.00 | 7047.89 | 7037.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 12:15:00 | 7035.00 | 7047.89 | 7037.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 13:00:00 | 7035.00 | 7047.89 | 7037.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 13:15:00 | 7035.00 | 7045.31 | 7037.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 14:00:00 | 7035.00 | 7045.31 | 7037.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 7029.55 | 7042.16 | 7036.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-19 14:45:00 | 7031.95 | 7042.16 | 7036.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 15:15:00 | 7044.00 | 7042.53 | 7037.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 09:30:00 | 7039.95 | 7044.22 | 7038.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 10:15:00 | 7058.00 | 7046.98 | 7040.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 10:45:00 | 7031.05 | 7046.98 | 7040.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 14:15:00 | 7090.00 | 7077.10 | 7058.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-20 14:45:00 | 7055.15 | 7077.10 | 7058.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 7158.40 | 7145.03 | 7113.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 10:45:00 | 7194.65 | 7137.88 | 7132.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 13:30:00 | 7186.00 | 7144.58 | 7136.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-27 15:00:00 | 7231.70 | 7162.01 | 7145.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 09:30:00 | 7188.75 | 7201.46 | 7184.61 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 10:15:00 | 7186.50 | 7198.46 | 7184.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 10:30:00 | 7194.90 | 7198.46 | 7184.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 11:15:00 | 7188.05 | 7196.38 | 7185.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:15:00 | 7171.75 | 7196.38 | 7185.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 12:15:00 | 7146.30 | 7186.37 | 7181.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 12:45:00 | 7170.75 | 7186.37 | 7181.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 7153.60 | 7179.81 | 7179.01 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-01 14:15:00 | 7165.75 | 7177.00 | 7177.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2024-10-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 14:15:00 | 7165.75 | 7177.00 | 7177.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 15:15:00 | 7135.00 | 7168.60 | 7173.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 6881.30 | 6811.79 | 6861.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 10:15:00 | 6881.30 | 6811.79 | 6861.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 6881.30 | 6811.79 | 6861.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 11:00:00 | 6881.30 | 6811.79 | 6861.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 6902.00 | 6829.83 | 6865.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 12:00:00 | 6902.00 | 6829.83 | 6865.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 12:15:00 | 6894.45 | 6842.76 | 6867.78 | EMA400 retest candle locked (from downside) |

### Cycle 114 — BUY (started 2024-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 15:15:00 | 6925.00 | 6881.26 | 6881.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 09:15:00 | 7019.05 | 6908.82 | 6893.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 09:15:00 | 6997.60 | 6997.93 | 6958.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-10 09:15:00 | 6997.60 | 6997.93 | 6958.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 6997.60 | 6997.93 | 6958.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 09:45:00 | 6987.50 | 6997.93 | 6958.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 6978.95 | 6994.13 | 6959.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 10:45:00 | 6996.90 | 6994.13 | 6959.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 6915.00 | 6978.31 | 6955.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 12:00:00 | 6915.00 | 6978.31 | 6955.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 12:15:00 | 6928.40 | 6968.33 | 6953.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 12:30:00 | 6919.35 | 6968.33 | 6953.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 15:15:00 | 6971.00 | 6957.03 | 6950.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-11 09:15:00 | 6971.85 | 6957.03 | 6950.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 09:15:00 | 6981.95 | 6962.01 | 6953.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 11:15:00 | 7052.30 | 6991.21 | 6974.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 11:15:00 | 7045.90 | 7066.25 | 7068.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — SELL (started 2024-10-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 11:15:00 | 7045.90 | 7066.25 | 7068.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 12:15:00 | 7013.60 | 7055.72 | 7063.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 10:15:00 | 7025.00 | 6998.48 | 7015.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-21 10:15:00 | 7025.00 | 6998.48 | 7015.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 10:15:00 | 7025.00 | 6998.48 | 7015.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 11:00:00 | 7025.00 | 6998.48 | 7015.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 11:15:00 | 7000.00 | 6998.78 | 7013.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 11:30:00 | 7014.90 | 6998.78 | 7013.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 12:15:00 | 6987.85 | 6996.59 | 7011.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 13:15:00 | 6972.35 | 6996.59 | 7011.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 10:00:00 | 6964.55 | 6989.28 | 7003.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 09:30:00 | 6969.85 | 6943.26 | 6943.97 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-25 10:15:00 | 6950.70 | 6944.75 | 6944.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2024-10-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-25 10:15:00 | 6950.70 | 6944.75 | 6944.58 | EMA200 above EMA400 |

### Cycle 117 — SELL (started 2024-10-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-25 11:15:00 | 6897.00 | 6935.20 | 6940.25 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2024-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 13:15:00 | 6959.50 | 6932.01 | 6931.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-29 14:15:00 | 6995.55 | 6944.72 | 6937.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-30 13:15:00 | 6972.55 | 6974.20 | 6957.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-30 14:00:00 | 6972.55 | 6974.20 | 6957.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 09:15:00 | 6970.90 | 6974.65 | 6962.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 09:30:00 | 6971.60 | 6974.65 | 6962.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 10:15:00 | 6990.10 | 6977.74 | 6964.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 10:45:00 | 6982.00 | 6977.74 | 6964.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 6973.60 | 7009.34 | 6993.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 09:45:00 | 6970.90 | 7009.34 | 6993.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 6955.60 | 6998.59 | 6989.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 6955.60 | 6998.59 | 6989.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 11:15:00 | 6958.00 | 6990.47 | 6987.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:45:00 | 6923.00 | 6990.47 | 6987.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2024-11-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 12:15:00 | 6946.70 | 6981.72 | 6983.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-05 09:15:00 | 6908.45 | 6956.86 | 6970.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 13:15:00 | 6933.50 | 6910.55 | 6939.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-05 13:15:00 | 6933.50 | 6910.55 | 6939.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 13:15:00 | 6933.50 | 6910.55 | 6939.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 14:00:00 | 6933.50 | 6910.55 | 6939.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 14:15:00 | 6959.85 | 6920.41 | 6941.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 15:00:00 | 6959.85 | 6920.41 | 6941.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 15:15:00 | 6986.55 | 6933.64 | 6945.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:15:00 | 7040.75 | 6933.64 | 6945.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 6977.75 | 6946.23 | 6949.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 11:00:00 | 6977.75 | 6946.23 | 6949.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2024-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 11:15:00 | 6992.75 | 6955.54 | 6953.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-07 09:15:00 | 7417.40 | 7058.04 | 7002.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-11 09:15:00 | 7364.65 | 7393.88 | 7306.40 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-11 10:00:00 | 7364.65 | 7393.88 | 7306.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 12:15:00 | 7176.00 | 7333.95 | 7299.57 | EMA400 retest candle locked (from upside) |

### Cycle 121 — SELL (started 2024-11-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 14:15:00 | 7155.00 | 7270.76 | 7275.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-12 11:15:00 | 7140.85 | 7205.92 | 7239.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 12:15:00 | 6782.45 | 6777.32 | 6830.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-19 12:45:00 | 6781.30 | 6777.32 | 6830.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 15:15:00 | 6745.95 | 6728.45 | 6760.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 09:15:00 | 6841.70 | 6728.45 | 6760.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 09:15:00 | 6901.60 | 6763.08 | 6773.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 10:00:00 | 6901.60 | 6763.08 | 6773.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 10:15:00 | 6834.05 | 6777.28 | 6779.19 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 11:15:00 | 6888.20 | 6799.46 | 6789.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 12:15:00 | 6897.10 | 6818.99 | 6798.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 15:15:00 | 7058.60 | 7069.17 | 7016.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 09:15:00 | 7030.00 | 7069.17 | 7016.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 09:15:00 | 7032.50 | 7061.84 | 7018.09 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2024-11-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 10:15:00 | 6887.15 | 6985.34 | 6996.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 11:15:00 | 6867.80 | 6961.84 | 6984.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-02 09:15:00 | 6920.10 | 6864.80 | 6894.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-02 09:15:00 | 6920.10 | 6864.80 | 6894.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 6920.10 | 6864.80 | 6894.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:45:00 | 6936.05 | 6864.80 | 6894.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 10:15:00 | 6926.55 | 6877.15 | 6897.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-02 11:15:00 | 6912.90 | 6877.15 | 6897.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-02 12:15:00 | 7013.95 | 6908.90 | 6908.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2024-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-02 12:15:00 | 7013.95 | 6908.90 | 6908.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-02 13:15:00 | 7053.00 | 6937.72 | 6921.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-05 10:15:00 | 7188.85 | 7197.25 | 7136.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-05 10:45:00 | 7174.10 | 7197.25 | 7136.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 11:15:00 | 7237.90 | 7205.38 | 7145.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-05 11:30:00 | 7150.50 | 7205.38 | 7145.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 14:15:00 | 7232.80 | 7243.69 | 7211.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-06 14:30:00 | 7234.95 | 7243.69 | 7211.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 7195.70 | 7231.71 | 7211.61 | EMA400 retest candle locked (from upside) |

### Cycle 125 — SELL (started 2024-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-09 14:15:00 | 7189.80 | 7201.45 | 7202.36 | EMA200 below EMA400 |

### Cycle 126 — BUY (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-10 09:15:00 | 7280.00 | 7217.41 | 7209.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 10:15:00 | 7310.30 | 7259.11 | 7238.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 09:15:00 | 7244.25 | 7295.09 | 7271.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 09:15:00 | 7244.25 | 7295.09 | 7271.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 7244.25 | 7295.09 | 7271.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:45:00 | 7254.25 | 7295.09 | 7271.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 7215.25 | 7279.12 | 7266.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 11:00:00 | 7215.25 | 7279.12 | 7266.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2024-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 14:15:00 | 7224.30 | 7255.01 | 7257.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-16 10:15:00 | 7215.40 | 7242.79 | 7248.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 12:15:00 | 7253.35 | 7242.38 | 7247.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 12:15:00 | 7253.35 | 7242.38 | 7247.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 7253.35 | 7242.38 | 7247.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:00:00 | 7253.35 | 7242.38 | 7247.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 7243.60 | 7242.62 | 7246.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 14:30:00 | 7230.10 | 7245.47 | 7247.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-17 09:30:00 | 7224.35 | 7245.44 | 7247.35 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-17 10:15:00 | 7267.80 | 7249.91 | 7249.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2024-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-17 10:15:00 | 7267.80 | 7249.91 | 7249.21 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2024-12-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 13:15:00 | 7228.50 | 7249.79 | 7249.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 14:15:00 | 7220.00 | 7243.83 | 7247.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 12:15:00 | 7231.00 | 7222.78 | 7233.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 12:15:00 | 7231.00 | 7222.78 | 7233.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 12:15:00 | 7231.00 | 7222.78 | 7233.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 13:00:00 | 7231.00 | 7222.78 | 7233.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 13:15:00 | 7253.00 | 7228.82 | 7235.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 14:00:00 | 7253.00 | 7228.82 | 7235.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 14:15:00 | 7242.15 | 7231.49 | 7236.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 14:45:00 | 7248.10 | 7231.49 | 7236.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 15:15:00 | 7239.00 | 7232.99 | 7236.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 09:15:00 | 7191.00 | 7232.99 | 7236.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-19 10:00:00 | 7219.65 | 7230.32 | 7234.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-19 12:15:00 | 7257.15 | 7239.07 | 7238.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2024-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 12:15:00 | 7257.15 | 7239.07 | 7238.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 13:15:00 | 7295.35 | 7250.33 | 7243.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 14:15:00 | 7243.85 | 7305.17 | 7285.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 14:15:00 | 7243.85 | 7305.17 | 7285.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 14:15:00 | 7243.85 | 7305.17 | 7285.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 15:00:00 | 7243.85 | 7305.17 | 7285.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 15:15:00 | 7244.90 | 7293.12 | 7281.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-23 09:15:00 | 7204.65 | 7293.12 | 7281.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 7287.20 | 7278.69 | 7276.48 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2024-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 13:15:00 | 7244.65 | 7270.93 | 7273.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 14:15:00 | 7233.45 | 7249.66 | 7259.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 09:15:00 | 7259.55 | 7247.06 | 7256.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 09:15:00 | 7259.55 | 7247.06 | 7256.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 09:15:00 | 7259.55 | 7247.06 | 7256.50 | EMA400 retest candle locked (from downside) |

### Cycle 132 — BUY (started 2024-12-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 14:15:00 | 7268.20 | 7260.64 | 7260.48 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2024-12-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-26 15:15:00 | 7252.00 | 7258.91 | 7259.71 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 09:15:00 | 7294.85 | 7258.09 | 7258.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 10:15:00 | 7319.50 | 7270.37 | 7263.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 7267.40 | 7292.85 | 7281.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 09:15:00 | 7267.40 | 7292.85 | 7281.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 7267.40 | 7292.85 | 7281.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 11:45:00 | 7343.75 | 7308.52 | 7290.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 09:15:00 | 7386.70 | 7310.77 | 7297.77 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-03 12:30:00 | 7346.45 | 7372.95 | 7369.56 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-03 13:15:00 | 7321.75 | 7362.71 | 7365.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2025-01-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 13:15:00 | 7321.75 | 7362.71 | 7365.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 14:15:00 | 7295.00 | 7349.17 | 7358.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-06 10:15:00 | 7342.65 | 7342.10 | 7352.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-06 10:15:00 | 7342.65 | 7342.10 | 7352.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 7342.65 | 7342.10 | 7352.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 7342.65 | 7342.10 | 7352.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 7398.90 | 7353.46 | 7356.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:30:00 | 7370.50 | 7353.46 | 7356.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — BUY (started 2025-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-06 12:15:00 | 7397.85 | 7362.34 | 7360.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-06 14:15:00 | 7430.00 | 7377.65 | 7367.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-07 09:15:00 | 7378.15 | 7387.73 | 7374.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-07 09:15:00 | 7378.15 | 7387.73 | 7374.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 7378.15 | 7387.73 | 7374.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:45:00 | 7355.50 | 7387.73 | 7374.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 10:15:00 | 7438.70 | 7397.92 | 7380.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-07 11:15:00 | 7498.85 | 7397.92 | 7380.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-08 09:15:00 | 7358.25 | 7419.08 | 7404.61 | SL hit (close<static) qty=1.00 sl=7368.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 11:15:00 | 7308.25 | 7388.15 | 7392.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-08 12:15:00 | 7225.00 | 7355.52 | 7377.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 13:15:00 | 6730.15 | 6720.85 | 6799.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 14:00:00 | 6730.15 | 6720.85 | 6799.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 6729.25 | 6720.20 | 6779.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 09:45:00 | 6779.65 | 6720.20 | 6779.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 13:15:00 | 6796.45 | 6738.69 | 6769.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 14:00:00 | 6796.45 | 6738.69 | 6769.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 14:15:00 | 6782.10 | 6747.37 | 6770.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 15:00:00 | 6782.10 | 6747.37 | 6770.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 15:15:00 | 6800.95 | 6758.09 | 6773.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 09:15:00 | 6798.60 | 6758.09 | 6773.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 6784.95 | 6771.13 | 6777.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:30:00 | 6790.70 | 6771.13 | 6777.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — BUY (started 2025-01-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 11:15:00 | 6823.65 | 6781.64 | 6781.52 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2025-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 15:15:00 | 6767.00 | 6786.04 | 6787.68 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 09:15:00 | 6980.80 | 6824.99 | 6805.24 | EMA200 above EMA400 |

### Cycle 141 — SELL (started 2025-01-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 10:15:00 | 6830.70 | 6875.38 | 6879.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-24 14:15:00 | 6741.20 | 6816.73 | 6848.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 6730.00 | 6714.26 | 6751.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 12:45:00 | 6722.30 | 6714.26 | 6751.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 09:15:00 | 6745.00 | 6722.02 | 6743.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:00:00 | 6745.00 | 6722.02 | 6743.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 10:15:00 | 6753.95 | 6728.41 | 6744.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 10:45:00 | 6759.95 | 6728.41 | 6744.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 11:15:00 | 6792.90 | 6741.31 | 6748.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-29 12:00:00 | 6792.90 | 6741.31 | 6748.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-29 12:15:00 | 6765.50 | 6746.14 | 6750.46 | EMA400 retest candle locked (from downside) |

### Cycle 142 — BUY (started 2025-01-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 14:15:00 | 6785.60 | 6757.63 | 6755.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 6832.00 | 6777.04 | 6764.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 6814.45 | 6814.83 | 6789.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 13:30:00 | 6815.85 | 6814.83 | 6789.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 6828.25 | 6824.92 | 6801.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 11:30:00 | 6933.90 | 6852.73 | 6830.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 13:15:00 | 6891.05 | 6849.20 | 6830.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 13:45:00 | 6903.95 | 6859.16 | 6837.12 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 11:15:00 | 6776.15 | 6822.73 | 6826.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 6776.15 | 6822.73 | 6826.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 09:15:00 | 6709.80 | 6783.05 | 6804.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-04 14:15:00 | 6780.00 | 6771.17 | 6788.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-04 15:00:00 | 6780.00 | 6771.17 | 6788.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-04 15:15:00 | 6786.00 | 6774.14 | 6788.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:15:00 | 6885.95 | 6774.14 | 6788.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-05 09:15:00 | 6876.25 | 6794.56 | 6796.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-05 09:30:00 | 6896.60 | 6794.56 | 6796.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 144 — BUY (started 2025-02-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 10:15:00 | 6904.15 | 6816.48 | 6806.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-05 11:15:00 | 6954.25 | 6844.03 | 6819.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 09:15:00 | 6876.55 | 6907.74 | 6867.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-06 09:15:00 | 6876.55 | 6907.74 | 6867.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 6876.55 | 6907.74 | 6867.30 | EMA400 retest candle locked (from upside) |

### Cycle 145 — SELL (started 2025-02-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-07 11:15:00 | 6795.80 | 6852.91 | 6857.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 12:15:00 | 6770.00 | 6836.33 | 6849.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-10 14:15:00 | 6758.90 | 6748.25 | 6786.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-10 14:45:00 | 6759.25 | 6748.25 | 6786.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-10 15:15:00 | 6792.95 | 6757.19 | 6787.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-11 09:15:00 | 6482.50 | 6757.19 | 6787.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-11 09:15:00 | 6512.25 | 6708.20 | 6762.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-11 10:45:00 | 6389.25 | 6644.89 | 6728.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-12 12:30:00 | 6456.65 | 6423.38 | 6524.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-18 12:15:00 | 6355.95 | 6326.59 | 6325.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — BUY (started 2025-02-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-18 12:15:00 | 6355.95 | 6326.59 | 6325.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 09:15:00 | 6414.50 | 6360.32 | 6343.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-20 11:15:00 | 6407.05 | 6416.95 | 6389.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-20 12:00:00 | 6407.05 | 6416.95 | 6389.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 14:15:00 | 6396.65 | 6408.68 | 6392.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-20 15:00:00 | 6396.65 | 6408.68 | 6392.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 15:15:00 | 6394.00 | 6405.75 | 6392.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 09:15:00 | 6348.45 | 6405.75 | 6392.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 6318.70 | 6388.34 | 6385.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 09:45:00 | 6310.95 | 6388.34 | 6385.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 147 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 6335.30 | 6377.73 | 6381.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 15:15:00 | 6303.10 | 6338.71 | 6358.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 15:15:00 | 6085.00 | 6084.04 | 6145.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-03 09:15:00 | 6111.00 | 6084.04 | 6145.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 6131.85 | 6093.60 | 6144.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-03 09:30:00 | 6134.65 | 6093.60 | 6144.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 10:15:00 | 6127.00 | 6100.28 | 6142.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-03 11:15:00 | 6114.20 | 6100.28 | 6142.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-03 15:15:00 | 6182.00 | 6146.19 | 6151.59 | SL hit (close>static) qty=1.00 sl=6175.00 alert=retest2 |

### Cycle 148 — BUY (started 2025-03-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 10:15:00 | 6192.10 | 6162.33 | 6158.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 11:15:00 | 6227.30 | 6188.20 | 6176.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 11:15:00 | 6265.35 | 6267.02 | 6241.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:00:00 | 6265.35 | 6267.02 | 6241.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 13:15:00 | 6250.00 | 6262.70 | 6244.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 14:00:00 | 6250.00 | 6262.70 | 6244.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 14:15:00 | 6207.85 | 6251.73 | 6240.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 15:00:00 | 6207.85 | 6251.73 | 6240.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 6207.80 | 6242.94 | 6237.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 6160.00 | 6242.94 | 6237.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 149 — SELL (started 2025-03-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 09:15:00 | 6181.50 | 6230.66 | 6232.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 6149.75 | 6184.57 | 6206.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 14:15:00 | 6128.85 | 6110.71 | 6150.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-11 15:00:00 | 6128.85 | 6110.71 | 6150.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 6134.10 | 6115.39 | 6149.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-12 09:15:00 | 6086.85 | 6115.39 | 6149.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 09:15:00 | 6113.90 | 6118.31 | 6132.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-13 11:15:00 | 6119.45 | 6125.01 | 6132.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-17 11:15:00 | 6160.70 | 6128.16 | 6127.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 150 — BUY (started 2025-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 11:15:00 | 6160.70 | 6128.16 | 6127.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 09:15:00 | 6252.60 | 6165.89 | 6146.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 11:15:00 | 6367.05 | 6372.60 | 6312.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-20 11:45:00 | 6371.55 | 6372.60 | 6312.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 09:15:00 | 6627.70 | 6623.91 | 6564.73 | EMA400 retest candle locked (from upside) |

### Cycle 151 — SELL (started 2025-03-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 14:15:00 | 6519.15 | 6587.69 | 6589.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 10:15:00 | 6487.45 | 6554.62 | 6573.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 10:15:00 | 6537.05 | 6507.69 | 6534.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 10:15:00 | 6537.05 | 6507.69 | 6534.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 10:15:00 | 6537.05 | 6507.69 | 6534.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 11:00:00 | 6537.05 | 6507.69 | 6534.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 11:15:00 | 6553.50 | 6516.85 | 6536.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 12:00:00 | 6553.50 | 6516.85 | 6536.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 12:15:00 | 6555.50 | 6524.58 | 6538.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 12:45:00 | 6570.30 | 6524.58 | 6538.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 13:15:00 | 6575.00 | 6534.66 | 6541.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-28 13:30:00 | 6577.00 | 6534.66 | 6541.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2025-03-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 14:15:00 | 6633.05 | 6554.34 | 6549.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-02 09:15:00 | 6719.30 | 6633.75 | 6601.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-03 13:15:00 | 6723.40 | 6723.53 | 6685.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-03 14:00:00 | 6723.40 | 6723.53 | 6685.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 14:15:00 | 6670.30 | 6712.88 | 6684.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-03 15:00:00 | 6670.30 | 6712.88 | 6684.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 15:15:00 | 6670.05 | 6704.32 | 6683.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:15:00 | 6644.40 | 6704.32 | 6683.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 09:15:00 | 6593.10 | 6682.07 | 6674.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-04 09:45:00 | 6606.00 | 6682.07 | 6674.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-04 10:15:00 | 6673.65 | 6680.39 | 6674.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 14:45:00 | 6709.15 | 6686.18 | 6679.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 10:15:00 | 6711.35 | 6692.17 | 6683.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 12:15:00 | 6648.10 | 6677.37 | 6678.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 153 — SELL (started 2025-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 12:15:00 | 6648.10 | 6677.37 | 6678.44 | EMA200 below EMA400 |

### Cycle 154 — BUY (started 2025-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-07 15:15:00 | 6714.50 | 6685.62 | 6681.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 09:15:00 | 6772.20 | 6702.94 | 6690.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-11 11:15:00 | 6797.90 | 6812.32 | 6783.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-11 11:15:00 | 6797.90 | 6812.32 | 6783.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 11:15:00 | 6797.90 | 6812.32 | 6783.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-11 12:00:00 | 6797.90 | 6812.32 | 6783.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 14:15:00 | 6776.95 | 6800.93 | 6785.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-11 15:00:00 | 6776.95 | 6800.93 | 6785.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 15:15:00 | 6798.00 | 6800.35 | 6786.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 09:15:00 | 6870.00 | 6800.35 | 6786.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 7014.00 | 7105.23 | 7106.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 155 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 7014.00 | 7105.23 | 7106.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 6951.00 | 7074.38 | 7092.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 10:15:00 | 6997.50 | 6996.63 | 7034.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 11:00:00 | 6997.50 | 6996.63 | 7034.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 12:15:00 | 7029.00 | 7002.76 | 7030.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 12:45:00 | 7039.00 | 7002.76 | 7030.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 13:15:00 | 7023.00 | 7006.81 | 7029.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 10:15:00 | 7003.00 | 7021.10 | 7031.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-02 09:15:00 | 7067.50 | 7000.90 | 7005.27 | SL hit (close>static) qty=1.00 sl=7039.50 alert=retest2 |

### Cycle 156 — BUY (started 2025-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 09:15:00 | 7045.00 | 6992.80 | 6989.70 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2025-05-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 15:15:00 | 6970.00 | 7001.42 | 7002.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 09:15:00 | 6915.00 | 6984.13 | 6994.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 6911.50 | 6786.65 | 6840.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 6911.50 | 6786.65 | 6840.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 6911.50 | 6786.65 | 6840.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:45:00 | 6907.00 | 6786.65 | 6840.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 6926.00 | 6814.52 | 6848.13 | EMA400 retest candle locked (from downside) |

### Cycle 158 — BUY (started 2025-05-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 14:15:00 | 6927.00 | 6876.31 | 6870.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-13 09:15:00 | 6995.50 | 6908.58 | 6886.80 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 12:15:00 | 6929.50 | 6930.00 | 6903.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:00:00 | 6929.50 | 6930.00 | 6903.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 6926.00 | 6925.84 | 6907.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 09:15:00 | 6927.00 | 6925.84 | 6907.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 6967.50 | 6934.17 | 6913.37 | EMA400 retest candle locked (from upside) |

### Cycle 159 — SELL (started 2025-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 10:15:00 | 6876.00 | 6908.77 | 6910.65 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2025-05-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 12:15:00 | 6940.00 | 6911.46 | 6911.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-15 13:15:00 | 7013.50 | 6931.86 | 6920.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-16 15:15:00 | 7000.00 | 7013.02 | 6984.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-19 09:15:00 | 7011.50 | 7013.02 | 6984.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 7051.50 | 7020.72 | 6991.03 | EMA400 retest candle locked (from upside) |

### Cycle 161 — SELL (started 2025-05-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 09:15:00 | 6930.50 | 6980.27 | 6983.19 | EMA200 below EMA400 |

### Cycle 162 — BUY (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 14:15:00 | 6979.50 | 6975.06 | 6974.61 | EMA200 above EMA400 |

### Cycle 163 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 6933.50 | 6968.50 | 6971.81 | EMA200 below EMA400 |

### Cycle 164 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 7029.00 | 6971.17 | 6968.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 10:15:00 | 7043.00 | 6985.54 | 6975.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 7054.00 | 7079.29 | 7053.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 7054.00 | 7079.29 | 7053.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 7054.00 | 7079.29 | 7053.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:45:00 | 7037.50 | 7079.29 | 7053.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 7069.50 | 7077.33 | 7054.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 7087.00 | 7077.33 | 7054.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 14:45:00 | 7073.50 | 7077.37 | 7062.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 7000.50 | 7057.62 | 7055.63 | SL hit (close<static) qty=1.00 sl=7043.50 alert=retest2 |

### Cycle 165 — SELL (started 2025-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 10:15:00 | 6986.50 | 7043.39 | 7049.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 14:15:00 | 6956.00 | 6998.52 | 7023.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 6947.00 | 6938.61 | 6968.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 6947.00 | 6938.61 | 6968.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 6947.00 | 6938.61 | 6968.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 13:00:00 | 6931.50 | 6942.39 | 6963.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 14:00:00 | 6932.00 | 6940.31 | 6960.44 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 14:30:00 | 6928.00 | 6928.05 | 6953.04 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 7065.00 | 6948.71 | 6957.67 | SL hit (close>static) qty=1.00 sl=6988.50 alert=retest2 |

### Cycle 166 — BUY (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 13:15:00 | 6880.00 | 6870.30 | 6869.28 | EMA200 above EMA400 |

### Cycle 167 — SELL (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 09:15:00 | 6841.00 | 6868.03 | 6868.72 | EMA200 below EMA400 |

### Cycle 168 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 6880.00 | 6869.28 | 6869.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 13:15:00 | 6930.00 | 6881.42 | 6874.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-09 09:15:00 | 6897.50 | 6901.77 | 6887.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-09 09:45:00 | 6915.00 | 6901.77 | 6887.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 6916.50 | 6904.71 | 6889.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-09 14:15:00 | 6921.50 | 6907.09 | 6894.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-10 13:45:00 | 6924.50 | 6934.34 | 6918.58 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 09:15:00 | 6878.50 | 6918.15 | 6914.68 | SL hit (close<static) qty=1.00 sl=6886.00 alert=retest2 |

### Cycle 169 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 6899.50 | 6912.06 | 6912.73 | EMA200 below EMA400 |

### Cycle 170 — BUY (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 14:15:00 | 6926.50 | 6914.95 | 6913.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 15:15:00 | 6932.00 | 6918.36 | 6915.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 13:15:00 | 6985.50 | 6996.54 | 6975.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-13 14:00:00 | 6985.50 | 6996.54 | 6975.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 11:15:00 | 7037.50 | 7066.60 | 7039.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:45:00 | 7039.00 | 7066.60 | 7039.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 12:15:00 | 7049.00 | 7063.08 | 7040.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 12:30:00 | 7045.50 | 7063.08 | 7040.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 13:15:00 | 7044.50 | 7059.37 | 7040.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 14:15:00 | 7036.00 | 7059.37 | 7040.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 14:15:00 | 7001.50 | 7047.79 | 7036.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 15:00:00 | 7001.50 | 7047.79 | 7036.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 15:15:00 | 7019.00 | 7042.03 | 7035.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:15:00 | 7011.00 | 7042.03 | 7035.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 171 — SELL (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 10:15:00 | 6990.00 | 7028.58 | 7030.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-18 11:15:00 | 6946.50 | 7012.17 | 7022.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-19 09:15:00 | 7003.50 | 6975.57 | 6996.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 09:15:00 | 7003.50 | 6975.57 | 6996.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 7003.50 | 6975.57 | 6996.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:45:00 | 7009.50 | 6975.57 | 6996.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 6991.00 | 6978.66 | 6995.99 | EMA400 retest candle locked (from downside) |

### Cycle 172 — BUY (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 11:15:00 | 7052.00 | 7010.20 | 7004.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 13:15:00 | 7052.50 | 7022.07 | 7011.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 7003.00 | 7029.08 | 7018.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 7003.00 | 7029.08 | 7018.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 7003.00 | 7029.08 | 7018.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 09:30:00 | 7014.50 | 7029.08 | 7018.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 7034.00 | 7030.07 | 7019.68 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:15:00 | 7042.00 | 7030.07 | 7019.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 10:00:00 | 7046.00 | 7026.63 | 7021.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 15:15:00 | 7000.00 | 7021.19 | 7022.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 173 — SELL (started 2025-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 15:15:00 | 7000.00 | 7021.19 | 7022.70 | EMA200 below EMA400 |

### Cycle 174 — BUY (started 2025-06-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 13:15:00 | 7054.50 | 7024.85 | 7023.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-26 09:15:00 | 7057.00 | 7034.94 | 7028.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-26 11:15:00 | 7031.00 | 7036.56 | 7030.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 11:15:00 | 7031.00 | 7036.56 | 7030.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 11:15:00 | 7031.00 | 7036.56 | 7030.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:00:00 | 7031.00 | 7036.56 | 7030.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 7041.00 | 7037.45 | 7031.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 13:30:00 | 7058.00 | 7048.46 | 7037.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 11:15:00 | 7498.00 | 7548.02 | 7554.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 175 — SELL (started 2025-07-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 11:15:00 | 7498.00 | 7548.02 | 7554.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 12:15:00 | 7481.00 | 7534.61 | 7547.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 11:15:00 | 7236.00 | 7235.01 | 7306.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 12:00:00 | 7236.00 | 7235.01 | 7306.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 7294.50 | 7248.60 | 7277.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:00:00 | 7294.50 | 7248.60 | 7277.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 7314.00 | 7261.68 | 7281.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 7314.00 | 7261.68 | 7281.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 7356.00 | 7280.54 | 7287.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:45:00 | 7362.00 | 7280.54 | 7287.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 176 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 7366.00 | 7297.64 | 7295.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 7396.50 | 7325.79 | 7308.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 13:15:00 | 7343.00 | 7353.84 | 7330.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 13:15:00 | 7343.00 | 7353.84 | 7330.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 7343.00 | 7353.84 | 7330.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:30:00 | 7399.50 | 7359.76 | 7338.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 10:30:00 | 7397.00 | 7368.01 | 7344.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 11:00:00 | 7401.00 | 7368.01 | 7344.34 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 7276.50 | 7352.10 | 7348.71 | SL hit (close<static) qty=1.00 sl=7310.00 alert=retest2 |

### Cycle 177 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 7304.00 | 7342.48 | 7344.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 13:15:00 | 7253.00 | 7291.63 | 7312.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 7319.00 | 7266.89 | 7279.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 7319.00 | 7266.89 | 7279.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 7319.00 | 7266.89 | 7279.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:45:00 | 7322.50 | 7266.89 | 7279.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 7317.50 | 7277.01 | 7283.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:30:00 | 7327.00 | 7277.01 | 7283.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 178 — BUY (started 2025-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 11:15:00 | 7329.00 | 7287.41 | 7287.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 13:15:00 | 7344.00 | 7303.06 | 7294.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 14:15:00 | 7371.50 | 7393.78 | 7358.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 14:45:00 | 7372.00 | 7393.78 | 7358.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 15:15:00 | 7366.00 | 7388.22 | 7359.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-25 09:15:00 | 7413.50 | 7388.22 | 7359.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-25 09:15:00 | 7417.00 | 7393.98 | 7364.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 11:15:00 | 7435.00 | 7400.88 | 7370.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 12:30:00 | 7435.00 | 7411.17 | 7380.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-25 13:15:00 | 7448.00 | 7411.17 | 7380.49 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-28 09:45:00 | 7443.50 | 7439.18 | 7405.31 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 7367.00 | 7424.24 | 7404.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-28 12:00:00 | 7367.00 | 7424.24 | 7404.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 7340.00 | 7407.39 | 7398.51 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-28 12:15:00 | 7340.00 | 7407.39 | 7398.51 | SL hit (close<static) qty=1.00 sl=7343.50 alert=retest2 |

### Cycle 179 — SELL (started 2025-07-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-28 14:15:00 | 7357.00 | 7387.57 | 7390.44 | EMA200 below EMA400 |

### Cycle 180 — BUY (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 10:15:00 | 7407.50 | 7392.50 | 7391.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 12:15:00 | 7427.00 | 7402.84 | 7396.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 10:15:00 | 7432.00 | 7432.58 | 7416.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 10:15:00 | 7432.00 | 7432.58 | 7416.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 7432.00 | 7432.58 | 7416.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:45:00 | 7427.50 | 7432.58 | 7416.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 7417.00 | 7429.85 | 7418.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 12:30:00 | 7414.50 | 7429.85 | 7418.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 7450.00 | 7433.88 | 7421.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 13:30:00 | 7423.50 | 7433.88 | 7421.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 15:15:00 | 7448.00 | 7439.52 | 7426.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:15:00 | 7424.00 | 7439.52 | 7426.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 7396.00 | 7430.82 | 7423.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-31 13:00:00 | 7445.00 | 7427.43 | 7423.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 12:15:00 | 7389.00 | 7426.40 | 7429.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 181 — SELL (started 2025-08-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 12:15:00 | 7389.00 | 7426.40 | 7429.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 14:15:00 | 7334.00 | 7399.30 | 7416.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 7169.50 | 7144.14 | 7197.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 7169.50 | 7144.14 | 7197.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 7133.50 | 7149.11 | 7190.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:30:00 | 7121.50 | 7143.49 | 7184.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 12:00:00 | 7118.50 | 7138.49 | 7178.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 13:15:00 | 7215.00 | 7134.07 | 7146.53 | SL hit (close>static) qty=1.00 sl=7198.50 alert=retest2 |

### Cycle 182 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 7259.50 | 7159.16 | 7156.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 09:15:00 | 7680.00 | 7321.51 | 7247.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 14:15:00 | 7819.50 | 7836.24 | 7731.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 15:00:00 | 7819.50 | 7836.24 | 7731.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 7922.00 | 7913.07 | 7879.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 11:15:00 | 7927.00 | 7913.26 | 7882.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 14:30:00 | 7931.00 | 7921.89 | 7897.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 15:15:00 | 7930.00 | 7921.89 | 7897.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-25 10:15:00 | 7849.50 | 7899.75 | 7893.23 | SL hit (close<static) qty=1.00 sl=7852.00 alert=retest2 |

### Cycle 183 — SELL (started 2025-08-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 12:15:00 | 7854.00 | 7885.44 | 7887.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 15:15:00 | 7848.50 | 7868.04 | 7878.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 09:15:00 | 7791.00 | 7786.13 | 7819.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-28 09:15:00 | 7791.00 | 7786.13 | 7819.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 7791.00 | 7786.13 | 7819.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 10:00:00 | 7791.00 | 7786.13 | 7819.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 10:15:00 | 7822.00 | 7793.31 | 7819.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-28 11:00:00 | 7822.00 | 7793.31 | 7819.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 11:15:00 | 7798.50 | 7794.35 | 7817.99 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 12:30:00 | 7785.50 | 7789.38 | 7813.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-03 10:15:00 | 7739.50 | 7695.68 | 7691.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 184 — BUY (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 10:15:00 | 7739.50 | 7695.68 | 7691.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 13:15:00 | 7762.00 | 7717.20 | 7703.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 11:15:00 | 7866.00 | 7867.73 | 7816.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-05 11:45:00 | 7872.00 | 7867.73 | 7816.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 7846.00 | 7860.23 | 7821.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:30:00 | 7824.00 | 7860.23 | 7821.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 7800.50 | 7848.28 | 7819.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 15:00:00 | 7800.50 | 7848.28 | 7819.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 7829.50 | 7844.53 | 7820.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 7811.50 | 7844.53 | 7820.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 7797.50 | 7835.12 | 7818.54 | EMA400 retest candle locked (from upside) |

### Cycle 185 — SELL (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 12:15:00 | 7786.50 | 7806.47 | 7807.84 | EMA200 below EMA400 |

### Cycle 186 — BUY (started 2025-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 11:15:00 | 7838.00 | 7810.04 | 7808.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 7859.00 | 7826.80 | 7817.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 11:15:00 | 7879.50 | 7885.07 | 7861.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 11:15:00 | 7879.50 | 7885.07 | 7861.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 7879.50 | 7885.07 | 7861.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:45:00 | 7866.00 | 7885.07 | 7861.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 7869.50 | 7879.58 | 7867.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:45:00 | 7862.00 | 7879.58 | 7867.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 7842.50 | 7872.16 | 7864.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:00:00 | 7842.50 | 7872.16 | 7864.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 7870.50 | 7871.83 | 7865.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 12:15:00 | 7881.00 | 7871.83 | 7865.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 14:15:00 | 7880.00 | 7874.65 | 7867.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 7827.00 | 7863.54 | 7864.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 187 — SELL (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 09:15:00 | 7827.00 | 7863.54 | 7864.43 | EMA200 below EMA400 |

### Cycle 188 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 7899.00 | 7847.54 | 7844.72 | EMA200 above EMA400 |

### Cycle 189 — SELL (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 11:15:00 | 7834.00 | 7853.04 | 7853.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 7825.00 | 7843.07 | 7848.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 09:15:00 | 7702.50 | 7700.63 | 7734.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 10:00:00 | 7702.50 | 7700.63 | 7734.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 15:15:00 | 7450.50 | 7436.44 | 7463.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:15:00 | 7416.50 | 7436.44 | 7463.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 7422.00 | 7433.55 | 7459.73 | EMA400 retest candle locked (from downside) |

### Cycle 190 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 7591.50 | 7473.80 | 7467.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 11:15:00 | 7740.00 | 7695.37 | 7661.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 14:15:00 | 7697.50 | 7701.58 | 7673.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-09 15:00:00 | 7697.50 | 7701.58 | 7673.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 7682.00 | 7697.05 | 7678.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:15:00 | 7698.50 | 7697.05 | 7678.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 11:15:00 | 7718.00 | 7701.24 | 7681.97 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-13 09:15:00 | 7721.50 | 7691.17 | 7682.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 11:15:00 | 7652.50 | 7682.19 | 7680.61 | SL hit (close<static) qty=1.00 sl=7670.50 alert=retest2 |

### Cycle 191 — SELL (started 2025-10-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 12:15:00 | 7665.00 | 7678.75 | 7679.20 | EMA200 below EMA400 |

### Cycle 192 — BUY (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-14 09:15:00 | 7718.50 | 7685.26 | 7681.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 7859.50 | 7813.34 | 7770.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 14:15:00 | 7902.00 | 7906.84 | 7859.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 14:30:00 | 7918.00 | 7906.84 | 7859.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 8071.00 | 8011.90 | 7958.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 10:15:00 | 8095.50 | 8011.90 | 7958.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 09:15:00 | 7900.00 | 7978.33 | 7969.79 | SL hit (close<static) qty=1.00 sl=7920.00 alert=retest2 |

### Cycle 193 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 7872.00 | 7957.06 | 7960.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 7853.50 | 7911.95 | 7937.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 7883.50 | 7883.36 | 7915.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-27 09:45:00 | 7891.50 | 7883.36 | 7915.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 7848.50 | 7860.71 | 7886.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 09:30:00 | 7900.00 | 7860.71 | 7886.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 7841.50 | 7850.73 | 7875.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:45:00 | 7866.50 | 7850.73 | 7875.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 7878.00 | 7856.54 | 7873.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 7878.00 | 7856.54 | 7873.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 7861.50 | 7857.54 | 7872.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 09:15:00 | 7864.00 | 7857.54 | 7872.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 7863.50 | 7858.73 | 7871.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 09:15:00 | 7811.50 | 7859.68 | 7867.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 15:15:00 | 7835.00 | 7787.03 | 7782.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 194 — BUY (started 2025-11-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 15:15:00 | 7835.00 | 7787.03 | 7782.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 09:15:00 | 7870.50 | 7803.72 | 7790.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-04 13:15:00 | 7820.50 | 7825.12 | 7806.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-04 14:00:00 | 7820.50 | 7825.12 | 7806.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 14:15:00 | 7824.00 | 7824.89 | 7808.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 14:45:00 | 7784.00 | 7824.89 | 7808.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 7790.00 | 7817.91 | 7806.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:15:00 | 7733.50 | 7817.91 | 7806.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 7768.00 | 7807.93 | 7803.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 11:15:00 | 7799.50 | 7804.95 | 7802.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-06 14:00:00 | 7815.50 | 7812.19 | 7806.62 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 15:15:00 | 7783.00 | 7801.68 | 7802.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 195 — SELL (started 2025-11-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 15:15:00 | 7783.00 | 7801.68 | 7802.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 09:15:00 | 7732.50 | 7787.85 | 7796.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 15:15:00 | 7520.00 | 7505.05 | 7566.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-12 09:15:00 | 7467.00 | 7505.05 | 7566.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 7454.00 | 7431.52 | 7457.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:45:00 | 7488.00 | 7431.52 | 7457.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 7446.00 | 7434.42 | 7456.15 | EMA400 retest candle locked (from downside) |

### Cycle 196 — BUY (started 2025-11-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 15:15:00 | 7492.50 | 7468.16 | 7466.17 | EMA200 above EMA400 |

### Cycle 197 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 7413.50 | 7457.23 | 7461.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 7382.00 | 7427.40 | 7443.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 10:15:00 | 7408.00 | 7404.32 | 7427.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 11:00:00 | 7408.00 | 7404.32 | 7427.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 12:15:00 | 7456.00 | 7417.88 | 7429.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:00:00 | 7456.00 | 7417.88 | 7429.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 13:15:00 | 7466.00 | 7427.51 | 7433.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 13:30:00 | 7462.50 | 7427.51 | 7433.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 198 — BUY (started 2025-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 15:15:00 | 7457.00 | 7438.76 | 7437.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 10:15:00 | 7471.00 | 7447.89 | 7442.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 14:15:00 | 7426.50 | 7451.31 | 7446.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 14:15:00 | 7426.50 | 7451.31 | 7446.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 14:15:00 | 7426.50 | 7451.31 | 7446.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 15:00:00 | 7426.50 | 7451.31 | 7446.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 15:15:00 | 7440.00 | 7449.05 | 7445.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:15:00 | 7417.00 | 7449.05 | 7445.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 199 — SELL (started 2025-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 09:15:00 | 7410.00 | 7441.24 | 7442.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 15:15:00 | 7385.00 | 7413.67 | 7426.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 09:15:00 | 7416.00 | 7414.13 | 7425.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 10:00:00 | 7416.00 | 7414.13 | 7425.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 7405.00 | 7382.61 | 7401.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 14:45:00 | 7348.00 | 7367.86 | 7386.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 10:00:00 | 7345.00 | 7354.91 | 7377.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 11:45:00 | 7350.50 | 7371.19 | 7376.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 09:30:00 | 7342.50 | 7337.97 | 7346.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 7347.00 | 7339.77 | 7346.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 12:00:00 | 7335.50 | 7338.92 | 7345.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 12:30:00 | 7335.50 | 7337.24 | 7344.39 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 12:15:00 | 6980.60 | 7023.49 | 7061.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 12:15:00 | 6977.75 | 7023.49 | 7061.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 12:15:00 | 6982.97 | 7023.49 | 7061.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-11 13:15:00 | 6975.38 | 7022.19 | 7057.32 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-12 09:15:00 | 7050.00 | 7024.13 | 7049.12 | SL hit (close>ema200) qty=0.50 sl=7024.13 alert=retest2 |

### Cycle 200 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 7103.00 | 7061.99 | 7061.17 | EMA200 above EMA400 |

### Cycle 201 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 7021.00 | 7061.99 | 7062.28 | EMA200 below EMA400 |

### Cycle 202 — BUY (started 2025-12-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 12:15:00 | 7065.00 | 7062.47 | 7062.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 13:15:00 | 7076.50 | 7065.27 | 7063.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 12:15:00 | 7083.50 | 7090.90 | 7079.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 12:15:00 | 7083.50 | 7090.90 | 7079.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 12:15:00 | 7083.50 | 7090.90 | 7079.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 13:00:00 | 7083.50 | 7090.90 | 7079.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 13:15:00 | 7045.00 | 7081.72 | 7076.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:00:00 | 7045.00 | 7081.72 | 7076.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 14:15:00 | 7046.50 | 7074.68 | 7073.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-16 14:30:00 | 7039.50 | 7074.68 | 7073.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 203 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 7045.00 | 7068.74 | 7071.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 7010.00 | 7056.99 | 7065.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 09:15:00 | 6963.00 | 6937.61 | 6968.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 6963.00 | 6937.61 | 6968.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 6963.00 | 6937.61 | 6968.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 6963.00 | 6937.61 | 6968.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 6967.50 | 6943.58 | 6967.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 6967.50 | 6943.58 | 6967.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 6985.00 | 6951.87 | 6969.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 12:00:00 | 6985.00 | 6951.87 | 6969.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 6978.00 | 6957.09 | 6970.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:15:00 | 6998.50 | 6957.09 | 6970.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 6992.00 | 6964.08 | 6972.26 | EMA400 retest candle locked (from downside) |

### Cycle 204 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 7022.00 | 6983.97 | 6980.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 7085.00 | 7004.17 | 6989.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-26 13:15:00 | 7164.50 | 7170.60 | 7137.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-26 13:45:00 | 7159.00 | 7170.60 | 7137.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 7156.50 | 7167.78 | 7139.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:30:00 | 7139.50 | 7167.78 | 7139.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 7135.50 | 7160.00 | 7140.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 7135.50 | 7160.00 | 7140.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 7130.00 | 7154.00 | 7139.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:00:00 | 7130.00 | 7154.00 | 7139.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 7130.00 | 7149.20 | 7138.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:30:00 | 7117.00 | 7149.20 | 7138.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 7096.00 | 7138.56 | 7135.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 13:00:00 | 7096.00 | 7138.56 | 7135.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 205 — SELL (started 2025-12-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 13:15:00 | 7080.00 | 7126.85 | 7130.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 09:15:00 | 7030.50 | 7094.73 | 7113.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 10:15:00 | 7011.00 | 7006.35 | 7045.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 10:30:00 | 7008.50 | 7006.35 | 7045.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 13:15:00 | 7040.00 | 7017.47 | 7041.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 13:45:00 | 7041.50 | 7017.47 | 7041.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 14:15:00 | 7045.50 | 7023.08 | 7041.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 15:00:00 | 7045.50 | 7023.08 | 7041.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 15:15:00 | 7032.50 | 7024.96 | 7040.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:15:00 | 7057.50 | 7024.96 | 7040.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 7047.50 | 7029.47 | 7041.45 | EMA400 retest candle locked (from downside) |

### Cycle 206 — BUY (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 12:15:00 | 7083.50 | 7054.51 | 7051.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 13:15:00 | 7087.00 | 7061.01 | 7054.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 13:15:00 | 7092.50 | 7096.71 | 7080.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-02 14:00:00 | 7092.50 | 7096.71 | 7080.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 7124.50 | 7109.86 | 7091.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 09:30:00 | 7069.00 | 7109.86 | 7091.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 7091.50 | 7109.35 | 7097.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:00:00 | 7091.50 | 7109.35 | 7097.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 14:15:00 | 7088.00 | 7105.08 | 7096.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 14:30:00 | 7079.50 | 7105.08 | 7096.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 11:15:00 | 7372.00 | 7383.17 | 7326.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 12:15:00 | 7396.50 | 7383.17 | 7326.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 13:30:00 | 7382.50 | 7383.95 | 7336.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 14:30:00 | 7383.50 | 7379.46 | 7338.82 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-09 11:15:00 | 7281.00 | 7342.06 | 7332.74 | SL hit (close<static) qty=1.00 sl=7325.00 alert=retest2 |

### Cycle 207 — SELL (started 2026-01-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-09 13:15:00 | 7256.00 | 7313.24 | 7320.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 7183.50 | 7270.39 | 7297.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 12:15:00 | 7254.50 | 7246.68 | 7277.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 12:15:00 | 7254.50 | 7246.68 | 7277.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 12:15:00 | 7254.50 | 7246.68 | 7277.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 12:30:00 | 7272.50 | 7246.68 | 7277.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 7265.00 | 7252.24 | 7275.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:45:00 | 7284.00 | 7252.24 | 7275.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 7267.00 | 7255.19 | 7274.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 7268.50 | 7255.19 | 7274.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 7255.00 | 7255.15 | 7272.63 | EMA400 retest candle locked (from downside) |

### Cycle 208 — BUY (started 2026-01-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 15:15:00 | 7308.00 | 7280.74 | 7278.76 | EMA200 above EMA400 |

### Cycle 209 — SELL (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 09:15:00 | 7273.00 | 7279.89 | 7280.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 6909.00 | 7016.22 | 7105.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 6891.00 | 6868.01 | 6953.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 09:45:00 | 6895.50 | 6868.01 | 6953.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 6820.00 | 6793.39 | 6818.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 6861.50 | 6793.39 | 6818.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 6854.00 | 6805.51 | 6821.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 6843.50 | 6805.51 | 6821.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 6883.50 | 6821.11 | 6827.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 11:00:00 | 6883.50 | 6821.11 | 6827.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 210 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 6866.50 | 6836.73 | 6833.68 | EMA200 above EMA400 |

### Cycle 211 — SELL (started 2026-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 10:15:00 | 6807.00 | 6833.63 | 6834.67 | EMA200 below EMA400 |

### Cycle 212 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 6962.00 | 6843.44 | 6835.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 11:15:00 | 6989.00 | 6941.22 | 6903.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 14:15:00 | 6941.50 | 6951.97 | 6918.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 14:15:00 | 6941.50 | 6951.97 | 6918.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 6941.50 | 6951.97 | 6918.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 14:45:00 | 6956.00 | 6951.97 | 6918.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 6890.00 | 6939.57 | 6915.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 6890.00 | 6939.57 | 6915.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 6864.00 | 6924.46 | 6911.20 | EMA400 retest candle locked (from upside) |

### Cycle 213 — SELL (started 2026-02-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 12:15:00 | 6876.00 | 6902.22 | 6903.09 | EMA200 below EMA400 |

### Cycle 214 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 6928.00 | 6906.70 | 6904.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 7034.50 | 6935.11 | 6918.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 7072.00 | 7088.44 | 7049.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 10:00:00 | 7072.00 | 7088.44 | 7049.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 7065.00 | 7079.60 | 7052.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:15:00 | 7086.00 | 7080.08 | 7054.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-06 09:15:00 | 7040.00 | 7085.53 | 7066.85 | SL hit (close<static) qty=1.00 sl=7043.00 alert=retest2 |

### Cycle 215 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 7568.00 | 7595.58 | 7598.56 | EMA200 below EMA400 |

### Cycle 216 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 7622.50 | 7600.30 | 7599.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 7653.50 | 7612.20 | 7605.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 7755.00 | 7765.78 | 7733.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-26 11:15:00 | 7755.00 | 7765.78 | 7733.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 11:15:00 | 7755.00 | 7765.78 | 7733.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 12:00:00 | 7755.00 | 7765.78 | 7733.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 7807.00 | 7777.38 | 7751.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 11:45:00 | 7827.00 | 7787.32 | 7760.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 12:15:00 | 7829.50 | 7787.32 | 7760.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 7701.50 | 7792.65 | 7777.31 | SL hit (close<static) qty=1.00 sl=7731.50 alert=retest2 |

### Cycle 217 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 7720.50 | 7766.76 | 7767.49 | EMA200 below EMA400 |

### Cycle 218 — BUY (started 2026-03-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 14:15:00 | 7805.50 | 7768.37 | 7767.34 | EMA200 above EMA400 |

### Cycle 219 — SELL (started 2026-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-04 09:15:00 | 7556.50 | 7727.86 | 7749.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 11:15:00 | 7497.50 | 7656.61 | 7711.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 14:15:00 | 7645.00 | 7621.55 | 7678.75 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 14:15:00 | 7645.00 | 7621.55 | 7678.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 7645.00 | 7621.55 | 7678.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:45:00 | 7699.50 | 7621.55 | 7678.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 15:15:00 | 7715.00 | 7640.24 | 7682.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:15:00 | 7706.00 | 7640.24 | 7682.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 7720.00 | 7656.19 | 7685.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 09:30:00 | 7735.50 | 7656.19 | 7685.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 7664.00 | 7657.75 | 7683.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 11:45:00 | 7640.00 | 7655.40 | 7680.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 7761.50 | 7685.80 | 7688.65 | SL hit (close>static) qty=1.00 sl=7722.00 alert=retest2 |

### Cycle 220 — BUY (started 2026-03-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 15:15:00 | 7792.50 | 7707.14 | 7698.09 | EMA200 above EMA400 |

### Cycle 221 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 7626.50 | 7702.97 | 7703.47 | EMA200 below EMA400 |

### Cycle 222 — BUY (started 2026-03-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-09 14:15:00 | 7776.00 | 7712.05 | 7704.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 09:15:00 | 7827.50 | 7747.77 | 7722.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 09:15:00 | 7722.50 | 7776.94 | 7755.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 09:15:00 | 7722.50 | 7776.94 | 7755.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 7722.50 | 7776.94 | 7755.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 10:00:00 | 7722.50 | 7776.94 | 7755.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 7712.50 | 7764.05 | 7751.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:00:00 | 7712.50 | 7764.05 | 7751.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 223 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 7698.00 | 7744.91 | 7744.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 7667.00 | 7723.03 | 7734.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 11:15:00 | 7598.00 | 7585.83 | 7634.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-13 11:45:00 | 7610.00 | 7585.83 | 7634.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 7548.50 | 7560.16 | 7601.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 10:15:00 | 7494.50 | 7560.16 | 7601.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 10:30:00 | 7536.50 | 7504.67 | 7539.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 12:15:00 | 7540.00 | 7531.78 | 7535.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 7119.77 | 7283.23 | 7349.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 7159.67 | 7283.23 | 7349.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 10:15:00 | 7163.00 | 7283.23 | 7349.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-24 09:15:00 | 7371.00 | 7217.59 | 7274.90 | SL hit (close>ema200) qty=0.50 sl=7217.59 alert=retest2 |

### Cycle 224 — BUY (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 12:15:00 | 7431.50 | 7304.06 | 7303.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 7541.00 | 7396.69 | 7351.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 7533.00 | 7536.52 | 7461.04 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 11:00:00 | 7600.50 | 7549.32 | 7473.71 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 11:30:00 | 7618.00 | 7556.45 | 7483.83 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-03-27 12:15:00 | 7602.00 | 7556.45 | 7483.83 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 7427.50 | 7529.07 | 7498.99 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-30 09:15:00 | 7427.50 | 7529.07 | 7498.99 | SL hit (close<ema400) qty=1.00 sl=7498.99 alert=retest1 |

### Cycle 225 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 7394.00 | 7464.66 | 7473.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-01 10:15:00 | 7306.50 | 7424.00 | 7449.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-02 13:15:00 | 7346.50 | 7295.50 | 7345.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 13:15:00 | 7346.50 | 7295.50 | 7345.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 13:15:00 | 7346.50 | 7295.50 | 7345.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:00:00 | 7346.50 | 7295.50 | 7345.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 14:15:00 | 7325.50 | 7301.50 | 7344.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 14:30:00 | 7330.00 | 7301.50 | 7344.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 09:15:00 | 7274.50 | 7295.70 | 7334.02 | EMA400 retest candle locked (from downside) |

### Cycle 226 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 7381.00 | 7335.74 | 7332.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 10:15:00 | 7545.50 | 7480.84 | 7434.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-10 14:15:00 | 7506.50 | 7507.91 | 7464.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 7467.00 | 7501.58 | 7469.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 7467.00 | 7501.58 | 7469.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 7518.00 | 7509.37 | 7475.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 15:15:00 | 7740.00 | 7760.47 | 7761.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 227 — SELL (started 2026-04-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 15:15:00 | 7740.00 | 7760.47 | 7761.13 | EMA200 below EMA400 |

### Cycle 228 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 7777.00 | 7763.78 | 7762.57 | EMA200 above EMA400 |

### Cycle 229 — SELL (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 10:15:00 | 7753.50 | 7761.72 | 7761.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 12:15:00 | 7722.00 | 7752.54 | 7757.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 7762.00 | 7663.53 | 7686.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 7762.00 | 7663.53 | 7686.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 7762.00 | 7663.53 | 7686.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:00:00 | 7762.00 | 7663.53 | 7686.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 7760.00 | 7682.82 | 7692.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:30:00 | 7776.00 | 7682.82 | 7692.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 230 — BUY (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 12:15:00 | 7737.00 | 7704.33 | 7701.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 14:15:00 | 7774.50 | 7738.69 | 7723.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 15:15:00 | 7755.00 | 7759.20 | 7745.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 09:15:00 | 7765.50 | 7759.20 | 7745.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 7768.50 | 7761.06 | 7747.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:30:00 | 7760.00 | 7761.06 | 7747.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2023-05-24 09:15:00 | 4562.55 | 2023-05-24 14:15:00 | 4534.95 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2023-05-25 09:15:00 | 4558.45 | 2023-05-31 13:15:00 | 4561.65 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2023-05-25 15:00:00 | 4563.95 | 2023-05-31 13:15:00 | 4561.65 | STOP_HIT | 1.00 | -0.05% |
| SELL | retest2 | 2023-06-12 14:15:00 | 4924.00 | 2023-06-13 09:15:00 | 4969.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2023-07-10 12:15:00 | 5185.00 | 2023-07-10 15:15:00 | 5160.00 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2023-07-21 13:30:00 | 5225.95 | 2023-07-24 14:15:00 | 5187.15 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2023-08-11 11:15:00 | 4957.45 | 2023-08-16 15:15:00 | 4946.00 | STOP_HIT | 1.00 | 0.23% |
| SELL | retest2 | 2023-08-29 09:30:00 | 4872.00 | 2023-08-29 10:15:00 | 4906.00 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2023-08-31 13:45:00 | 4850.10 | 2023-09-05 09:15:00 | 4950.00 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2023-09-11 09:15:00 | 5078.00 | 2023-09-12 12:15:00 | 4983.05 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2023-09-27 09:15:00 | 5064.55 | 2023-09-27 09:15:00 | 5038.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2023-09-27 10:45:00 | 5068.00 | 2023-09-28 13:15:00 | 5042.95 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2023-10-09 12:45:00 | 5028.60 | 2023-10-10 12:15:00 | 5063.25 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2023-10-20 09:15:00 | 4959.60 | 2023-10-31 09:15:00 | 4879.60 | STOP_HIT | 1.00 | 1.61% |
| SELL | retest2 | 2023-10-20 12:30:00 | 4967.00 | 2023-10-31 09:15:00 | 4879.60 | STOP_HIT | 1.00 | 1.76% |
| SELL | retest2 | 2023-10-20 15:15:00 | 4952.00 | 2023-10-31 09:15:00 | 4879.60 | STOP_HIT | 1.00 | 1.46% |
| SELL | retest2 | 2023-10-23 15:00:00 | 4966.45 | 2023-10-31 09:15:00 | 4879.60 | STOP_HIT | 1.00 | 1.75% |
| SELL | retest2 | 2023-10-27 14:45:00 | 4835.15 | 2023-10-31 09:15:00 | 4879.60 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2023-10-30 11:00:00 | 4827.70 | 2023-10-31 09:15:00 | 4879.60 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2023-11-01 13:00:00 | 4817.50 | 2023-11-02 10:15:00 | 4868.20 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2023-11-08 15:00:00 | 5110.00 | 2023-11-24 10:15:00 | 5447.50 | STOP_HIT | 1.00 | 6.60% |
| BUY | retest2 | 2023-12-05 12:45:00 | 5604.60 | 2023-12-06 12:15:00 | 5548.35 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2023-12-05 14:00:00 | 5596.95 | 2023-12-06 12:15:00 | 5548.35 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2023-12-06 09:15:00 | 5621.50 | 2023-12-06 12:15:00 | 5548.35 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2024-01-01 14:00:00 | 5764.45 | 2024-01-08 09:15:00 | 5694.00 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-01-03 14:00:00 | 5771.50 | 2024-01-08 09:15:00 | 5694.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2024-01-04 10:30:00 | 5770.05 | 2024-01-08 09:15:00 | 5694.00 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2024-01-04 11:30:00 | 5759.75 | 2024-01-08 09:15:00 | 5694.00 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2024-01-05 12:00:00 | 5796.15 | 2024-01-08 09:15:00 | 5694.00 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2024-01-10 14:30:00 | 5763.95 | 2024-01-23 09:15:00 | 6340.35 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2024-02-22 09:15:00 | 6838.80 | 2024-02-23 11:15:00 | 6727.05 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-03-20 11:30:00 | 6148.45 | 2024-03-27 14:15:00 | 6237.95 | STOP_HIT | 1.00 | 1.46% |
| BUY | retest2 | 2024-03-20 12:30:00 | 6149.00 | 2024-03-27 14:15:00 | 6237.95 | STOP_HIT | 1.00 | 1.45% |
| BUY | retest2 | 2024-03-21 09:45:00 | 6151.50 | 2024-03-27 14:15:00 | 6237.95 | STOP_HIT | 1.00 | 1.41% |
| SELL | retest2 | 2024-04-18 09:15:00 | 6305.00 | 2024-04-23 09:15:00 | 6252.85 | STOP_HIT | 1.00 | 0.83% |
| BUY | retest2 | 2024-04-25 13:15:00 | 6252.35 | 2024-04-29 09:15:00 | 5814.85 | STOP_HIT | 1.00 | -7.00% |
| SELL | retest2 | 2024-05-10 13:00:00 | 5835.75 | 2024-05-14 14:15:00 | 5865.20 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest2 | 2024-05-13 09:15:00 | 5817.80 | 2024-05-14 14:15:00 | 5865.20 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2024-05-14 09:15:00 | 5790.35 | 2024-05-14 14:15:00 | 5865.20 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-05-29 15:15:00 | 5898.00 | 2024-06-03 12:15:00 | 5871.55 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2024-06-03 10:00:00 | 5890.55 | 2024-06-03 12:15:00 | 5871.55 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest1 | 2024-06-12 11:30:00 | 6138.00 | 2024-06-19 09:15:00 | 6140.85 | STOP_HIT | 1.00 | 0.05% |
| BUY | retest1 | 2024-06-12 14:00:00 | 6142.10 | 2024-06-19 09:15:00 | 6140.85 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2024-06-18 15:00:00 | 6242.90 | 2024-06-19 09:15:00 | 6140.85 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2024-06-28 12:30:00 | 6157.10 | 2024-07-03 12:15:00 | 6183.75 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2024-07-01 09:15:00 | 6113.00 | 2024-07-03 12:15:00 | 6183.75 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2024-07-05 14:00:00 | 6287.10 | 2024-07-19 13:15:00 | 6401.00 | STOP_HIT | 1.00 | 1.81% |
| BUY | retest2 | 2024-07-09 11:30:00 | 6281.55 | 2024-07-19 13:15:00 | 6401.00 | STOP_HIT | 1.00 | 1.90% |
| SELL | retest2 | 2024-07-24 12:15:00 | 6391.05 | 2024-07-24 15:15:00 | 6422.65 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2024-07-30 14:30:00 | 6637.80 | 2024-08-06 14:15:00 | 6654.05 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest2 | 2024-07-31 09:30:00 | 6649.15 | 2024-08-06 14:15:00 | 6654.05 | STOP_HIT | 1.00 | 0.07% |
| BUY | retest2 | 2024-07-31 10:00:00 | 6641.20 | 2024-08-06 14:15:00 | 6654.05 | STOP_HIT | 1.00 | 0.19% |
| BUY | retest2 | 2024-07-31 11:45:00 | 6637.00 | 2024-08-06 14:15:00 | 6654.05 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest2 | 2024-08-01 10:00:00 | 6670.40 | 2024-08-06 14:15:00 | 6654.05 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2024-08-05 09:45:00 | 6707.00 | 2024-08-06 14:15:00 | 6654.05 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2024-08-16 09:15:00 | 6650.00 | 2024-08-26 15:15:00 | 6780.00 | STOP_HIT | 1.00 | 1.95% |
| BUY | retest2 | 2024-09-03 09:30:00 | 6903.25 | 2024-09-03 11:15:00 | 6866.50 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2024-09-18 09:30:00 | 7046.00 | 2024-09-18 12:15:00 | 6995.15 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2024-09-27 10:45:00 | 7194.65 | 2024-10-01 14:15:00 | 7165.75 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2024-09-27 13:30:00 | 7186.00 | 2024-10-01 14:15:00 | 7165.75 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2024-09-27 15:00:00 | 7231.70 | 2024-10-01 14:15:00 | 7165.75 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-10-01 09:30:00 | 7188.75 | 2024-10-01 14:15:00 | 7165.75 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2024-10-14 11:15:00 | 7052.30 | 2024-10-17 11:15:00 | 7045.90 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2024-10-21 13:15:00 | 6972.35 | 2024-10-25 10:15:00 | 6950.70 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2024-10-22 10:00:00 | 6964.55 | 2024-10-25 10:15:00 | 6950.70 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2024-10-25 09:30:00 | 6969.85 | 2024-10-25 10:15:00 | 6950.70 | STOP_HIT | 1.00 | 0.27% |
| SELL | retest2 | 2024-12-02 11:15:00 | 6912.90 | 2024-12-02 12:15:00 | 7013.95 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2024-12-16 14:30:00 | 7230.10 | 2024-12-17 10:15:00 | 7267.80 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2024-12-17 09:30:00 | 7224.35 | 2024-12-17 10:15:00 | 7267.80 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-12-19 09:15:00 | 7191.00 | 2024-12-19 12:15:00 | 7257.15 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2024-12-19 10:00:00 | 7219.65 | 2024-12-19 12:15:00 | 7257.15 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2024-12-31 11:45:00 | 7343.75 | 2025-01-03 13:15:00 | 7321.75 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-01-01 09:15:00 | 7386.70 | 2025-01-03 13:15:00 | 7321.75 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-01-03 12:30:00 | 7346.45 | 2025-01-03 13:15:00 | 7321.75 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-01-07 11:15:00 | 7498.85 | 2025-01-08 09:15:00 | 7358.25 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-02-01 11:30:00 | 6933.90 | 2025-02-03 11:15:00 | 6776.15 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2025-02-01 13:15:00 | 6891.05 | 2025-02-03 11:15:00 | 6776.15 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2025-02-01 13:45:00 | 6903.95 | 2025-02-03 11:15:00 | 6776.15 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-02-11 10:45:00 | 6389.25 | 2025-02-18 12:15:00 | 6355.95 | STOP_HIT | 1.00 | 0.52% |
| SELL | retest2 | 2025-02-12 12:30:00 | 6456.65 | 2025-02-18 12:15:00 | 6355.95 | STOP_HIT | 1.00 | 1.56% |
| SELL | retest2 | 2025-03-03 11:15:00 | 6114.20 | 2025-03-03 15:15:00 | 6182.00 | STOP_HIT | 1.00 | -1.11% |
| SELL | retest2 | 2025-03-12 09:15:00 | 6086.85 | 2025-03-17 11:15:00 | 6160.70 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2025-03-13 09:15:00 | 6113.90 | 2025-03-17 11:15:00 | 6160.70 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-03-13 11:15:00 | 6119.45 | 2025-03-17 11:15:00 | 6160.70 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-04-04 14:45:00 | 6709.15 | 2025-04-07 12:15:00 | 6648.10 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-04-07 10:15:00 | 6711.35 | 2025-04-07 12:15:00 | 6648.10 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-04-15 09:15:00 | 6870.00 | 2025-04-25 09:15:00 | 7014.00 | STOP_HIT | 1.00 | 2.10% |
| SELL | retest2 | 2025-04-29 10:15:00 | 7003.00 | 2025-05-02 09:15:00 | 7067.50 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-05-02 10:45:00 | 7010.50 | 2025-05-06 09:15:00 | 7045.00 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2025-05-05 10:45:00 | 7009.50 | 2025-05-06 09:15:00 | 7045.00 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2025-05-05 11:30:00 | 7000.00 | 2025-05-06 09:15:00 | 7045.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-05-27 11:15:00 | 7087.00 | 2025-05-28 09:15:00 | 7000.50 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-05-27 14:45:00 | 7073.50 | 2025-05-28 09:15:00 | 7000.50 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-05-30 13:00:00 | 6931.50 | 2025-06-02 09:15:00 | 7065.00 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-05-30 14:00:00 | 6932.00 | 2025-06-02 09:15:00 | 7065.00 | STOP_HIT | 1.00 | -1.92% |
| SELL | retest2 | 2025-05-30 14:30:00 | 6928.00 | 2025-06-02 09:15:00 | 7065.00 | STOP_HIT | 1.00 | -1.98% |
| SELL | retest2 | 2025-06-02 14:00:00 | 6920.50 | 2025-06-05 13:15:00 | 6880.00 | STOP_HIT | 1.00 | 0.59% |
| SELL | retest2 | 2025-06-03 10:45:00 | 6828.50 | 2025-06-05 13:15:00 | 6880.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-06-03 13:45:00 | 6825.50 | 2025-06-05 13:15:00 | 6880.00 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-06-03 14:45:00 | 6818.00 | 2025-06-05 13:15:00 | 6880.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-06-04 10:45:00 | 6820.50 | 2025-06-05 13:15:00 | 6880.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-06-05 11:15:00 | 6877.50 | 2025-06-05 13:15:00 | 6880.00 | STOP_HIT | 1.00 | -0.04% |
| SELL | retest2 | 2025-06-05 12:15:00 | 6884.00 | 2025-06-05 13:15:00 | 6880.00 | STOP_HIT | 1.00 | 0.06% |
| BUY | retest2 | 2025-06-09 14:15:00 | 6921.50 | 2025-06-11 09:15:00 | 6878.50 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-06-10 13:45:00 | 6924.50 | 2025-06-11 09:15:00 | 6878.50 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-06-11 12:15:00 | 6921.00 | 2025-06-11 13:15:00 | 6899.50 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-06-23 11:15:00 | 7042.00 | 2025-06-24 15:15:00 | 7000.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-06-24 10:00:00 | 7046.00 | 2025-06-24 15:15:00 | 7000.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-06-26 13:30:00 | 7058.00 | 2025-07-09 11:15:00 | 7498.00 | STOP_HIT | 1.00 | 6.23% |
| BUY | retest2 | 2025-07-17 09:30:00 | 7399.50 | 2025-07-18 09:15:00 | 7276.50 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-07-17 10:30:00 | 7397.00 | 2025-07-18 09:15:00 | 7276.50 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-07-17 11:00:00 | 7401.00 | 2025-07-18 09:15:00 | 7276.50 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-07-25 11:15:00 | 7435.00 | 2025-07-28 12:15:00 | 7340.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-07-25 12:30:00 | 7435.00 | 2025-07-28 12:15:00 | 7340.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-07-25 13:15:00 | 7448.00 | 2025-07-28 12:15:00 | 7340.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-07-28 09:45:00 | 7443.50 | 2025-07-28 12:15:00 | 7340.00 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-07-31 13:00:00 | 7445.00 | 2025-08-01 12:15:00 | 7389.00 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2025-08-08 10:30:00 | 7121.50 | 2025-08-11 13:15:00 | 7215.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-08-08 12:00:00 | 7118.50 | 2025-08-11 13:15:00 | 7215.00 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-08-22 11:15:00 | 7927.00 | 2025-08-25 10:15:00 | 7849.50 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-08-22 14:30:00 | 7931.00 | 2025-08-25 10:15:00 | 7849.50 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2025-08-22 15:15:00 | 7930.00 | 2025-08-25 10:15:00 | 7849.50 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-08-28 12:30:00 | 7785.50 | 2025-09-03 10:15:00 | 7739.50 | STOP_HIT | 1.00 | 0.59% |
| BUY | retest2 | 2025-09-12 12:15:00 | 7881.00 | 2025-09-15 09:15:00 | 7827.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-09-12 14:15:00 | 7880.00 | 2025-09-15 09:15:00 | 7827.00 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-10-13 09:15:00 | 7721.50 | 2025-10-13 11:15:00 | 7652.50 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-10-23 10:15:00 | 8095.50 | 2025-10-24 09:15:00 | 7900.00 | STOP_HIT | 1.00 | -2.41% |
| SELL | retest2 | 2025-10-30 09:15:00 | 7811.50 | 2025-11-03 15:15:00 | 7835.00 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-11-06 11:15:00 | 7799.50 | 2025-11-06 15:15:00 | 7783.00 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-11-06 14:00:00 | 7815.50 | 2025-11-06 15:15:00 | 7783.00 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-11-25 14:45:00 | 7348.00 | 2025-12-11 12:15:00 | 6980.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 10:00:00 | 7345.00 | 2025-12-11 12:15:00 | 6977.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-27 11:45:00 | 7350.50 | 2025-12-11 12:15:00 | 6982.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 09:30:00 | 7342.50 | 2025-12-11 13:15:00 | 6975.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-25 14:45:00 | 7348.00 | 2025-12-12 09:15:00 | 7050.00 | STOP_HIT | 0.50 | 4.06% |
| SELL | retest2 | 2025-11-26 10:00:00 | 7345.00 | 2025-12-12 09:15:00 | 7050.00 | STOP_HIT | 0.50 | 4.02% |
| SELL | retest2 | 2025-11-27 11:45:00 | 7350.50 | 2025-12-12 09:15:00 | 7050.00 | STOP_HIT | 0.50 | 4.09% |
| SELL | retest2 | 2025-12-01 09:30:00 | 7342.50 | 2025-12-12 09:15:00 | 7050.00 | STOP_HIT | 0.50 | 3.98% |
| SELL | retest2 | 2025-12-01 12:00:00 | 7335.50 | 2025-12-12 13:15:00 | 7103.00 | STOP_HIT | 1.00 | 3.17% |
| SELL | retest2 | 2025-12-01 12:30:00 | 7335.50 | 2025-12-12 13:15:00 | 7103.00 | STOP_HIT | 1.00 | 3.17% |
| BUY | retest2 | 2026-01-08 12:15:00 | 7396.50 | 2026-01-09 11:15:00 | 7281.00 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2026-01-08 13:30:00 | 7382.50 | 2026-01-09 11:15:00 | 7281.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-01-08 14:30:00 | 7383.50 | 2026-01-09 11:15:00 | 7281.00 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2026-02-05 13:15:00 | 7086.00 | 2026-02-06 09:15:00 | 7040.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2026-02-06 11:45:00 | 7109.00 | 2026-02-20 09:15:00 | 7568.00 | STOP_HIT | 1.00 | 6.46% |
| BUY | retest2 | 2026-02-27 11:45:00 | 7827.00 | 2026-03-02 09:15:00 | 7701.50 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2026-02-27 12:15:00 | 7829.50 | 2026-03-02 09:15:00 | 7701.50 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2026-03-05 11:45:00 | 7640.00 | 2026-03-05 14:15:00 | 7761.50 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-03-16 10:15:00 | 7494.50 | 2026-03-23 10:15:00 | 7119.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-17 10:30:00 | 7536.50 | 2026-03-23 10:15:00 | 7159.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-18 12:15:00 | 7540.00 | 2026-03-23 10:15:00 | 7163.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-16 10:15:00 | 7494.50 | 2026-03-24 09:15:00 | 7371.00 | STOP_HIT | 0.50 | 1.65% |
| SELL | retest2 | 2026-03-17 10:30:00 | 7536.50 | 2026-03-24 09:15:00 | 7371.00 | STOP_HIT | 0.50 | 2.20% |
| SELL | retest2 | 2026-03-18 12:15:00 | 7540.00 | 2026-03-24 09:15:00 | 7371.00 | STOP_HIT | 0.50 | 2.24% |
| BUY | retest1 | 2026-03-27 11:00:00 | 7600.50 | 2026-03-30 09:15:00 | 7427.50 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest1 | 2026-03-27 11:30:00 | 7618.00 | 2026-03-30 09:15:00 | 7427.50 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest1 | 2026-03-27 12:15:00 | 7602.00 | 2026-03-30 09:15:00 | 7427.50 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2026-04-13 10:45:00 | 7518.00 | 2026-04-28 15:15:00 | 7740.00 | STOP_HIT | 1.00 | 2.95% |
