# Sundaram Finance Ltd. (SUNDARMFIN)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 4700.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 81 |
| ALERT1 | 59 |
| ALERT2 | 58 |
| ALERT2_SKIP | 30 |
| ALERT3 | 143 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 7 |
| ENTRY2 | 65 |
| PARTIAL | 7 |
| TARGET_HIT | 2 |
| STOP_HIT | 70 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 79 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 19 / 60
- **Target hits / Stop hits / Partials:** 2 / 70 / 7
- **Avg / median % per leg:** -0.10% / -1.05%
- **Sum % (uncompounded):** -7.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 39 | 2 | 5.1% | 1 | 38 | 0 | -0.95% | -37.0% |
| BUY @ 2nd Alert (retest1) | 6 | 0 | 0.0% | 0 | 6 | 0 | -1.07% | -6.4% |
| BUY @ 3rd Alert (retest2) | 33 | 2 | 6.1% | 1 | 32 | 0 | -0.93% | -30.6% |
| SELL (all) | 40 | 17 | 42.5% | 1 | 32 | 7 | 0.73% | 29.3% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.61% | -1.6% |
| SELL @ 3rd Alert (retest2) | 39 | 17 | 43.6% | 1 | 31 | 7 | 0.79% | 30.9% |
| retest1 (combined) | 7 | 0 | 0.0% | 0 | 7 | 0 | -1.15% | -8.0% |
| retest2 (combined) | 72 | 19 | 26.4% | 2 | 63 | 7 | 0.00% | 0.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 10:15:00 | 5141.00 | 5054.63 | 5048.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 12:15:00 | 5170.00 | 5092.16 | 5067.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 10:15:00 | 5140.50 | 5145.85 | 5108.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-15 11:00:00 | 5140.50 | 5145.85 | 5108.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 5076.50 | 5129.04 | 5112.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 15:00:00 | 5076.50 | 5129.04 | 5112.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 5078.00 | 5118.83 | 5109.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 09:15:00 | 5078.00 | 5118.83 | 5109.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — SELL (started 2025-05-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 10:15:00 | 5044.00 | 5092.21 | 5098.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 12:15:00 | 5035.00 | 5073.70 | 5088.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 14:15:00 | 5100.00 | 5072.76 | 5084.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 14:15:00 | 5100.00 | 5072.76 | 5084.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 14:15:00 | 5100.00 | 5072.76 | 5084.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 15:00:00 | 5100.00 | 5072.76 | 5084.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 15:15:00 | 5094.50 | 5077.11 | 5085.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:15:00 | 5095.00 | 5077.11 | 5085.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 5075.50 | 5076.79 | 5084.78 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 12:15:00 | 5124.00 | 5096.28 | 5092.62 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 10:15:00 | 5065.00 | 5087.10 | 5089.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 11:15:00 | 5059.00 | 5081.48 | 5086.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-20 13:15:00 | 5104.00 | 5078.23 | 5084.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 13:15:00 | 5104.00 | 5078.23 | 5084.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 5104.00 | 5078.23 | 5084.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 5104.00 | 5078.23 | 5084.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 5108.50 | 5084.28 | 5086.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:45:00 | 5100.50 | 5084.28 | 5086.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 5118.50 | 5091.08 | 5089.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-21 12:15:00 | 5182.50 | 5125.69 | 5106.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-22 10:15:00 | 5150.00 | 5151.01 | 5128.79 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-22 10:45:00 | 5144.50 | 5151.01 | 5128.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 5103.00 | 5141.41 | 5126.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:45:00 | 5093.50 | 5141.41 | 5126.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 5114.00 | 5135.93 | 5125.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-22 12:30:00 | 5107.00 | 5135.93 | 5125.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 5115.00 | 5128.43 | 5124.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:15:00 | 5121.00 | 5128.43 | 5124.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 5172.00 | 5137.15 | 5128.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 11:15:00 | 5198.00 | 5143.92 | 5132.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 11:15:00 | 5201.50 | 5222.47 | 5187.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 14:30:00 | 5344.50 | 5229.09 | 5199.59 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 09:15:00 | 5114.00 | 5191.10 | 5200.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 09:15:00 | 5114.00 | 5191.10 | 5200.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 13:15:00 | 5074.00 | 5136.54 | 5168.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 12:15:00 | 5097.00 | 5066.00 | 5109.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-29 13:00:00 | 5097.00 | 5066.00 | 5109.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 13:15:00 | 5095.00 | 5071.80 | 5107.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 13:45:00 | 5103.00 | 5071.80 | 5107.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 5077.50 | 5078.26 | 5102.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 10:30:00 | 5062.00 | 5071.01 | 5096.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 13:00:00 | 5072.00 | 5075.53 | 5094.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 5147.00 | 5092.94 | 5099.49 | SL hit (close>static) qty=1.00 sl=5125.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 10:15:00 | 5140.00 | 5109.65 | 5106.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 11:15:00 | 5149.50 | 5117.62 | 5110.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 09:15:00 | 5080.50 | 5133.19 | 5123.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 09:15:00 | 5080.50 | 5133.19 | 5123.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 5080.50 | 5133.19 | 5123.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 10:00:00 | 5080.50 | 5133.19 | 5123.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 5080.00 | 5122.55 | 5119.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:15:00 | 5075.50 | 5122.55 | 5119.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 11:15:00 | 5075.00 | 5113.04 | 5115.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 5045.50 | 5099.53 | 5108.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 10:15:00 | 5097.50 | 5057.65 | 5080.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 10:15:00 | 5097.50 | 5057.65 | 5080.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 5097.50 | 5057.65 | 5080.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 5097.50 | 5057.65 | 5080.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 5100.00 | 5066.12 | 5081.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 14:15:00 | 5078.00 | 5078.15 | 5084.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-04 15:00:00 | 5058.50 | 5074.22 | 5082.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 13:00:00 | 5075.50 | 5045.95 | 5052.15 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-06 14:15:00 | 5075.00 | 5056.41 | 5056.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 14:15:00 | 5075.00 | 5056.41 | 5056.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 5112.50 | 5069.88 | 5062.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 09:15:00 | 5089.50 | 5092.28 | 5080.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-10 09:15:00 | 5089.50 | 5092.28 | 5080.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 5089.50 | 5092.28 | 5080.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 09:30:00 | 5087.00 | 5092.28 | 5080.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 10:15:00 | 5096.00 | 5093.02 | 5081.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 10:45:00 | 5080.00 | 5093.02 | 5081.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 11:15:00 | 5087.00 | 5091.82 | 5082.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 11:45:00 | 5088.50 | 5091.82 | 5082.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 12:15:00 | 5065.00 | 5086.45 | 5080.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 13:00:00 | 5065.00 | 5086.45 | 5080.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 13:15:00 | 5068.00 | 5082.76 | 5079.65 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-06-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 15:15:00 | 5046.00 | 5072.81 | 5075.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 15:15:00 | 5031.50 | 5060.68 | 5068.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 12:15:00 | 5024.00 | 5023.30 | 5039.34 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 12:30:00 | 5021.00 | 5023.30 | 5039.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 4977.50 | 5011.82 | 5028.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 14:30:00 | 4908.00 | 4960.22 | 4987.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 11:15:00 | 4860.00 | 4834.23 | 4832.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 11:15:00 | 4860.00 | 4834.23 | 4832.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 14:15:00 | 4904.00 | 4854.04 | 4842.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 15:15:00 | 4896.50 | 4898.20 | 4877.15 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 09:15:00 | 4937.50 | 4898.20 | 4877.15 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-06-25 10:15:00 | 4928.50 | 4899.26 | 4879.54 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 12:15:00 | 4912.50 | 4902.89 | 4886.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 12:30:00 | 4893.00 | 4902.89 | 4886.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 13:15:00 | 4898.00 | 4901.91 | 4887.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-25 13:45:00 | 4897.00 | 4901.91 | 4887.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 4889.50 | 4905.86 | 4893.38 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-06-26 09:15:00 | 4889.50 | 4905.86 | 4893.38 | SL hit (close<ema400) qty=1.00 sl=4893.38 alert=retest1 |

### Cycle 12 — SELL (started 2025-06-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 12:15:00 | 4869.50 | 4885.79 | 4886.13 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-06-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 11:15:00 | 4957.50 | 4894.34 | 4888.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 12:15:00 | 5056.50 | 4926.77 | 4903.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 09:15:00 | 5097.00 | 5114.22 | 5071.91 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-02 10:15:00 | 5096.50 | 5114.22 | 5071.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 5103.50 | 5110.44 | 5077.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 11:45:00 | 5064.00 | 5110.44 | 5077.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 5060.00 | 5100.35 | 5075.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 13:00:00 | 5060.00 | 5100.35 | 5075.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 13:15:00 | 5041.00 | 5088.48 | 5072.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 14:00:00 | 5041.00 | 5088.48 | 5072.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 5072.50 | 5082.57 | 5072.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:15:00 | 5114.50 | 5082.57 | 5072.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 10:15:00 | 5076.00 | 5137.18 | 5136.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 10:15:00 | 5074.50 | 5124.64 | 5131.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 10:15:00 | 5074.50 | 5124.64 | 5131.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 12:15:00 | 5067.00 | 5105.41 | 5120.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-10 12:15:00 | 5058.50 | 5049.72 | 5078.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-10 12:30:00 | 5061.00 | 5049.72 | 5078.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 5080.00 | 5056.38 | 5076.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 09:15:00 | 4985.00 | 5060.11 | 5076.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 14:15:00 | 5038.00 | 5021.67 | 5030.99 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-14 15:00:00 | 5042.00 | 5025.73 | 5031.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 09:30:00 | 5044.00 | 5030.67 | 5033.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 10:15:00 | 5060.00 | 5036.54 | 5035.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 5060.00 | 5036.54 | 5035.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 14:15:00 | 5090.00 | 5052.69 | 5043.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 11:15:00 | 5070.00 | 5078.66 | 5060.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 12:00:00 | 5070.00 | 5078.66 | 5060.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 5065.50 | 5076.03 | 5061.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:30:00 | 5065.00 | 5076.03 | 5061.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 13:15:00 | 5057.00 | 5072.23 | 5060.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:30:00 | 5062.00 | 5072.23 | 5060.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 14:15:00 | 5031.00 | 5063.98 | 5058.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 15:00:00 | 5031.00 | 5063.98 | 5058.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 15:15:00 | 5034.00 | 5057.98 | 5056.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 5062.50 | 5057.98 | 5056.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 5131.50 | 5167.57 | 5171.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 10:15:00 | 5131.50 | 5167.57 | 5171.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 11:15:00 | 5100.00 | 5154.06 | 5165.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 10:15:00 | 4520.00 | 4496.83 | 4557.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-05 10:30:00 | 4482.10 | 4496.83 | 4557.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 4628.00 | 4526.29 | 4560.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 13:00:00 | 4628.00 | 4526.29 | 4560.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 4621.20 | 4545.28 | 4566.42 | EMA400 retest candle locked (from downside) |

### Cycle 17 — BUY (started 2025-08-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 15:15:00 | 4650.00 | 4582.64 | 4580.76 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 4536.70 | 4571.49 | 4575.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 11:15:00 | 4515.60 | 4560.31 | 4570.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 09:15:00 | 4606.70 | 4545.34 | 4556.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 4606.70 | 4545.34 | 4556.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 4606.70 | 4545.34 | 4556.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:45:00 | 4609.00 | 4545.34 | 4556.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 4583.90 | 4553.05 | 4558.55 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 11:15:00 | 4626.60 | 4567.76 | 4564.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 15:15:00 | 4657.00 | 4608.28 | 4586.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 09:15:00 | 4557.40 | 4598.10 | 4583.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 09:15:00 | 4557.40 | 4598.10 | 4583.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 4557.40 | 4598.10 | 4583.97 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 11:15:00 | 4523.10 | 4571.65 | 4573.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 10:15:00 | 4494.20 | 4528.62 | 4544.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 09:15:00 | 4541.90 | 4516.25 | 4529.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 4541.90 | 4516.25 | 4529.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 4541.90 | 4516.25 | 4529.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 09:30:00 | 4556.70 | 4516.25 | 4529.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 4510.00 | 4515.00 | 4527.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 14:45:00 | 4482.50 | 4504.46 | 4518.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 13:15:00 | 4539.20 | 4522.11 | 4521.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 13:15:00 | 4539.20 | 4522.11 | 4521.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 09:15:00 | 4763.90 | 4572.54 | 4544.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-25 09:15:00 | 5112.30 | 5214.94 | 5146.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 5112.30 | 5214.94 | 5146.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 5112.30 | 5214.94 | 5146.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:30:00 | 5125.00 | 5214.94 | 5146.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 5079.30 | 5187.82 | 5139.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-25 11:00:00 | 5079.30 | 5187.82 | 5139.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-08-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 13:15:00 | 5030.00 | 5113.88 | 5114.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 14:15:00 | 4963.00 | 5083.71 | 5100.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 10:15:00 | 5100.00 | 5061.26 | 5083.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 10:15:00 | 5100.00 | 5061.26 | 5083.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 10:15:00 | 5100.00 | 5061.26 | 5083.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 11:00:00 | 5100.00 | 5061.26 | 5083.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 5047.20 | 5058.45 | 5080.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 12:15:00 | 5035.10 | 5058.45 | 5080.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-26 15:15:00 | 4783.35 | 4966.24 | 5027.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-29 09:15:00 | 4531.59 | 4670.62 | 4812.15 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 13:15:00 | 4514.30 | 4472.61 | 4470.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 13:15:00 | 4532.20 | 4494.44 | 4482.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-05 15:15:00 | 4500.30 | 4504.20 | 4489.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 15:15:00 | 4500.30 | 4504.20 | 4489.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 4500.30 | 4504.20 | 4489.92 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 4462.90 | 4481.35 | 4482.84 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 10:15:00 | 4514.80 | 4488.80 | 4485.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 14:15:00 | 4573.90 | 4512.89 | 4498.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-10 14:15:00 | 4561.20 | 4561.96 | 4535.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-10 15:00:00 | 4561.20 | 4561.96 | 4535.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 4525.70 | 4553.45 | 4536.31 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 14:15:00 | 4494.70 | 4524.51 | 4527.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 13:15:00 | 4475.00 | 4505.11 | 4515.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 4525.90 | 4457.19 | 4472.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 4525.90 | 4457.19 | 4472.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 4525.90 | 4457.19 | 4472.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:45:00 | 4515.00 | 4457.19 | 4472.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 4518.10 | 4469.37 | 4476.25 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 12:15:00 | 4524.20 | 4487.29 | 4483.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 13:15:00 | 4526.90 | 4495.21 | 4487.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 09:15:00 | 4597.60 | 4613.18 | 4573.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 4597.60 | 4613.18 | 4573.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 4597.60 | 4613.18 | 4573.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:00:00 | 4597.60 | 4613.18 | 4573.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 4590.60 | 4601.97 | 4580.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 13:30:00 | 4594.30 | 4601.97 | 4580.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 14:15:00 | 4585.40 | 4598.66 | 4580.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 14:30:00 | 4588.60 | 4598.66 | 4580.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 4590.00 | 4596.93 | 4581.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 4588.40 | 4596.93 | 4581.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 4593.90 | 4596.32 | 4582.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:30:00 | 4630.00 | 4596.32 | 4582.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 4585.00 | 4594.06 | 4582.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 10:45:00 | 4584.70 | 4594.06 | 4582.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 11:15:00 | 4531.70 | 4581.59 | 4578.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-19 12:00:00 | 4531.70 | 4581.59 | 4578.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-09-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 12:15:00 | 4530.80 | 4571.43 | 4574.00 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 4592.50 | 4575.94 | 4575.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 12:15:00 | 4645.50 | 4596.00 | 4585.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 11:15:00 | 4619.50 | 4635.02 | 4613.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 11:15:00 | 4619.50 | 4635.02 | 4613.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 4619.50 | 4635.02 | 4613.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 4591.70 | 4635.02 | 4613.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 4642.00 | 4638.71 | 4622.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 09:45:00 | 4675.00 | 4643.89 | 4626.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 12:45:00 | 4655.60 | 4648.39 | 4633.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 09:30:00 | 4669.90 | 4647.60 | 4637.52 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-25 13:15:00 | 4613.80 | 4629.77 | 4631.45 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-09-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 13:15:00 | 4613.80 | 4629.77 | 4631.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 4577.80 | 4619.37 | 4626.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 10:15:00 | 4428.90 | 4401.58 | 4464.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 11:00:00 | 4428.90 | 4401.58 | 4464.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 4470.90 | 4415.83 | 4460.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:00:00 | 4470.90 | 4415.83 | 4460.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 4459.00 | 4424.47 | 4460.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 14:00:00 | 4459.00 | 4424.47 | 4460.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 4460.10 | 4431.59 | 4460.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 4460.10 | 4431.59 | 4460.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 4230.00 | 4391.27 | 4439.32 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 4461.60 | 4415.68 | 4414.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 14:15:00 | 4478.00 | 4449.03 | 4433.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 10:15:00 | 4441.50 | 4456.23 | 4441.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 10:15:00 | 4441.50 | 4456.23 | 4441.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 4441.50 | 4456.23 | 4441.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:00:00 | 4441.50 | 4456.23 | 4441.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 4434.50 | 4451.88 | 4440.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 12:00:00 | 4434.50 | 4451.88 | 4440.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 4416.70 | 4444.85 | 4438.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:00:00 | 4416.70 | 4444.85 | 4438.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 13:15:00 | 4421.50 | 4440.18 | 4436.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 13:30:00 | 4417.10 | 4440.18 | 4436.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — SELL (started 2025-10-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 15:15:00 | 4419.70 | 4432.02 | 4433.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 09:15:00 | 4395.80 | 4424.78 | 4430.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 13:15:00 | 4371.00 | 4365.05 | 4385.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 14:00:00 | 4371.00 | 4365.05 | 4385.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 4391.60 | 4370.36 | 4386.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:30:00 | 4408.60 | 4370.36 | 4386.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 4390.00 | 4374.29 | 4386.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 4408.80 | 4379.99 | 4388.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 4421.00 | 4388.19 | 4391.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:00:00 | 4421.00 | 4388.19 | 4391.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 11:15:00 | 4413.60 | 4393.28 | 4393.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 12:15:00 | 4423.90 | 4399.40 | 4396.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 15:15:00 | 4398.90 | 4404.00 | 4399.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 15:15:00 | 4398.90 | 4404.00 | 4399.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 4398.90 | 4404.00 | 4399.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 4448.70 | 4404.00 | 4399.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 4419.00 | 4407.00 | 4401.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 13:45:00 | 4482.90 | 4427.23 | 4418.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:30:00 | 4465.00 | 4433.89 | 4423.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 10:00:00 | 4466.70 | 4433.89 | 4423.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 11:15:00 | 4465.40 | 4438.25 | 4426.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 4483.00 | 4489.96 | 4467.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:45:00 | 4480.10 | 4489.96 | 4467.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 09:15:00 | 4494.00 | 4491.52 | 4475.38 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-20 10:15:00 | 4455.70 | 4473.15 | 4474.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-10-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 10:15:00 | 4455.70 | 4473.15 | 4474.42 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2025-10-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 11:15:00 | 4516.10 | 4481.74 | 4478.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 13:15:00 | 4541.20 | 4499.75 | 4487.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 15:15:00 | 4651.00 | 4662.11 | 4631.43 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-28 09:30:00 | 4711.20 | 4670.91 | 4638.22 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 4626.80 | 4677.41 | 4661.53 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 4626.80 | 4677.41 | 4661.53 | SL hit (close<ema400) qty=1.00 sl=4661.53 alert=retest1 |

### Cycle 36 — SELL (started 2025-10-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 13:15:00 | 4635.20 | 4649.47 | 4651.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 09:15:00 | 4616.30 | 4640.55 | 4646.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 09:15:00 | 4601.10 | 4597.04 | 4617.34 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-10-31 15:15:00 | 4542.00 | 4598.78 | 4610.68 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 4542.00 | 4587.42 | 4604.43 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 4614.90 | 4592.92 | 4605.39 | SL hit (close>ema400) qty=1.00 sl=4605.39 alert=retest1 |

### Cycle 37 — BUY (started 2025-11-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 13:15:00 | 4692.50 | 4623.83 | 4616.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-04 09:15:00 | 4761.40 | 4668.72 | 4640.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-07 13:15:00 | 4817.20 | 4823.62 | 4785.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-07 14:00:00 | 4817.20 | 4823.62 | 4785.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 4800.00 | 4812.96 | 4789.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-10 10:15:00 | 4815.00 | 4812.96 | 4789.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-11 09:15:00 | 4826.10 | 4805.53 | 4795.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-11 15:15:00 | 4778.10 | 4797.26 | 4796.32 | SL hit (close<static) qty=1.00 sl=4784.80 alert=retest2 |

### Cycle 38 — SELL (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 09:15:00 | 4623.20 | 4762.45 | 4780.58 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 4779.60 | 4745.97 | 4744.92 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 09:15:00 | 4626.00 | 4734.27 | 4742.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 10:15:00 | 4619.70 | 4711.35 | 4731.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 14:15:00 | 4710.60 | 4647.90 | 4670.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 14:15:00 | 4710.60 | 4647.90 | 4670.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 14:15:00 | 4710.60 | 4647.90 | 4670.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 14:45:00 | 4703.40 | 4647.90 | 4670.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 15:15:00 | 4702.00 | 4658.72 | 4672.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:15:00 | 4720.50 | 4658.72 | 4672.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — BUY (started 2025-11-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 10:15:00 | 4736.40 | 4685.66 | 4683.46 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-11-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 14:15:00 | 4661.90 | 4680.08 | 4681.69 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-21 10:15:00 | 4711.00 | 4681.31 | 4681.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 11:15:00 | 4724.00 | 4689.85 | 4685.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 09:15:00 | 4673.60 | 4692.87 | 4689.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 09:15:00 | 4673.60 | 4692.87 | 4689.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 4673.60 | 4692.87 | 4689.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 4673.60 | 4692.87 | 4689.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 4729.70 | 4700.23 | 4692.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 12:30:00 | 4747.40 | 4710.20 | 4699.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 13:15:00 | 4739.40 | 4710.20 | 4699.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 13:45:00 | 4737.40 | 4714.76 | 4702.11 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 14:45:00 | 4740.90 | 4716.97 | 4704.26 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 4691.20 | 4713.97 | 4706.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 11:00:00 | 4691.20 | 4713.97 | 4706.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 11:15:00 | 4734.80 | 4718.13 | 4708.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 13:00:00 | 4745.00 | 4723.51 | 4712.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-25 15:00:00 | 4757.50 | 4733.12 | 4718.66 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 09:15:00 | 4759.30 | 4733.10 | 4719.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-26 13:45:00 | 4755.80 | 4749.18 | 4734.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 14:15:00 | 4732.80 | 4745.91 | 4734.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-26 15:00:00 | 4732.80 | 4745.91 | 4734.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 15:15:00 | 4720.00 | 4740.73 | 4732.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 09:15:00 | 4695.00 | 4740.73 | 4732.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-27 09:15:00 | 4670.80 | 4726.74 | 4727.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 09:15:00 | 4670.80 | 4726.74 | 4727.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 4645.40 | 4679.26 | 4698.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 13:15:00 | 4671.80 | 4668.75 | 4686.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-28 14:00:00 | 4671.80 | 4668.75 | 4686.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 4732.90 | 4681.58 | 4690.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:00:00 | 4732.90 | 4681.58 | 4690.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 15:15:00 | 4720.00 | 4689.27 | 4693.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 09:30:00 | 4733.70 | 4695.21 | 4695.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-12-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 10:15:00 | 4740.00 | 4704.17 | 4699.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 14:15:00 | 4765.80 | 4730.57 | 4714.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 4691.70 | 4725.91 | 4715.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 09:15:00 | 4691.70 | 4725.91 | 4715.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 09:15:00 | 4691.70 | 4725.91 | 4715.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:30:00 | 4685.20 | 4725.91 | 4715.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 4687.70 | 4718.27 | 4713.17 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 4685.70 | 4708.46 | 4709.39 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 14:15:00 | 4749.60 | 4716.90 | 4713.08 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 4665.00 | 4705.42 | 4708.46 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 10:15:00 | 4853.10 | 4715.45 | 4702.48 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 13:15:00 | 4690.00 | 4727.40 | 4730.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 13:15:00 | 4674.60 | 4699.41 | 4713.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-10 15:15:00 | 4740.00 | 4705.48 | 4713.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 15:15:00 | 4740.00 | 4705.48 | 4713.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 4740.00 | 4705.48 | 4713.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 10:45:00 | 4670.00 | 4694.95 | 4707.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 10:15:00 | 4794.50 | 4722.77 | 4715.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 10:15:00 | 4794.50 | 4722.77 | 4715.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 12:15:00 | 4821.50 | 4753.60 | 4731.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 15:15:00 | 4751.00 | 4764.15 | 4743.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 09:15:00 | 4756.40 | 4764.15 | 4743.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 4762.20 | 4763.76 | 4744.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 10:15:00 | 4782.60 | 4763.76 | 4744.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 15:00:00 | 4773.60 | 4772.00 | 4769.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 15:15:00 | 4732.50 | 4764.10 | 4765.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — SELL (started 2025-12-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 15:15:00 | 4732.50 | 4764.10 | 4765.86 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-12-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 10:15:00 | 4779.70 | 4768.99 | 4767.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-17 12:15:00 | 4787.00 | 4774.34 | 4770.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-18 13:15:00 | 4780.00 | 4793.17 | 4785.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-18 13:15:00 | 4780.00 | 4793.17 | 4785.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 13:15:00 | 4780.00 | 4793.17 | 4785.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 13:30:00 | 4787.90 | 4793.17 | 4785.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 14:15:00 | 4809.50 | 4796.43 | 4787.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-18 15:00:00 | 4809.50 | 4796.43 | 4787.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 4785.10 | 4794.91 | 4788.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 13:15:00 | 4847.00 | 4799.19 | 4791.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-01-01 14:15:00 | 5331.70 | 5294.46 | 5261.42 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-01-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 15:15:00 | 5298.50 | 5311.38 | 5312.63 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-01-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-08 09:15:00 | 5325.00 | 5314.10 | 5313.76 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 5245.00 | 5300.28 | 5307.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 5177.00 | 5275.62 | 5295.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 5296.50 | 5271.41 | 5285.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-09 09:15:00 | 5296.50 | 5271.41 | 5285.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 5296.50 | 5271.41 | 5285.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:45:00 | 5283.00 | 5271.41 | 5285.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 5274.00 | 5271.93 | 5284.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 15:15:00 | 5230.00 | 5264.88 | 5276.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 11:15:00 | 5241.00 | 5257.38 | 5269.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 13:15:00 | 5250.00 | 5254.92 | 5266.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-12 15:00:00 | 5242.50 | 5256.77 | 5265.19 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 5255.00 | 5253.41 | 5262.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:45:00 | 5191.00 | 5235.32 | 5252.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 4968.50 | 5075.91 | 5135.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 4978.95 | 5075.91 | 5135.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 4987.50 | 5075.91 | 5135.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 4980.38 | 5075.91 | 5135.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-16 09:15:00 | 4931.45 | 5075.91 | 5135.57 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-16 13:15:00 | 5055.00 | 5033.55 | 5092.32 | SL hit (close>ema200) qty=0.50 sl=5033.55 alert=retest2 |

### Cycle 57 — BUY (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 11:15:00 | 5214.00 | 5122.92 | 5117.06 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 5061.50 | 5115.10 | 5120.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 15:15:00 | 5022.50 | 5079.15 | 5100.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 5090.00 | 5066.59 | 5086.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 12:15:00 | 5090.00 | 5066.59 | 5086.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 5090.00 | 5066.59 | 5086.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 13:00:00 | 5090.00 | 5066.59 | 5086.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 5048.00 | 5062.88 | 5082.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-21 15:15:00 | 5010.00 | 5060.00 | 5079.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 09:30:00 | 5012.50 | 5037.40 | 5065.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 10:15:00 | 5094.00 | 5055.32 | 5056.16 | SL hit (close>static) qty=1.00 sl=5090.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 11:15:00 | 5070.00 | 5058.25 | 5057.42 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-01-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 14:15:00 | 5024.00 | 5056.77 | 5057.44 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 09:15:00 | 5158.00 | 5075.93 | 5065.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 10:15:00 | 5198.50 | 5100.45 | 5078.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-27 15:15:00 | 5153.50 | 5158.44 | 5120.33 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 10:00:00 | 5212.50 | 5169.25 | 5128.71 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-28 15:00:00 | 5240.00 | 5193.21 | 5156.27 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-29 11:00:00 | 5217.50 | 5204.02 | 5171.25 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 5182.00 | 5200.17 | 5177.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 13:30:00 | 5175.00 | 5200.17 | 5177.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 14:15:00 | 5173.00 | 5194.74 | 5177.32 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-29 14:15:00 | 5173.00 | 5194.74 | 5177.32 | SL hit (close<ema400) qty=1.00 sl=5177.32 alert=retest1 |

### Cycle 62 — SELL (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 11:15:00 | 5151.00 | 5167.65 | 5168.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 12:15:00 | 5133.50 | 5160.82 | 5165.44 | Break + close below crossover candle low |

### Cycle 63 — BUY (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 13:15:00 | 5240.00 | 5176.66 | 5172.21 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 5075.00 | 5159.92 | 5171.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 4980.00 | 5123.94 | 5153.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 5088.50 | 5066.63 | 5107.12 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 12:30:00 | 5083.00 | 5066.63 | 5107.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 5181.50 | 5089.60 | 5113.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 14:00:00 | 5181.50 | 5089.60 | 5113.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 5370.50 | 5145.78 | 5137.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 15:15:00 | 5399.50 | 5196.53 | 5161.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 15:15:00 | 5357.00 | 5374.08 | 5320.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-05 09:15:00 | 5365.00 | 5374.08 | 5320.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 5337.00 | 5366.66 | 5322.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 14:30:00 | 5406.50 | 5357.82 | 5335.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 15:00:00 | 5443.50 | 5388.39 | 5361.73 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 13:15:00 | 5344.00 | 5390.21 | 5396.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-02-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 13:15:00 | 5344.00 | 5390.21 | 5396.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-10 14:15:00 | 5322.00 | 5376.57 | 5389.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-11 10:15:00 | 5343.00 | 5342.44 | 5368.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 10:15:00 | 5343.00 | 5342.44 | 5368.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 5343.00 | 5342.44 | 5368.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:30:00 | 5350.00 | 5342.44 | 5368.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 14:15:00 | 5360.00 | 5335.41 | 5355.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-11 15:00:00 | 5360.00 | 5335.41 | 5355.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 15:15:00 | 5372.50 | 5342.83 | 5357.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-12 09:15:00 | 5283.50 | 5342.83 | 5357.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 14:15:00 | 5492.50 | 5379.60 | 5367.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-12 14:15:00 | 5492.50 | 5379.60 | 5367.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-12 15:15:00 | 5550.00 | 5413.68 | 5383.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 5281.00 | 5387.15 | 5374.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 5281.00 | 5387.15 | 5374.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 5281.00 | 5387.15 | 5374.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:30:00 | 5283.50 | 5387.15 | 5374.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 5270.50 | 5363.82 | 5365.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 09:15:00 | 5248.00 | 5319.64 | 5335.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-19 10:15:00 | 5204.00 | 5200.73 | 5233.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 15:15:00 | 5230.00 | 5202.41 | 5221.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 5230.00 | 5202.41 | 5221.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 09:15:00 | 5150.00 | 5202.41 | 5221.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-20 10:00:00 | 5135.00 | 5188.93 | 5213.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-20 11:15:00 | 5231.00 | 5201.84 | 5215.65 | SL hit (close>static) qty=1.00 sl=5230.00 alert=retest2 |

### Cycle 69 — BUY (started 2026-02-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-23 14:15:00 | 5379.50 | 5240.18 | 5225.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 14:15:00 | 5485.00 | 5308.85 | 5267.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 11:15:00 | 5350.00 | 5350.53 | 5304.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-25 12:00:00 | 5350.00 | 5350.53 | 5304.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 5470.00 | 5465.93 | 5417.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 10:15:00 | 5491.50 | 5465.93 | 5417.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 11:00:00 | 5482.50 | 5469.24 | 5423.72 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-27 11:30:00 | 5499.00 | 5478.80 | 5432.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 5329.00 | 5463.52 | 5445.57 | SL hit (close<static) qty=1.00 sl=5391.50 alert=retest2 |

### Cycle 70 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 5256.50 | 5406.27 | 5421.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 5216.00 | 5368.22 | 5402.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 12:15:00 | 5315.00 | 5278.66 | 5328.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-04 13:00:00 | 5315.00 | 5278.66 | 5328.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 13:15:00 | 5478.50 | 5318.63 | 5342.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 14:00:00 | 5478.50 | 5318.63 | 5342.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 14:15:00 | 5428.00 | 5340.50 | 5350.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-04 15:15:00 | 5397.00 | 5340.50 | 5350.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 09:15:00 | 5498.00 | 5381.04 | 5367.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 09:15:00 | 5498.00 | 5381.04 | 5367.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 14:15:00 | 5590.00 | 5467.26 | 5418.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 10:15:00 | 5443.00 | 5484.24 | 5440.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 10:15:00 | 5443.00 | 5484.24 | 5440.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 5443.00 | 5484.24 | 5440.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 11:00:00 | 5443.00 | 5484.24 | 5440.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 11:15:00 | 5462.00 | 5479.79 | 5442.61 | EMA400 retest candle locked (from upside) |

### Cycle 72 — SELL (started 2026-03-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-06 15:15:00 | 5349.50 | 5418.47 | 5422.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 09:15:00 | 5231.00 | 5380.97 | 5405.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-09 14:15:00 | 5480.00 | 5315.77 | 5352.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 14:15:00 | 5480.00 | 5315.77 | 5352.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 14:15:00 | 5480.00 | 5315.77 | 5352.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-09 15:00:00 | 5480.00 | 5315.77 | 5352.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 15:15:00 | 5410.00 | 5334.62 | 5357.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:30:00 | 5419.50 | 5353.59 | 5364.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 5505.50 | 5383.98 | 5377.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 11:15:00 | 5517.50 | 5410.68 | 5390.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-10 15:15:00 | 5445.00 | 5446.07 | 5416.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 09:15:00 | 5429.50 | 5446.07 | 5416.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 09:15:00 | 5431.50 | 5443.15 | 5418.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 09:30:00 | 5396.50 | 5443.15 | 5418.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 10:15:00 | 5373.00 | 5429.12 | 5414.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:00:00 | 5373.00 | 5429.12 | 5414.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 5362.00 | 5415.70 | 5409.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:30:00 | 5370.50 | 5415.70 | 5409.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 5357.50 | 5404.06 | 5404.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 13:15:00 | 5316.50 | 5386.55 | 5396.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 12:15:00 | 5339.50 | 5338.80 | 5363.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-12 12:30:00 | 5335.50 | 5338.80 | 5363.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 5345.50 | 5338.66 | 5358.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 15:00:00 | 5345.50 | 5338.66 | 5358.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 5301.00 | 5331.12 | 5353.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 5270.50 | 5331.12 | 5353.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-17 09:15:00 | 5006.97 | 5066.22 | 5150.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 5020.50 | 5014.51 | 5094.32 | SL hit (close>ema200) qty=0.50 sl=5014.51 alert=retest2 |

### Cycle 75 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 4750.50 | 4688.53 | 4688.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 4793.00 | 4720.94 | 4703.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 4617.00 | 4720.84 | 4711.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 4617.00 | 4720.84 | 4711.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 4617.00 | 4720.84 | 4711.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 4617.00 | 4720.84 | 4711.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 4606.00 | 4697.87 | 4701.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 4592.50 | 4676.80 | 4691.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 4467.00 | 4462.84 | 4538.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 10:15:00 | 4538.30 | 4477.93 | 4538.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 4538.30 | 4477.93 | 4538.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:00:00 | 4538.30 | 4477.93 | 4538.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 4535.20 | 4489.38 | 4538.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 12:00:00 | 4535.20 | 4489.38 | 4538.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 4579.90 | 4507.49 | 4541.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 13:00:00 | 4579.90 | 4507.49 | 4541.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 13:15:00 | 4564.30 | 4518.85 | 4543.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 4470.50 | 4541.68 | 4550.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 14:45:00 | 4516.50 | 4507.27 | 4522.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 4519.30 | 4521.61 | 4527.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 10:00:00 | 4521.80 | 4521.65 | 4526.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 4513.00 | 4519.92 | 4525.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 11:00:00 | 4513.00 | 4519.92 | 4525.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 13:15:00 | 4522.10 | 4512.77 | 4520.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 14:00:00 | 4522.10 | 4512.77 | 4520.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-04-06 14:15:00 | 4591.70 | 4528.55 | 4526.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 4591.70 | 4528.55 | 4526.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 14:15:00 | 4683.10 | 4597.12 | 4565.29 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 4804.30 | 4831.52 | 4738.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 4804.30 | 4831.52 | 4738.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 4830.00 | 4904.74 | 4865.73 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 14:15:00 | 4801.60 | 4839.81 | 4844.26 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2026-04-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 14:15:00 | 4922.00 | 4854.12 | 4846.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 09:15:00 | 5040.00 | 4898.59 | 4868.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 15:15:00 | 5000.00 | 5008.51 | 4986.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-21 09:15:00 | 5011.30 | 5008.51 | 4986.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 5019.30 | 5010.67 | 4989.53 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 4916.30 | 4973.57 | 4979.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 10:15:00 | 4909.10 | 4960.68 | 4972.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-22 13:15:00 | 4952.30 | 4950.91 | 4964.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-22 14:00:00 | 4952.30 | 4950.91 | 4964.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 4611.00 | 4561.47 | 4597.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:45:00 | 4617.40 | 4561.47 | 4597.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 4637.30 | 4576.64 | 4601.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:00:00 | 4637.30 | 4576.64 | 4601.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 4641.10 | 4589.53 | 4605.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 12:00:00 | 4641.10 | 4589.53 | 4605.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 4601.60 | 4565.25 | 4579.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:00:00 | 4601.60 | 4565.25 | 4579.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 4626.00 | 4577.40 | 4583.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:45:00 | 4611.60 | 4577.40 | 4583.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 4669.80 | 4603.08 | 4594.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 12:15:00 | 4678.50 | 4636.39 | 4613.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 15:15:00 | 4712.70 | 4723.63 | 4685.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 09:15:00 | 4733.50 | 4723.63 | 4685.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 13:15:00 | 4703.30 | 4723.26 | 4700.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:45:00 | 4702.30 | 4723.26 | 4700.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 4722.00 | 4723.00 | 4702.26 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 12:00:00 | 5018.00 | 2025-05-14 10:15:00 | 5141.00 | STOP_HIT | 1.00 | -2.45% |
| SELL | retest2 | 2025-05-13 09:15:00 | 5045.00 | 2025-05-14 10:15:00 | 5141.00 | STOP_HIT | 1.00 | -1.90% |
| SELL | retest2 | 2025-05-13 09:45:00 | 5029.50 | 2025-05-14 10:15:00 | 5141.00 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-05-23 11:15:00 | 5198.00 | 2025-05-28 09:15:00 | 5114.00 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2025-05-26 11:15:00 | 5201.50 | 2025-05-28 09:15:00 | 5114.00 | STOP_HIT | 1.00 | -1.68% |
| BUY | retest2 | 2025-05-26 14:30:00 | 5344.50 | 2025-05-28 09:15:00 | 5114.00 | STOP_HIT | 1.00 | -4.31% |
| SELL | retest2 | 2025-05-30 10:30:00 | 5062.00 | 2025-05-30 14:15:00 | 5147.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-05-30 13:00:00 | 5072.00 | 2025-05-30 14:15:00 | 5147.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-06-04 14:15:00 | 5078.00 | 2025-06-06 14:15:00 | 5075.00 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-06-04 15:00:00 | 5058.50 | 2025-06-06 14:15:00 | 5075.00 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-06-06 13:00:00 | 5075.50 | 2025-06-06 14:15:00 | 5075.00 | STOP_HIT | 1.00 | 0.01% |
| SELL | retest2 | 2025-06-17 14:30:00 | 4908.00 | 2025-06-23 11:15:00 | 4860.00 | STOP_HIT | 1.00 | 0.98% |
| BUY | retest1 | 2025-06-25 09:15:00 | 4937.50 | 2025-06-26 09:15:00 | 4889.50 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest1 | 2025-06-25 10:15:00 | 4928.50 | 2025-06-26 09:15:00 | 4889.50 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-07-03 09:15:00 | 5114.50 | 2025-07-09 10:15:00 | 5074.50 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-07-09 10:15:00 | 5076.00 | 2025-07-09 10:15:00 | 5074.50 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-07-11 09:15:00 | 4985.00 | 2025-07-15 10:15:00 | 5060.00 | STOP_HIT | 1.00 | -1.50% |
| SELL | retest2 | 2025-07-14 14:15:00 | 5038.00 | 2025-07-15 10:15:00 | 5060.00 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-07-14 15:00:00 | 5042.00 | 2025-07-15 10:15:00 | 5060.00 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-07-15 09:30:00 | 5044.00 | 2025-07-15 10:15:00 | 5060.00 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-07-17 09:15:00 | 5062.50 | 2025-07-24 10:15:00 | 5131.50 | STOP_HIT | 1.00 | 1.36% |
| SELL | retest2 | 2025-08-13 14:45:00 | 4482.50 | 2025-08-14 13:15:00 | 4539.20 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-08-26 12:15:00 | 5035.10 | 2025-08-26 15:15:00 | 4783.35 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 12:15:00 | 5035.10 | 2025-08-29 09:15:00 | 4531.59 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-09-24 09:45:00 | 4675.00 | 2025-09-25 13:15:00 | 4613.80 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-09-24 12:45:00 | 4655.60 | 2025-09-25 13:15:00 | 4613.80 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-09-25 09:30:00 | 4669.90 | 2025-09-25 13:15:00 | 4613.80 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-10-14 13:45:00 | 4482.90 | 2025-10-20 10:15:00 | 4455.70 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-10-15 09:30:00 | 4465.00 | 2025-10-20 10:15:00 | 4455.70 | STOP_HIT | 1.00 | -0.21% |
| BUY | retest2 | 2025-10-15 10:00:00 | 4466.70 | 2025-10-20 10:15:00 | 4455.70 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-10-15 11:15:00 | 4465.40 | 2025-10-20 10:15:00 | 4455.70 | STOP_HIT | 1.00 | -0.22% |
| BUY | retest1 | 2025-10-28 09:30:00 | 4711.20 | 2025-10-29 09:15:00 | 4626.80 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest1 | 2025-10-31 15:15:00 | 4542.00 | 2025-11-03 09:15:00 | 4614.90 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-11-10 10:15:00 | 4815.00 | 2025-11-11 15:15:00 | 4778.10 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-11-11 09:15:00 | 4826.10 | 2025-11-11 15:15:00 | 4778.10 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-11-24 12:30:00 | 4747.40 | 2025-11-27 09:15:00 | 4670.80 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-11-24 13:15:00 | 4739.40 | 2025-11-27 09:15:00 | 4670.80 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-11-24 13:45:00 | 4737.40 | 2025-11-27 09:15:00 | 4670.80 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2025-11-24 14:45:00 | 4740.90 | 2025-11-27 09:15:00 | 4670.80 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2025-11-25 13:00:00 | 4745.00 | 2025-11-27 09:15:00 | 4670.80 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-11-25 15:00:00 | 4757.50 | 2025-11-27 09:15:00 | 4670.80 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-11-26 09:15:00 | 4759.30 | 2025-11-27 09:15:00 | 4670.80 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-11-26 13:45:00 | 4755.80 | 2025-11-27 09:15:00 | 4670.80 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-12-11 10:45:00 | 4670.00 | 2025-12-12 10:15:00 | 4794.50 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2025-12-15 10:15:00 | 4782.60 | 2025-12-16 15:15:00 | 4732.50 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-12-16 15:00:00 | 4773.60 | 2025-12-16 15:15:00 | 4732.50 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-12-19 13:15:00 | 4847.00 | 2026-01-01 14:15:00 | 5331.70 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-01-09 15:15:00 | 5230.00 | 2026-01-16 09:15:00 | 4968.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-12 11:15:00 | 5241.00 | 2026-01-16 09:15:00 | 4978.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-12 13:15:00 | 5250.00 | 2026-01-16 09:15:00 | 4987.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-12 15:00:00 | 5242.50 | 2026-01-16 09:15:00 | 4980.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 11:45:00 | 5191.00 | 2026-01-16 09:15:00 | 4931.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-09 15:15:00 | 5230.00 | 2026-01-16 13:15:00 | 5055.00 | STOP_HIT | 0.50 | 3.35% |
| SELL | retest2 | 2026-01-12 11:15:00 | 5241.00 | 2026-01-16 13:15:00 | 5055.00 | STOP_HIT | 0.50 | 3.55% |
| SELL | retest2 | 2026-01-12 13:15:00 | 5250.00 | 2026-01-16 13:15:00 | 5055.00 | STOP_HIT | 0.50 | 3.71% |
| SELL | retest2 | 2026-01-12 15:00:00 | 5242.50 | 2026-01-16 13:15:00 | 5055.00 | STOP_HIT | 0.50 | 3.58% |
| SELL | retest2 | 2026-01-13 11:45:00 | 5191.00 | 2026-01-16 13:15:00 | 5055.00 | STOP_HIT | 0.50 | 2.62% |
| SELL | retest2 | 2026-01-21 15:15:00 | 5010.00 | 2026-01-23 10:15:00 | 5094.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-01-22 09:30:00 | 5012.50 | 2026-01-23 10:15:00 | 5094.00 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest1 | 2026-01-28 10:00:00 | 5212.50 | 2026-01-29 14:15:00 | 5173.00 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest1 | 2026-01-28 15:00:00 | 5240.00 | 2026-01-29 14:15:00 | 5173.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest1 | 2026-01-29 11:00:00 | 5217.50 | 2026-01-29 14:15:00 | 5173.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2026-01-30 10:15:00 | 5167.00 | 2026-01-30 11:15:00 | 5151.00 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2026-01-30 11:15:00 | 5153.00 | 2026-01-30 11:15:00 | 5151.00 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2026-02-05 14:30:00 | 5406.50 | 2026-02-10 13:15:00 | 5344.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2026-02-06 15:00:00 | 5443.50 | 2026-02-10 13:15:00 | 5344.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2026-02-12 09:15:00 | 5283.50 | 2026-02-12 14:15:00 | 5492.50 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2026-02-20 09:15:00 | 5150.00 | 2026-02-20 11:15:00 | 5231.00 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2026-02-20 10:00:00 | 5135.00 | 2026-02-20 11:15:00 | 5231.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2026-02-27 10:15:00 | 5491.50 | 2026-03-02 09:15:00 | 5329.00 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2026-02-27 11:00:00 | 5482.50 | 2026-03-02 09:15:00 | 5329.00 | STOP_HIT | 1.00 | -2.80% |
| BUY | retest2 | 2026-02-27 11:30:00 | 5499.00 | 2026-03-02 09:15:00 | 5329.00 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2026-03-04 15:15:00 | 5397.00 | 2026-03-05 09:15:00 | 5498.00 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2026-03-13 09:15:00 | 5270.50 | 2026-03-17 09:15:00 | 5006.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-13 09:15:00 | 5270.50 | 2026-03-17 13:15:00 | 5020.50 | STOP_HIT | 0.50 | 4.74% |
| SELL | retest2 | 2026-04-02 09:15:00 | 4470.50 | 2026-04-06 14:15:00 | 4591.70 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2026-04-02 14:45:00 | 4516.50 | 2026-04-06 14:15:00 | 4591.70 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2026-04-06 09:15:00 | 4519.30 | 2026-04-06 14:15:00 | 4591.70 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2026-04-06 10:00:00 | 4521.80 | 2026-04-06 14:15:00 | 4591.70 | STOP_HIT | 1.00 | -1.55% |
