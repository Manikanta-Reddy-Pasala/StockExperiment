# Alkem Laboratories Ltd. (ALKEM)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 5560.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 142 |
| ALERT1 | 97 |
| ALERT2 | 94 |
| ALERT2_SKIP | 40 |
| ALERT3 | 262 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 6 |
| ENTRY2 | 141 |
| PARTIAL | 2 |
| TARGET_HIT | 2 |
| STOP_HIT | 147 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 149 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 32 / 117
- **Target hits / Stop hits / Partials:** 2 / 145 / 2
- **Avg / median % per leg:** -0.45% / -0.90%
- **Sum % (uncompounded):** -66.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 76 | 24 | 31.6% | 2 | 74 | 0 | -0.02% | -1.2% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.00% | -4.0% |
| BUY @ 3rd Alert (retest2) | 72 | 24 | 33.3% | 2 | 70 | 0 | 0.04% | 2.8% |
| SELL (all) | 73 | 8 | 11.0% | 0 | 71 | 2 | -0.90% | -65.4% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.07% | -0.1% |
| SELL @ 3rd Alert (retest2) | 71 | 7 | 9.9% | 0 | 69 | 2 | -0.92% | -65.3% |
| retest1 (combined) | 6 | 1 | 16.7% | 0 | 6 | 0 | -0.69% | -4.1% |
| retest2 (combined) | 143 | 31 | 21.7% | 2 | 139 | 2 | -0.44% | -62.5% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 09:15:00 | 5237.70 | 5341.48 | 5343.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 11:15:00 | 5230.50 | 5306.11 | 5325.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-22 14:15:00 | 5306.90 | 5298.60 | 5316.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-05-22 15:00:00 | 5306.90 | 5298.60 | 5316.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 15:15:00 | 5285.10 | 5295.90 | 5313.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 09:15:00 | 5296.60 | 5295.90 | 5313.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 5285.40 | 5293.80 | 5311.29 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2024-05-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 11:15:00 | 5427.10 | 5321.96 | 5321.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 12:15:00 | 5447.95 | 5347.15 | 5332.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 09:15:00 | 5392.45 | 5394.45 | 5363.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 5392.45 | 5394.45 | 5363.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 5392.45 | 5394.45 | 5363.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 11:45:00 | 5472.80 | 5416.75 | 5379.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 13:00:00 | 5470.00 | 5427.40 | 5387.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-27 12:15:00 | 5325.50 | 5390.86 | 5389.43 | SL hit (close<static) qty=1.00 sl=5355.10 alert=retest2 |

### Cycle 3 — SELL (started 2024-05-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 13:15:00 | 5329.00 | 5378.49 | 5383.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-29 14:15:00 | 5270.85 | 5315.66 | 5333.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 13:15:00 | 4771.15 | 4740.30 | 4827.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-04 14:00:00 | 4771.15 | 4740.30 | 4827.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 4835.60 | 4754.07 | 4811.82 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2024-06-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 13:15:00 | 4863.05 | 4824.90 | 4822.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 14:15:00 | 4870.15 | 4833.95 | 4826.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-13 10:15:00 | 5060.20 | 5072.30 | 5044.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-13 11:00:00 | 5060.20 | 5072.30 | 5044.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 09:15:00 | 5084.80 | 5081.74 | 5060.93 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2024-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 10:15:00 | 4943.75 | 5040.47 | 5053.37 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 14:15:00 | 5214.40 | 5084.20 | 5068.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 12:15:00 | 5251.90 | 5175.15 | 5125.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 14:15:00 | 5178.95 | 5181.99 | 5137.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-19 15:00:00 | 5178.95 | 5181.99 | 5137.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 09:15:00 | 5148.25 | 5176.41 | 5142.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-20 11:45:00 | 5186.50 | 5180.05 | 5149.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-21 11:15:00 | 5127.90 | 5157.80 | 5152.51 | SL hit (close<static) qty=1.00 sl=5137.70 alert=retest2 |

### Cycle 7 — SELL (started 2024-06-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 14:15:00 | 5126.90 | 5145.81 | 5147.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 15:15:00 | 5104.75 | 5137.60 | 5143.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-25 09:15:00 | 5062.25 | 5045.30 | 5081.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-25 09:15:00 | 5062.25 | 5045.30 | 5081.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 5062.25 | 5045.30 | 5081.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 09:15:00 | 5006.15 | 5081.24 | 5087.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 13:00:00 | 5027.15 | 5057.62 | 5072.91 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-03 11:15:00 | 5029.00 | 4961.27 | 4958.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2024-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 11:15:00 | 5029.00 | 4961.27 | 4958.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 11:15:00 | 5071.95 | 5007.81 | 4989.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-10 09:15:00 | 5232.25 | 5278.57 | 5208.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-10 10:00:00 | 5232.25 | 5278.57 | 5208.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 5184.90 | 5259.84 | 5206.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 5200.10 | 5259.84 | 5206.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 11:15:00 | 5196.95 | 5247.26 | 5205.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 12:30:00 | 5225.00 | 5241.88 | 5206.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 13:45:00 | 5220.20 | 5237.10 | 5207.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-10 14:15:00 | 5221.55 | 5237.10 | 5207.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-11 14:30:00 | 5231.15 | 5222.43 | 5215.93 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 15:15:00 | 5208.85 | 5219.71 | 5215.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 09:15:00 | 5252.95 | 5219.71 | 5215.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-12 10:45:00 | 5212.85 | 5214.78 | 5213.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 11:15:00 | 5196.75 | 5211.17 | 5212.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-07-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-12 11:15:00 | 5196.75 | 5211.17 | 5212.09 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-15 09:15:00 | 5380.00 | 5238.23 | 5223.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 11:15:00 | 5408.10 | 5293.38 | 5252.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 11:15:00 | 5338.65 | 5357.58 | 5312.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-16 11:45:00 | 5342.10 | 5357.58 | 5312.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 13:15:00 | 5304.95 | 5341.83 | 5313.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 13:30:00 | 5313.10 | 5341.83 | 5313.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 14:15:00 | 5300.20 | 5333.51 | 5311.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 14:30:00 | 5296.55 | 5333.51 | 5311.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 5294.00 | 5325.61 | 5310.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 5285.00 | 5325.61 | 5310.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 5234.55 | 5296.48 | 5299.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 5192.70 | 5251.41 | 5272.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 5216.65 | 5188.05 | 5222.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-22 10:00:00 | 5216.65 | 5188.05 | 5222.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 5185.00 | 5187.44 | 5219.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 5203.40 | 5187.44 | 5219.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 5220.00 | 5195.84 | 5217.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:45:00 | 5217.75 | 5195.84 | 5217.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 5222.10 | 5201.09 | 5218.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:45:00 | 5232.20 | 5201.09 | 5218.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 15:15:00 | 5230.00 | 5209.60 | 5219.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:15:00 | 5199.15 | 5209.60 | 5219.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 09:15:00 | 5218.75 | 5211.43 | 5219.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 09:45:00 | 5233.00 | 5211.43 | 5219.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 10:15:00 | 5197.35 | 5208.62 | 5217.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 12:15:00 | 5111.55 | 5207.95 | 5216.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 13:30:00 | 5157.65 | 5201.00 | 5211.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 09:15:00 | 5231.15 | 5204.04 | 5209.68 | SL hit (close>static) qty=1.00 sl=5218.75 alert=retest2 |

### Cycle 12 — BUY (started 2024-07-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 10:15:00 | 5259.55 | 5215.14 | 5214.22 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 14:15:00 | 5199.05 | 5213.41 | 5214.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-25 09:15:00 | 5153.85 | 5197.07 | 5206.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-25 12:15:00 | 5196.90 | 5192.77 | 5201.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-25 12:15:00 | 5196.90 | 5192.77 | 5201.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 12:15:00 | 5196.90 | 5192.77 | 5201.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-25 13:00:00 | 5196.90 | 5192.77 | 5201.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-25 13:15:00 | 5173.05 | 5188.82 | 5199.25 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 09:15:00 | 5293.10 | 5207.06 | 5204.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-26 15:15:00 | 5314.00 | 5269.14 | 5241.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-29 10:15:00 | 5268.05 | 5272.42 | 5247.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-29 10:45:00 | 5278.70 | 5272.42 | 5247.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 11:15:00 | 5234.40 | 5264.82 | 5246.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 12:00:00 | 5234.40 | 5264.82 | 5246.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 12:15:00 | 5228.20 | 5257.49 | 5244.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 13:15:00 | 5226.10 | 5257.49 | 5244.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 13:15:00 | 5208.45 | 5247.68 | 5241.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-29 14:00:00 | 5208.45 | 5247.68 | 5241.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — SELL (started 2024-07-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 14:15:00 | 5193.60 | 5236.87 | 5237.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-30 09:15:00 | 5187.00 | 5222.44 | 5230.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-30 13:15:00 | 5233.20 | 5221.16 | 5226.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 13:15:00 | 5233.20 | 5221.16 | 5226.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 13:15:00 | 5233.20 | 5221.16 | 5226.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 13:45:00 | 5227.15 | 5221.16 | 5226.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 14:15:00 | 5221.70 | 5221.26 | 5226.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-30 15:15:00 | 5205.00 | 5221.26 | 5226.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-31 09:15:00 | 5290.00 | 5232.41 | 5230.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — BUY (started 2024-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 09:15:00 | 5290.00 | 5232.41 | 5230.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 13:15:00 | 5333.05 | 5273.90 | 5252.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 11:15:00 | 5265.25 | 5289.21 | 5269.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 11:15:00 | 5265.25 | 5289.21 | 5269.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 5265.25 | 5289.21 | 5269.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 12:00:00 | 5265.25 | 5289.21 | 5269.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 5310.10 | 5293.38 | 5273.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 13:45:00 | 5324.80 | 5293.30 | 5281.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-02 14:15:00 | 5323.65 | 5293.30 | 5281.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-05 10:15:00 | 5255.90 | 5289.30 | 5283.86 | SL hit (close<static) qty=1.00 sl=5257.70 alert=retest2 |

### Cycle 17 — SELL (started 2024-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-16 09:15:00 | 5650.60 | 5692.97 | 5697.19 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2024-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 11:15:00 | 5726.10 | 5689.45 | 5688.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-19 12:15:00 | 5745.25 | 5700.61 | 5694.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 10:15:00 | 5712.05 | 5722.28 | 5709.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 11:00:00 | 5712.05 | 5722.28 | 5709.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 5738.45 | 5729.24 | 5714.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 13:00:00 | 5738.45 | 5729.24 | 5714.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 14:15:00 | 5732.55 | 5729.02 | 5717.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:30:00 | 5761.90 | 5739.24 | 5724.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:30:00 | 5776.30 | 5764.35 | 5761.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2024-09-05 10:15:00 | 6338.09 | 6215.74 | 6167.58 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2024-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-17 09:15:00 | 6267.45 | 6343.96 | 6346.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-17 11:15:00 | 6221.30 | 6306.78 | 6328.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 09:15:00 | 6143.25 | 6139.62 | 6201.00 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-09-19 12:45:00 | 6115.70 | 6132.10 | 6182.36 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-20 09:15:00 | 6124.20 | 6129.59 | 6164.66 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-20 14:30:00 | 6055.70 | 6119.39 | 6147.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-23 10:00:00 | 6093.35 | 6110.99 | 6138.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-24 10:15:00 | 6071.50 | 6031.70 | 6069.55 | SL hit (close>ema400) qty=1.00 sl=6069.55 alert=retest1 |

### Cycle 20 — BUY (started 2024-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-25 11:15:00 | 6110.05 | 6073.08 | 6072.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-25 12:15:00 | 6135.00 | 6085.46 | 6078.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-26 09:15:00 | 6146.05 | 6150.74 | 6116.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-26 10:00:00 | 6146.05 | 6150.74 | 6116.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 6060.00 | 6128.79 | 6112.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 12:00:00 | 6060.00 | 6128.79 | 6112.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 12:15:00 | 6063.30 | 6115.69 | 6107.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 13:00:00 | 6063.30 | 6115.69 | 6107.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 6200.45 | 6133.85 | 6117.61 | EMA400 retest candle locked (from upside) |

### Cycle 21 — SELL (started 2024-09-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 13:15:00 | 6089.00 | 6132.04 | 6135.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 14:15:00 | 6051.00 | 6115.83 | 6127.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 12:15:00 | 6104.80 | 6094.40 | 6110.56 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-01 12:30:00 | 6105.65 | 6094.40 | 6110.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 6147.35 | 6104.99 | 6113.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 14:00:00 | 6147.35 | 6104.99 | 6113.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 6157.25 | 6115.44 | 6117.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 15:00:00 | 6157.25 | 6115.44 | 6117.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2024-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-03 10:15:00 | 6160.15 | 6123.02 | 6120.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-04 11:15:00 | 6234.30 | 6173.08 | 6150.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-04 14:15:00 | 6183.00 | 6210.84 | 6176.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-04 14:15:00 | 6183.00 | 6210.84 | 6176.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 6183.00 | 6210.84 | 6176.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-04 15:00:00 | 6183.00 | 6210.84 | 6176.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 15:15:00 | 6200.00 | 6208.67 | 6178.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:15:00 | 6156.30 | 6208.67 | 6178.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 6150.60 | 6197.06 | 6176.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 10:00:00 | 6150.60 | 6197.06 | 6176.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 10:15:00 | 6198.95 | 6197.44 | 6178.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 10:30:00 | 6120.00 | 6197.44 | 6178.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 11:15:00 | 6183.90 | 6194.73 | 6178.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 12:00:00 | 6183.90 | 6194.73 | 6178.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 12:15:00 | 6128.10 | 6181.40 | 6174.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 13:00:00 | 6128.10 | 6181.40 | 6174.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 13:15:00 | 6128.55 | 6170.83 | 6170.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-07 13:45:00 | 6144.00 | 6170.83 | 6170.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — SELL (started 2024-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-07 14:15:00 | 6147.35 | 6166.14 | 6167.99 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2024-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 10:15:00 | 6251.15 | 6177.05 | 6171.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-08 14:15:00 | 6260.00 | 6214.17 | 6192.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-10 09:15:00 | 6258.05 | 6280.98 | 6250.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 10:00:00 | 6258.05 | 6280.98 | 6250.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 6220.05 | 6268.79 | 6248.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:00:00 | 6220.05 | 6268.79 | 6248.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 6143.30 | 6243.69 | 6238.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 12:00:00 | 6143.30 | 6243.69 | 6238.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2024-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 12:15:00 | 6096.10 | 6214.17 | 6225.59 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2024-10-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 14:15:00 | 6253.85 | 6205.45 | 6201.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-15 09:15:00 | 6261.80 | 6223.80 | 6211.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 15:15:00 | 6251.05 | 6254.35 | 6235.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-16 09:15:00 | 6217.35 | 6254.35 | 6235.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 09:15:00 | 6134.60 | 6230.40 | 6226.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 10:00:00 | 6134.60 | 6230.40 | 6226.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2024-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 10:15:00 | 6125.10 | 6209.34 | 6217.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 11:15:00 | 6059.10 | 6179.29 | 6202.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 09:15:00 | 6078.45 | 6060.07 | 6105.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-18 09:15:00 | 6078.45 | 6060.07 | 6105.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 09:15:00 | 6078.45 | 6060.07 | 6105.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 09:30:00 | 6105.00 | 6060.07 | 6105.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 13:15:00 | 6136.25 | 6073.08 | 6096.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 13:45:00 | 6151.75 | 6073.08 | 6096.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-18 14:15:00 | 6129.30 | 6084.32 | 6099.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-18 15:15:00 | 6154.00 | 6084.32 | 6099.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — BUY (started 2024-10-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-21 10:15:00 | 6234.25 | 6133.22 | 6120.03 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2024-10-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-22 15:15:00 | 6111.05 | 6133.02 | 6134.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-23 11:15:00 | 6076.90 | 6119.00 | 6128.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-24 14:15:00 | 5955.95 | 5954.67 | 6007.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-24 15:00:00 | 5955.95 | 5954.67 | 6007.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 09:15:00 | 5947.10 | 5951.00 | 5996.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-25 13:45:00 | 5908.95 | 5940.45 | 5977.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-28 10:15:00 | 6016.70 | 5961.27 | 5975.82 | SL hit (close>static) qty=1.00 sl=6008.25 alert=retest2 |

### Cycle 30 — BUY (started 2024-10-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 12:15:00 | 6034.55 | 5985.32 | 5984.81 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 09:15:00 | 5891.80 | 5988.14 | 5989.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-29 10:15:00 | 5873.60 | 5965.23 | 5978.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-29 14:15:00 | 5925.00 | 5914.62 | 5946.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-29 15:00:00 | 5925.00 | 5914.62 | 5946.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-30 09:15:00 | 5910.40 | 5917.38 | 5942.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-30 10:30:00 | 5886.60 | 5903.78 | 5933.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-06 13:15:00 | 5828.55 | 5757.27 | 5753.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2024-11-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 13:15:00 | 5828.55 | 5757.27 | 5753.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 14:15:00 | 5849.85 | 5775.78 | 5762.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 5738.15 | 5780.69 | 5767.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-07 09:15:00 | 5738.15 | 5780.69 | 5767.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 09:15:00 | 5738.15 | 5780.69 | 5767.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:00:00 | 5738.15 | 5780.69 | 5767.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 5734.95 | 5771.54 | 5764.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:45:00 | 5717.50 | 5771.54 | 5764.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 5723.30 | 5761.89 | 5761.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:00:00 | 5723.30 | 5761.89 | 5761.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2024-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 12:15:00 | 5695.70 | 5748.66 | 5755.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 11:15:00 | 5591.55 | 5684.76 | 5712.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 12:15:00 | 5510.70 | 5490.75 | 5561.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-13 13:00:00 | 5510.70 | 5490.75 | 5561.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 13:15:00 | 5586.00 | 5509.80 | 5563.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-13 13:45:00 | 5570.95 | 5509.80 | 5563.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-13 14:15:00 | 5564.05 | 5520.65 | 5563.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 10:00:00 | 5512.65 | 5526.17 | 5559.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 11:45:00 | 5516.90 | 5532.36 | 5556.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 14:45:00 | 5521.95 | 5537.51 | 5553.43 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 10:15:00 | 5519.50 | 5537.32 | 5550.32 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 10:15:00 | 5552.20 | 5540.30 | 5550.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 10:45:00 | 5572.00 | 5540.30 | 5550.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 11:15:00 | 5577.40 | 5547.72 | 5552.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:00:00 | 5577.40 | 5547.72 | 5552.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 12:15:00 | 5577.10 | 5553.60 | 5555.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-18 12:30:00 | 5580.65 | 5553.60 | 5555.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 10:15:00 | 5561.95 | 5540.70 | 5546.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:00:00 | 5561.95 | 5540.70 | 5546.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 11:15:00 | 5586.10 | 5549.78 | 5550.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-19 11:45:00 | 5585.15 | 5549.78 | 5550.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-19 12:15:00 | 5577.85 | 5555.39 | 5552.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2024-11-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-19 12:15:00 | 5577.85 | 5555.39 | 5552.60 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2024-11-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-19 15:15:00 | 5534.00 | 5547.97 | 5549.75 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2024-11-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-21 09:15:00 | 5609.65 | 5560.31 | 5555.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-21 10:15:00 | 5659.65 | 5580.18 | 5564.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 12:15:00 | 5553.15 | 5580.98 | 5568.13 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 12:15:00 | 5553.15 | 5580.98 | 5568.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 12:15:00 | 5553.15 | 5580.98 | 5568.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 13:00:00 | 5553.15 | 5580.98 | 5568.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 13:15:00 | 5570.90 | 5578.97 | 5568.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-21 14:15:00 | 5586.35 | 5578.97 | 5568.39 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 10:15:00 | 5529.00 | 5570.56 | 5568.78 | SL hit (close<static) qty=1.00 sl=5541.55 alert=retest2 |

### Cycle 37 — SELL (started 2024-11-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-22 11:15:00 | 5536.40 | 5563.73 | 5565.84 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2024-11-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 15:15:00 | 5606.30 | 5569.55 | 5567.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 5650.00 | 5585.64 | 5574.88 | Break + close above crossover candle high |

### Cycle 39 — SELL (started 2024-11-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-25 14:15:00 | 5482.60 | 5577.55 | 5577.66 | EMA200 below EMA400 |

### Cycle 40 — BUY (started 2024-11-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 10:15:00 | 5591.55 | 5498.21 | 5486.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 11:15:00 | 5609.30 | 5520.43 | 5497.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-03 09:15:00 | 5660.05 | 5674.44 | 5623.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-03 10:00:00 | 5660.05 | 5674.44 | 5623.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 11:15:00 | 5640.35 | 5661.46 | 5626.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 11:45:00 | 5637.95 | 5661.46 | 5626.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 12:15:00 | 5639.70 | 5657.11 | 5627.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-03 12:30:00 | 5628.20 | 5657.11 | 5627.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 13:15:00 | 5675.00 | 5663.37 | 5645.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-04 14:15:00 | 5660.50 | 5663.37 | 5645.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 14:15:00 | 5678.10 | 5666.31 | 5648.50 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 09:15:00 | 5571.25 | 5630.23 | 5637.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 10:15:00 | 5531.80 | 5610.55 | 5627.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-11 09:15:00 | 5482.80 | 5472.99 | 5501.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-11 09:15:00 | 5482.80 | 5472.99 | 5501.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 09:15:00 | 5482.80 | 5472.99 | 5501.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 09:30:00 | 5511.55 | 5472.99 | 5501.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 10:15:00 | 5499.90 | 5478.37 | 5501.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 10:45:00 | 5498.35 | 5478.37 | 5501.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 11:15:00 | 5492.35 | 5481.17 | 5500.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 11:30:00 | 5498.50 | 5481.17 | 5500.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 12:15:00 | 5510.70 | 5487.07 | 5501.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-11 13:00:00 | 5510.70 | 5487.07 | 5501.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 13:15:00 | 5486.85 | 5487.03 | 5500.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-12 09:15:00 | 5449.30 | 5488.11 | 5498.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-18 09:15:00 | 5502.00 | 5414.41 | 5413.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2024-12-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 09:15:00 | 5502.00 | 5414.41 | 5413.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-18 10:15:00 | 5533.60 | 5438.25 | 5424.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-19 09:15:00 | 5470.65 | 5476.80 | 5453.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-19 09:15:00 | 5470.65 | 5476.80 | 5453.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 5470.65 | 5476.80 | 5453.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 10:30:00 | 5502.45 | 5481.98 | 5458.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 15:00:00 | 5501.95 | 5498.99 | 5475.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 09:30:00 | 5504.60 | 5496.63 | 5477.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-20 13:15:00 | 5500.50 | 5500.95 | 5485.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 13:15:00 | 5505.20 | 5501.80 | 5487.11 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-12-20 15:15:00 | 5425.00 | 5473.15 | 5475.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2024-12-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 15:15:00 | 5425.00 | 5473.15 | 5475.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 15:15:00 | 5395.00 | 5438.72 | 5452.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 13:15:00 | 5398.20 | 5394.71 | 5421.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-26 14:00:00 | 5398.20 | 5394.71 | 5421.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 14:15:00 | 5414.00 | 5398.57 | 5421.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 15:00:00 | 5414.00 | 5398.57 | 5421.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 5419.95 | 5402.84 | 5421.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 5417.60 | 5402.84 | 5421.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 5471.65 | 5416.60 | 5425.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:45:00 | 5473.00 | 5416.60 | 5425.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 10:15:00 | 5464.90 | 5426.26 | 5429.28 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2024-12-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-27 11:15:00 | 5467.70 | 5434.55 | 5432.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-27 12:15:00 | 5487.40 | 5445.12 | 5437.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-30 13:15:00 | 5523.75 | 5526.19 | 5491.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-30 14:00:00 | 5523.75 | 5526.19 | 5491.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 13:15:00 | 5575.90 | 5613.15 | 5585.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 13:45:00 | 5579.95 | 5613.15 | 5585.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 14:15:00 | 5569.40 | 5604.40 | 5584.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-01 15:00:00 | 5569.40 | 5604.40 | 5584.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-01 15:15:00 | 5572.00 | 5597.92 | 5583.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-02 09:15:00 | 5553.70 | 5597.92 | 5583.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 10:15:00 | 5533.55 | 5572.58 | 5573.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-03 09:15:00 | 5495.90 | 5556.59 | 5565.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-06 11:15:00 | 5515.00 | 5505.92 | 5527.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-06 11:30:00 | 5500.00 | 5505.92 | 5527.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 12:15:00 | 5516.35 | 5508.01 | 5526.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 12:30:00 | 5531.75 | 5508.01 | 5526.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 14:15:00 | 5525.50 | 5509.21 | 5523.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-06 15:00:00 | 5525.50 | 5509.21 | 5523.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 15:15:00 | 5527.00 | 5512.77 | 5523.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 09:15:00 | 5537.90 | 5512.77 | 5523.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 09:15:00 | 5583.65 | 5526.94 | 5529.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 10:00:00 | 5583.65 | 5526.94 | 5529.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2025-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 10:15:00 | 5640.65 | 5549.68 | 5539.33 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2025-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 10:15:00 | 5486.40 | 5539.14 | 5542.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-09 14:15:00 | 5454.50 | 5508.28 | 5523.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 15:15:00 | 5277.00 | 5273.45 | 5316.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 09:15:00 | 5203.00 | 5273.45 | 5316.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 5258.60 | 5208.67 | 5228.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:00:00 | 5258.60 | 5208.67 | 5228.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 5240.20 | 5214.98 | 5229.75 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2025-01-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 14:15:00 | 5249.80 | 5236.41 | 5236.16 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2025-01-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 09:15:00 | 5199.75 | 5231.85 | 5234.31 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 10:15:00 | 5256.40 | 5236.76 | 5236.31 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 11:15:00 | 5225.20 | 5234.45 | 5235.30 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2025-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 14:15:00 | 5246.60 | 5237.01 | 5236.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 15:15:00 | 5265.00 | 5242.61 | 5238.87 | Break + close above crossover candle high |

### Cycle 53 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 5208.00 | 5236.58 | 5236.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 14:15:00 | 5158.45 | 5210.63 | 5223.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 5224.50 | 5165.55 | 5183.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-23 09:15:00 | 5224.50 | 5165.55 | 5183.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 09:15:00 | 5224.50 | 5165.55 | 5183.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:00:00 | 5224.50 | 5165.55 | 5183.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 10:15:00 | 5216.80 | 5175.80 | 5186.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 10:30:00 | 5215.85 | 5175.80 | 5186.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 13:15:00 | 5196.75 | 5186.98 | 5189.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-23 14:00:00 | 5196.75 | 5186.98 | 5189.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-23 14:15:00 | 5188.30 | 5187.24 | 5189.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-23 15:15:00 | 5166.05 | 5187.24 | 5189.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-27 10:15:00 | 4907.75 | 5027.77 | 5091.86 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-01-29 09:15:00 | 4930.80 | 4929.27 | 4975.09 | SL hit (close>ema200) qty=0.50 sl=4929.27 alert=retest2 |

### Cycle 54 — BUY (started 2025-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 09:15:00 | 5074.45 | 4994.48 | 4988.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-03 09:15:00 | 5088.80 | 5048.17 | 5037.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 13:15:00 | 5043.90 | 5062.29 | 5048.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 13:15:00 | 5043.90 | 5062.29 | 5048.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 13:15:00 | 5043.90 | 5062.29 | 5048.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 14:00:00 | 5043.90 | 5062.29 | 5048.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 14:15:00 | 5044.55 | 5058.75 | 5048.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 14:45:00 | 5028.90 | 5058.75 | 5048.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 15:15:00 | 5050.00 | 5057.00 | 5048.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:15:00 | 5073.45 | 5057.00 | 5048.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 09:45:00 | 5061.55 | 5056.69 | 5049.20 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-04 10:15:00 | 5066.50 | 5056.69 | 5049.20 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 4829.45 | 5123.45 | 5153.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 4829.45 | 5123.45 | 5153.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 09:15:00 | 4712.30 | 4858.71 | 4980.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-12 14:15:00 | 4704.60 | 4689.05 | 4770.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-12 15:00:00 | 4704.60 | 4689.05 | 4770.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 15:15:00 | 4696.20 | 4690.48 | 4763.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 09:15:00 | 4624.95 | 4714.58 | 4742.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-17 10:00:00 | 4689.55 | 4658.95 | 4688.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-17 13:15:00 | 4754.10 | 4709.88 | 4706.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2025-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-17 13:15:00 | 4754.10 | 4709.88 | 4706.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-17 14:15:00 | 4800.90 | 4728.08 | 4714.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-18 13:15:00 | 4743.50 | 4747.48 | 4732.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-18 13:15:00 | 4743.50 | 4747.48 | 4732.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 13:15:00 | 4743.50 | 4747.48 | 4732.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 14:00:00 | 4743.50 | 4747.48 | 4732.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 14:15:00 | 4721.15 | 4742.22 | 4731.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 15:00:00 | 4721.15 | 4742.22 | 4731.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 15:15:00 | 4711.70 | 4736.11 | 4729.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:15:00 | 4677.40 | 4736.11 | 4729.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 4745.35 | 4737.96 | 4731.36 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2025-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 13:15:00 | 4723.90 | 4731.87 | 4732.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-20 15:15:00 | 4709.15 | 4726.09 | 4729.47 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-25 10:15:00 | 4617.25 | 4613.39 | 4635.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-25 10:30:00 | 4616.10 | 4613.39 | 4635.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-27 09:15:00 | 4572.90 | 4583.53 | 4609.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:45:00 | 4510.20 | 4575.92 | 4593.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-03 13:15:00 | 4607.90 | 4593.57 | 4592.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — BUY (started 2025-03-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-03 13:15:00 | 4607.90 | 4593.57 | 4592.92 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2025-03-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-04 09:15:00 | 4554.95 | 4590.74 | 4592.15 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2025-03-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 12:15:00 | 4601.35 | 4592.67 | 4592.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 13:15:00 | 4624.65 | 4599.06 | 4595.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-04 15:15:00 | 4600.00 | 4602.35 | 4597.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-05 09:15:00 | 4619.85 | 4602.35 | 4597.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 09:15:00 | 4636.45 | 4609.17 | 4601.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-05 12:00:00 | 4649.15 | 4620.90 | 4608.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-05 14:45:00 | 4684.70 | 4639.50 | 4620.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-10 14:15:00 | 4624.00 | 4691.89 | 4699.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2025-03-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 14:15:00 | 4624.00 | 4691.89 | 4699.69 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 12:15:00 | 4725.65 | 4700.44 | 4699.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-11 13:15:00 | 4760.60 | 4712.47 | 4705.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 11:15:00 | 4715.00 | 4736.18 | 4722.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 11:15:00 | 4715.00 | 4736.18 | 4722.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 4715.00 | 4736.18 | 4722.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:00:00 | 4715.00 | 4736.18 | 4722.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 12:15:00 | 4715.35 | 4732.02 | 4722.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:45:00 | 4710.95 | 4732.02 | 4722.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 14:15:00 | 4758.40 | 4734.88 | 4725.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 10:30:00 | 4763.00 | 4737.59 | 4728.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 14:15:00 | 4704.45 | 4725.35 | 4725.27 | SL hit (close<static) qty=1.00 sl=4713.35 alert=retest2 |

### Cycle 63 — SELL (started 2025-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 15:15:00 | 4681.40 | 4716.56 | 4721.28 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2025-03-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-17 09:15:00 | 4782.15 | 4729.68 | 4726.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 12:15:00 | 4844.95 | 4803.65 | 4775.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-25 10:15:00 | 5021.00 | 5036.81 | 4991.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-25 10:45:00 | 5014.60 | 5036.81 | 4991.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 11:15:00 | 4986.35 | 5026.72 | 4991.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 11:45:00 | 4995.35 | 5026.72 | 4991.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 12:15:00 | 4945.05 | 5010.39 | 4986.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-25 13:00:00 | 4945.05 | 5010.39 | 4986.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 13:15:00 | 5039.65 | 5016.24 | 4991.73 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2025-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-27 09:15:00 | 4953.50 | 4991.92 | 4993.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-27 10:15:00 | 4924.10 | 4978.36 | 4987.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 14:15:00 | 4959.40 | 4956.18 | 4971.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 14:15:00 | 4959.40 | 4956.18 | 4971.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 4959.40 | 4956.18 | 4971.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 4959.40 | 4956.18 | 4971.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 09:15:00 | 4886.40 | 4939.00 | 4961.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 12:45:00 | 4841.05 | 4876.37 | 4908.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-01 15:00:00 | 4830.55 | 4859.11 | 4894.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:45:00 | 4830.00 | 4850.54 | 4884.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 11:00:00 | 4847.20 | 4849.87 | 4881.02 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 14:15:00 | 4885.00 | 4852.15 | 4871.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 15:00:00 | 4885.00 | 4852.15 | 4871.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 15:15:00 | 4903.00 | 4862.32 | 4874.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-03 09:15:00 | 4999.85 | 4862.32 | 4874.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-03 09:15:00 | 4965.50 | 4882.96 | 4882.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2025-04-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 09:15:00 | 4965.50 | 4882.96 | 4882.71 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 4770.75 | 4887.10 | 4890.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 4661.85 | 4803.71 | 4841.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 4778.55 | 4734.34 | 4772.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 10:15:00 | 4778.55 | 4734.34 | 4772.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 10:15:00 | 4778.55 | 4734.34 | 4772.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 11:00:00 | 4778.55 | 4734.34 | 4772.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 11:15:00 | 4817.70 | 4751.01 | 4776.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 11:45:00 | 4812.65 | 4751.01 | 4776.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 12:15:00 | 4838.55 | 4768.52 | 4782.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 12:30:00 | 4840.25 | 4768.52 | 4782.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2025-04-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-08 15:15:00 | 4837.15 | 4799.58 | 4794.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 11:15:00 | 4879.25 | 4847.95 | 4827.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 10:15:00 | 4853.40 | 4873.12 | 4851.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-15 11:00:00 | 4853.40 | 4873.12 | 4851.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-15 11:15:00 | 4878.00 | 4874.10 | 4853.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-15 14:00:00 | 4893.60 | 4878.41 | 4859.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 12:15:00 | 5041.20 | 5099.09 | 5106.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 69 — SELL (started 2025-04-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 12:15:00 | 5041.20 | 5099.09 | 5106.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 13:15:00 | 5016.10 | 5082.49 | 5098.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 5072.70 | 5070.24 | 5087.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 09:15:00 | 5072.70 | 5070.24 | 5087.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 5072.70 | 5070.24 | 5087.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:00:00 | 5072.70 | 5070.24 | 5087.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 14:15:00 | 5035.90 | 5021.70 | 5041.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-29 15:00:00 | 5035.90 | 5021.70 | 5041.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 5114.50 | 5041.62 | 5046.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-30 09:30:00 | 5108.50 | 5041.62 | 5046.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 10:15:00 | 5131.40 | 5059.58 | 5054.58 | EMA200 above EMA400 |

### Cycle 71 — SELL (started 2025-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-05 12:15:00 | 5061.00 | 5073.11 | 5073.80 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2025-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 15:15:00 | 5106.00 | 5078.52 | 5075.93 | EMA200 above EMA400 |

### Cycle 73 — SELL (started 2025-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 10:15:00 | 5053.00 | 5070.77 | 5072.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 11:15:00 | 5041.50 | 5064.92 | 5069.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 5034.00 | 5029.29 | 5045.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 12:00:00 | 5034.00 | 5029.29 | 5045.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 5027.50 | 5028.93 | 5043.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 13:00:00 | 5027.50 | 5028.93 | 5043.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 5044.00 | 5031.95 | 5043.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 5044.00 | 5031.95 | 5043.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 5035.50 | 5032.66 | 5043.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:30:00 | 5041.50 | 5032.66 | 5043.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 4948.00 | 4924.91 | 4958.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 15:00:00 | 4948.00 | 4924.91 | 4958.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 4950.00 | 4929.93 | 4958.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 4940.50 | 4929.93 | 4958.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 5032.50 | 4950.45 | 4964.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 5032.50 | 4950.45 | 4964.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 5019.50 | 4964.26 | 4969.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-12 11:15:00 | 5006.00 | 4964.26 | 4969.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-12 11:15:00 | 5019.00 | 4975.20 | 4974.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 74 — BUY (started 2025-05-12 11:15:00)

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

### Cycle 75 — SELL (started 2025-05-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 13:15:00 | 5294.00 | 5308.53 | 5310.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-29 13:15:00 | 5172.00 | 5229.31 | 5244.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 14:15:00 | 5293.50 | 5242.15 | 5249.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 14:15:00 | 5293.50 | 5242.15 | 5249.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 5293.50 | 5242.15 | 5249.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 5293.50 | 5242.15 | 5249.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2025-05-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 15:15:00 | 5325.00 | 5258.72 | 5255.93 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-05-30 09:15:00)

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

### Cycle 78 — BUY (started 2025-06-10 10:15:00)

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

### Cycle 79 — SELL (started 2025-06-11 15:15:00)

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

### Cycle 80 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 4867.00 | 4853.84 | 4852.17 | EMA200 above EMA400 |

### Cycle 81 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 4840.00 | 4850.85 | 4851.09 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2025-06-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 13:15:00 | 4855.00 | 4851.68 | 4851.44 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2025-06-17 14:15:00)

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

### Cycle 84 — BUY (started 2025-06-24 11:15:00)

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

### Cycle 85 — SELL (started 2025-07-01 12:15:00)

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

### Cycle 86 — BUY (started 2025-07-04 12:15:00)

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

### Cycle 87 — SELL (started 2025-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 10:15:00 | 4813.80 | 4859.01 | 4861.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 11:15:00 | 4801.40 | 4847.49 | 4856.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 10:15:00 | 4819.10 | 4816.19 | 4832.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 10:45:00 | 4822.10 | 4816.19 | 4832.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 4797.10 | 4761.70 | 4783.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:45:00 | 4804.50 | 4761.70 | 4783.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 4777.10 | 4764.78 | 4783.22 | EMA400 retest candle locked (from downside) |

### Cycle 88 — BUY (started 2025-07-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 13:15:00 | 4839.80 | 4796.59 | 4794.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-14 09:15:00 | 4843.30 | 4807.68 | 4800.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 09:15:00 | 4990.10 | 5005.13 | 4973.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 10:00:00 | 4990.10 | 5005.13 | 4973.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 5006.80 | 5002.42 | 4977.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:30:00 | 4991.70 | 5002.42 | 4977.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 4987.40 | 4997.96 | 4984.81 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 10:15:00 | 4951.00 | 4985.28 | 4986.18 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 12:15:00 | 4991.90 | 4982.18 | 4982.07 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 13:15:00 | 4978.90 | 4981.53 | 4981.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 15:15:00 | 4967.60 | 4978.48 | 4980.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 4980.40 | 4978.87 | 4980.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 4980.40 | 4978.87 | 4980.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 4980.40 | 4978.87 | 4980.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:30:00 | 5016.10 | 4978.87 | 4980.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2025-07-24 10:15:00)

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

### Cycle 93 — SELL (started 2025-07-31 14:15:00)

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

### Cycle 94 — BUY (started 2025-08-12 09:15:00)

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

### Cycle 95 — SELL (started 2025-08-26 10:15:00)

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

### Cycle 96 — BUY (started 2025-09-04 10:15:00)

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

### Cycle 97 — SELL (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 12:15:00 | 5313.00 | 5332.23 | 5332.90 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2025-09-11 09:15:00)

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

### Cycle 99 — SELL (started 2025-09-25 13:15:00)

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

### Cycle 100 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 5469.00 | 5442.91 | 5440.98 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 09:15:00 | 5401.50 | 5435.09 | 5439.53 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2025-10-06 13:15:00)

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

### Cycle 103 — SELL (started 2025-10-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 09:15:00 | 5455.00 | 5478.77 | 5479.63 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2025-10-13 14:15:00)

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

### Cycle 105 — SELL (started 2025-10-23 14:15:00)

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

### Cycle 106 — BUY (started 2025-10-29 14:15:00)

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

### Cycle 107 — SELL (started 2025-11-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-14 10:15:00 | 5655.00 | 5721.64 | 5728.13 | EMA200 below EMA400 |

### Cycle 108 — BUY (started 2025-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 13:15:00 | 5729.50 | 5707.38 | 5706.01 | EMA200 above EMA400 |

### Cycle 109 — SELL (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 09:15:00 | 5669.00 | 5701.00 | 5703.56 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2025-11-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 13:15:00 | 5737.00 | 5707.82 | 5705.27 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2025-11-20 15:15:00)

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

### Cycle 112 — BUY (started 2025-11-26 09:15:00)

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

### Cycle 113 — SELL (started 2025-11-28 14:15:00)

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

### Cycle 114 — BUY (started 2025-12-04 09:15:00)

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

### Cycle 115 — SELL (started 2025-12-08 13:15:00)

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

### Cycle 116 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 5672.00 | 5654.24 | 5653.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 5717.50 | 5666.89 | 5659.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 10:15:00 | 5663.50 | 5666.21 | 5659.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 10:15:00 | 5663.50 | 5666.21 | 5659.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 10:15:00 | 5663.50 | 5666.21 | 5659.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-10 11:00:00 | 5663.50 | 5666.21 | 5659.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 11:15:00 | 5643.00 | 5661.57 | 5658.15 | EMA400 retest candle locked (from upside) |

### Cycle 117 — SELL (started 2025-12-10 12:15:00)

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

### Cycle 118 — BUY (started 2025-12-16 11:15:00)

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

### Cycle 119 — SELL (started 2025-12-17 13:15:00)

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

### Cycle 120 — BUY (started 2025-12-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 11:15:00 | 5640.00 | 5580.27 | 5577.77 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2025-12-24 09:15:00)

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

### Cycle 122 — BUY (started 2026-01-02 11:15:00)

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

### Cycle 123 — SELL (started 2026-01-19 09:15:00)

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

### Cycle 124 — BUY (started 2026-01-22 15:15:00)

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

### Cycle 125 — SELL (started 2026-01-28 11:15:00)

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

### Cycle 126 — BUY (started 2026-02-03 11:15:00)

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

### Cycle 127 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 5580.50 | 5640.70 | 5643.60 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 5674.00 | 5645.03 | 5644.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 5693.00 | 5654.62 | 5648.56 | Break + close above crossover candle high |

### Cycle 129 — SELL (started 2026-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 15:15:00 | 5587.50 | 5641.20 | 5643.01 | EMA200 below EMA400 |

### Cycle 130 — BUY (started 2026-02-09 09:15:00)

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

### Cycle 131 — SELL (started 2026-02-13 11:15:00)

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

### Cycle 132 — BUY (started 2026-02-23 13:15:00)

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

### Cycle 133 — SELL (started 2026-03-02 10:15:00)

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

### Cycle 134 — BUY (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 13:15:00 | 5555.50 | 5525.67 | 5524.97 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 5461.00 | 5513.51 | 5519.72 | EMA200 below EMA400 |

### Cycle 136 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 5620.00 | 5520.94 | 5512.71 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 5467.50 | 5547.94 | 5550.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 13:15:00 | 5407.00 | 5486.23 | 5518.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 10:15:00 | 5322.00 | 5315.30 | 5361.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 5385.00 | 5340.27 | 5354.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 5385.00 | 5340.27 | 5354.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 5253.50 | 5358.86 | 5359.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 14:15:00 | 5294.50 | 5230.04 | 5228.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 138 — BUY (started 2026-03-24 14:15:00)

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

### Cycle 139 — SELL (started 2026-03-30 10:15:00)

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

### Cycle 140 — BUY (started 2026-04-08 11:15:00)

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

### Cycle 141 — SELL (started 2026-04-23 09:15:00)

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

### Cycle 142 — BUY (started 2026-05-05 14:15:00)

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
| BUY | retest2 | 2024-05-13 09:15:00 | 5179.60 | 2024-05-22 09:15:00 | 5237.70 | STOP_HIT | 1.00 | 1.12% |
| BUY | retest2 | 2024-05-13 10:15:00 | 5127.00 | 2024-05-22 09:15:00 | 5237.70 | STOP_HIT | 1.00 | 2.16% |
| BUY | retest2 | 2024-05-17 10:45:00 | 5329.95 | 2024-05-22 09:15:00 | 5237.70 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2024-05-21 11:00:00 | 5363.00 | 2024-05-22 09:15:00 | 5237.70 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2024-05-21 12:30:00 | 5337.45 | 2024-05-22 09:15:00 | 5237.70 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2024-05-21 15:00:00 | 5330.20 | 2024-05-22 09:15:00 | 5237.70 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2024-05-24 11:45:00 | 5472.80 | 2024-05-27 12:15:00 | 5325.50 | STOP_HIT | 1.00 | -2.69% |
| BUY | retest2 | 2024-05-24 13:00:00 | 5470.00 | 2024-05-27 12:15:00 | 5325.50 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2024-06-20 11:45:00 | 5186.50 | 2024-06-21 11:15:00 | 5127.90 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2024-06-26 09:15:00 | 5006.15 | 2024-07-03 11:15:00 | 5029.00 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2024-06-26 13:00:00 | 5027.15 | 2024-07-03 11:15:00 | 5029.00 | STOP_HIT | 1.00 | -0.04% |
| BUY | retest2 | 2024-07-10 12:30:00 | 5225.00 | 2024-07-12 11:15:00 | 5196.75 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2024-07-10 13:45:00 | 5220.20 | 2024-07-12 11:15:00 | 5196.75 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2024-07-10 14:15:00 | 5221.55 | 2024-07-12 11:15:00 | 5196.75 | STOP_HIT | 1.00 | -0.47% |
| BUY | retest2 | 2024-07-11 14:30:00 | 5231.15 | 2024-07-12 11:15:00 | 5196.75 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2024-07-12 09:15:00 | 5252.95 | 2024-07-12 11:15:00 | 5196.75 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-07-12 10:45:00 | 5212.85 | 2024-07-12 11:15:00 | 5196.75 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2024-07-23 12:15:00 | 5111.55 | 2024-07-24 09:15:00 | 5231.15 | STOP_HIT | 1.00 | -2.34% |
| SELL | retest2 | 2024-07-23 13:30:00 | 5157.65 | 2024-07-24 09:15:00 | 5231.15 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-07-30 15:15:00 | 5205.00 | 2024-07-31 09:15:00 | 5290.00 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2024-08-02 13:45:00 | 5324.80 | 2024-08-05 10:15:00 | 5255.90 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-08-02 14:15:00 | 5323.65 | 2024-08-05 10:15:00 | 5255.90 | STOP_HIT | 1.00 | -1.27% |
| BUY | retest2 | 2024-08-05 13:30:00 | 5339.95 | 2024-08-16 09:15:00 | 5650.60 | STOP_HIT | 1.00 | 5.82% |
| BUY | retest2 | 2024-08-06 09:15:00 | 5350.00 | 2024-08-16 09:15:00 | 5650.60 | STOP_HIT | 1.00 | 5.62% |
| BUY | retest2 | 2024-08-07 09:15:00 | 5442.05 | 2024-08-16 09:15:00 | 5650.60 | STOP_HIT | 1.00 | 3.83% |
| BUY | retest2 | 2024-08-21 09:30:00 | 5761.90 | 2024-09-05 10:15:00 | 6338.09 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-08-26 09:30:00 | 5776.30 | 2024-09-10 09:15:00 | 6353.93 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2024-09-19 12:45:00 | 6115.70 | 2024-09-24 10:15:00 | 6071.50 | STOP_HIT | 1.00 | 0.72% |
| SELL | retest2 | 2024-09-20 14:30:00 | 6055.70 | 2024-09-25 11:15:00 | 6110.05 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2024-09-23 10:00:00 | 6093.35 | 2024-09-25 11:15:00 | 6110.05 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2024-10-25 13:45:00 | 5908.95 | 2024-10-28 10:15:00 | 6016.70 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-10-30 10:30:00 | 5886.60 | 2024-11-06 13:15:00 | 5828.55 | STOP_HIT | 1.00 | 0.99% |
| SELL | retest2 | 2024-11-14 10:00:00 | 5512.65 | 2024-11-19 12:15:00 | 5577.85 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-11-14 11:45:00 | 5516.90 | 2024-11-19 12:15:00 | 5577.85 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2024-11-14 14:45:00 | 5521.95 | 2024-11-19 12:15:00 | 5577.85 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-11-18 10:15:00 | 5519.50 | 2024-11-19 12:15:00 | 5577.85 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2024-11-21 14:15:00 | 5586.35 | 2024-11-22 10:15:00 | 5529.00 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2024-12-12 09:15:00 | 5449.30 | 2024-12-18 09:15:00 | 5502.00 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-12-19 10:30:00 | 5502.45 | 2024-12-20 15:15:00 | 5425.00 | STOP_HIT | 1.00 | -1.41% |
| BUY | retest2 | 2024-12-19 15:00:00 | 5501.95 | 2024-12-20 15:15:00 | 5425.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-12-20 09:30:00 | 5504.60 | 2024-12-20 15:15:00 | 5425.00 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2024-12-20 13:15:00 | 5500.50 | 2024-12-20 15:15:00 | 5425.00 | STOP_HIT | 1.00 | -1.37% |
| SELL | retest2 | 2025-01-23 15:15:00 | 5166.05 | 2025-01-27 10:15:00 | 4907.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-23 15:15:00 | 5166.05 | 2025-01-29 09:15:00 | 4930.80 | STOP_HIT | 0.50 | 4.55% |
| BUY | retest2 | 2025-02-04 09:15:00 | 5073.45 | 2025-02-10 09:15:00 | 4829.45 | STOP_HIT | 1.00 | -4.81% |
| BUY | retest2 | 2025-02-04 09:45:00 | 5061.55 | 2025-02-10 09:15:00 | 4829.45 | STOP_HIT | 1.00 | -4.59% |
| BUY | retest2 | 2025-02-04 10:15:00 | 5066.50 | 2025-02-10 09:15:00 | 4829.45 | STOP_HIT | 1.00 | -4.68% |
| SELL | retest2 | 2025-02-14 09:15:00 | 4624.95 | 2025-02-17 13:15:00 | 4754.10 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-02-17 10:00:00 | 4689.55 | 2025-02-17 13:15:00 | 4754.10 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2025-02-28 09:45:00 | 4510.20 | 2025-03-03 13:15:00 | 4607.90 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2025-03-05 12:00:00 | 4649.15 | 2025-03-10 14:15:00 | 4624.00 | STOP_HIT | 1.00 | -0.54% |
| BUY | retest2 | 2025-03-05 14:45:00 | 4684.70 | 2025-03-10 14:15:00 | 4624.00 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-03-13 10:30:00 | 4763.00 | 2025-03-13 14:15:00 | 4704.45 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2025-04-01 12:45:00 | 4841.05 | 2025-04-03 09:15:00 | 4965.50 | STOP_HIT | 1.00 | -2.57% |
| SELL | retest2 | 2025-04-01 15:00:00 | 4830.55 | 2025-04-03 09:15:00 | 4965.50 | STOP_HIT | 1.00 | -2.79% |
| SELL | retest2 | 2025-04-02 09:45:00 | 4830.00 | 2025-04-03 09:15:00 | 4965.50 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2025-04-02 11:00:00 | 4847.20 | 2025-04-03 09:15:00 | 4965.50 | STOP_HIT | 1.00 | -2.44% |
| BUY | retest2 | 2025-04-15 14:00:00 | 4893.60 | 2025-04-25 12:15:00 | 5041.20 | STOP_HIT | 1.00 | 3.02% |
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
