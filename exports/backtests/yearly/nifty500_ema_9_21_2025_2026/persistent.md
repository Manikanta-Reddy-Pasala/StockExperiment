# Persistent Systems Ltd. (PERSISTENT)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1983 bars)
- **Last close:** 5115.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 65 |
| ALERT1 | 51 |
| ALERT2 | 50 |
| ALERT2_SKIP | 25 |
| ALERT3 | 143 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 65 |
| PARTIAL | 7 |
| TARGET_HIT | 3 |
| STOP_HIT | 62 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 72 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 17 / 55
- **Target hits / Stop hits / Partials:** 3 / 62 / 7
- **Avg / median % per leg:** 0.21% / -0.80%
- **Sum % (uncompounded):** 15.02%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 29 | 4 | 13.8% | 2 | 27 | 0 | -0.25% | -7.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 29 | 4 | 13.8% | 2 | 27 | 0 | -0.25% | -7.4% |
| SELL (all) | 43 | 13 | 30.2% | 1 | 35 | 7 | 0.52% | 22.4% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 5.40% | 10.8% |
| SELL @ 3rd Alert (retest2) | 41 | 11 | 26.8% | 1 | 34 | 6 | 0.28% | 11.6% |
| retest1 (combined) | 2 | 2 | 100.0% | 0 | 1 | 1 | 5.40% | 10.8% |
| retest2 (combined) | 70 | 15 | 21.4% | 3 | 61 | 6 | 0.06% | 4.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 5707.50 | 5498.53 | 5482.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 5802.00 | 5628.54 | 5551.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 5791.00 | 5792.57 | 5701.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 5791.00 | 5792.57 | 5701.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 11:15:00 | 5700.00 | 5753.71 | 5715.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 11:45:00 | 5700.00 | 5753.71 | 5715.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 12:15:00 | 5697.00 | 5742.37 | 5713.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 12:45:00 | 5643.50 | 5742.37 | 5713.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 5756.50 | 5740.58 | 5717.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 15:15:00 | 5768.00 | 5740.58 | 5717.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-15 09:15:00 | 5584.50 | 5713.75 | 5709.70 | SL hit (close<static) qty=1.00 sl=5710.50 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 10:15:00 | 5585.50 | 5688.10 | 5698.41 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 15:15:00 | 5693.00 | 5678.15 | 5678.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 5714.00 | 5685.32 | 5681.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 11:15:00 | 5679.50 | 5688.90 | 5683.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 11:15:00 | 5679.50 | 5688.90 | 5683.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 5679.50 | 5688.90 | 5683.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 12:00:00 | 5679.50 | 5688.90 | 5683.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 5700.00 | 5691.12 | 5685.40 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-05-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 15:15:00 | 5665.50 | 5681.82 | 5682.26 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-20 09:15:00 | 5764.50 | 5698.35 | 5689.74 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 5670.00 | 5690.86 | 5691.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-22 10:15:00 | 5566.00 | 5647.41 | 5667.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-23 09:15:00 | 5710.00 | 5627.79 | 5643.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-23 09:15:00 | 5710.00 | 5627.79 | 5643.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 5710.00 | 5627.79 | 5643.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 10:00:00 | 5710.00 | 5627.79 | 5643.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 10:15:00 | 5697.50 | 5641.73 | 5648.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 11:30:00 | 5680.00 | 5647.18 | 5650.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 12:15:00 | 5687.00 | 5655.15 | 5653.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 5687.00 | 5655.15 | 5653.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 13:15:00 | 5690.00 | 5673.56 | 5665.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 10:15:00 | 5656.00 | 5674.29 | 5668.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 10:15:00 | 5656.00 | 5674.29 | 5668.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 5656.00 | 5674.29 | 5668.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:45:00 | 5664.50 | 5674.29 | 5668.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-05-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 11:15:00 | 5628.00 | 5665.03 | 5665.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 13:15:00 | 5612.00 | 5637.22 | 5648.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 5661.50 | 5628.41 | 5640.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 5661.50 | 5628.41 | 5640.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 5661.50 | 5628.41 | 5640.27 | EMA400 retest candle locked (from downside) |

### Cycle 9 — BUY (started 2025-05-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 14:15:00 | 5667.00 | 5649.42 | 5647.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-29 15:15:00 | 5677.00 | 5654.94 | 5650.16 | Break + close above crossover candle high |

### Cycle 10 — SELL (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 09:15:00 | 5606.50 | 5645.25 | 5646.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 11:15:00 | 5602.50 | 5631.86 | 5639.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 14:15:00 | 5635.50 | 5626.92 | 5635.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 14:15:00 | 5635.50 | 5626.92 | 5635.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 5635.50 | 5626.92 | 5635.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 15:00:00 | 5635.50 | 5626.92 | 5635.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 5625.00 | 5626.53 | 5634.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 09:15:00 | 5553.50 | 5626.53 | 5634.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 11:15:00 | 5633.00 | 5539.11 | 5532.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 5633.00 | 5539.11 | 5532.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 13:15:00 | 5665.00 | 5616.65 | 5585.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 5884.00 | 5937.02 | 5885.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 5884.00 | 5937.02 | 5885.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 5884.00 | 5937.02 | 5885.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 09:45:00 | 5847.50 | 5937.02 | 5885.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 5911.50 | 5931.92 | 5888.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 11:15:00 | 5934.00 | 5931.92 | 5888.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-12 14:15:00 | 5919.00 | 5939.71 | 5903.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 5940.00 | 5925.40 | 5905.46 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 12:15:00 | 5920.50 | 5924.22 | 5908.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 12:15:00 | 5906.50 | 5920.67 | 5908.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 13:00:00 | 5906.50 | 5920.67 | 5908.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 13:15:00 | 5893.50 | 5915.24 | 5906.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 13:45:00 | 5882.50 | 5915.24 | 5906.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 5883.00 | 5908.79 | 5904.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 14:45:00 | 5861.50 | 5908.79 | 5904.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 10:15:00 | 5943.00 | 5912.70 | 5907.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 11:15:00 | 5964.00 | 5912.70 | 5907.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 10:15:00 | 5877.00 | 5992.60 | 6005.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2025-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 10:15:00 | 5877.00 | 5992.60 | 6005.58 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-06-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 13:15:00 | 6034.00 | 5989.57 | 5984.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 6106.00 | 6012.86 | 5995.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 14:15:00 | 6081.00 | 6086.21 | 6049.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-23 14:45:00 | 6085.00 | 6086.21 | 6049.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 14:15:00 | 6078.50 | 6100.93 | 6077.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 15:00:00 | 6078.50 | 6100.93 | 6077.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 6080.00 | 6096.74 | 6077.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 09:15:00 | 6142.00 | 6096.74 | 6077.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 12:45:00 | 6109.00 | 6112.49 | 6092.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-26 09:15:00 | 6043.50 | 6097.35 | 6091.63 | SL hit (close<static) qty=1.00 sl=6075.50 alert=retest2 |

### Cycle 14 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 6044.50 | 6081.28 | 6084.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 13:15:00 | 6031.50 | 6060.09 | 6070.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 11:15:00 | 6032.50 | 6031.99 | 6050.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 12:00:00 | 6032.50 | 6031.99 | 6050.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 13:15:00 | 6070.00 | 6040.31 | 6051.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:00:00 | 6070.00 | 6040.31 | 6051.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 6040.00 | 6040.25 | 6050.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 14:30:00 | 6066.00 | 6040.25 | 6050.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 6042.00 | 6040.60 | 6049.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 6057.00 | 6040.60 | 6049.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 6042.00 | 6040.88 | 6048.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:30:00 | 6078.50 | 6040.88 | 6048.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 6046.00 | 6041.90 | 6048.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:30:00 | 6060.00 | 6041.90 | 6048.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 6029.50 | 6039.42 | 6046.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:30:00 | 6044.00 | 6039.42 | 6046.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 12:15:00 | 5998.00 | 6031.14 | 6042.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:30:00 | 6027.00 | 6031.14 | 6042.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 14:15:00 | 6035.00 | 6030.29 | 6039.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 15:00:00 | 6035.00 | 6030.29 | 6039.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 6028.00 | 6029.83 | 6038.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 6076.50 | 6029.83 | 6038.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 6076.50 | 6039.16 | 6042.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:45:00 | 6093.50 | 6039.16 | 6042.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 10:15:00 | 6014.50 | 6034.23 | 6039.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 13:15:00 | 5990.00 | 6033.19 | 6038.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 11:30:00 | 5991.00 | 6014.22 | 6025.81 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:00:00 | 5986.50 | 6014.22 | 6025.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 09:15:00 | 5690.50 | 5735.26 | 5786.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 09:15:00 | 5691.45 | 5735.26 | 5786.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-10 09:15:00 | 5687.18 | 5735.26 | 5786.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-14 14:15:00 | 5612.00 | 5589.59 | 5628.58 | SL hit (close>ema200) qty=0.50 sl=5589.59 alert=retest2 |

### Cycle 15 — BUY (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 09:15:00 | 5650.00 | 5555.73 | 5551.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-21 10:15:00 | 5683.00 | 5581.18 | 5563.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-22 13:15:00 | 5748.50 | 5749.57 | 5687.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-22 14:00:00 | 5748.50 | 5749.57 | 5687.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 5684.00 | 5729.01 | 5692.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 5684.00 | 5729.01 | 5692.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 5646.00 | 5712.41 | 5688.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:15:00 | 5639.00 | 5712.41 | 5688.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — SELL (started 2025-07-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 13:15:00 | 5599.50 | 5662.83 | 5669.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 09:15:00 | 5105.00 | 5534.03 | 5607.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 14:15:00 | 5158.00 | 5151.60 | 5228.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-28 15:00:00 | 5158.00 | 5151.60 | 5228.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 5178.00 | 5160.28 | 5194.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:30:00 | 5186.00 | 5160.28 | 5194.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 5135.50 | 5158.48 | 5188.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:30:00 | 5101.50 | 5148.38 | 5180.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:30:00 | 5109.00 | 5146.54 | 5167.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 10:00:00 | 5103.50 | 5146.54 | 5167.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 11:30:00 | 5097.50 | 5137.33 | 5151.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 5116.50 | 5084.03 | 5108.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 5116.50 | 5084.03 | 5108.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 5157.50 | 5098.73 | 5113.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 5157.50 | 5098.73 | 5113.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 5175.50 | 5114.08 | 5118.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:45:00 | 5183.50 | 5114.08 | 5118.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-04 15:15:00 | 5175.00 | 5126.26 | 5123.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 15:15:00 | 5175.00 | 5126.26 | 5123.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 14:15:00 | 5197.50 | 5154.34 | 5139.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 5125.50 | 5154.35 | 5142.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 09:15:00 | 5125.50 | 5154.35 | 5142.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 09:15:00 | 5125.50 | 5154.35 | 5142.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 09:45:00 | 5136.00 | 5154.35 | 5142.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 5076.00 | 5138.68 | 5136.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:00:00 | 5076.00 | 5138.68 | 5136.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — SELL (started 2025-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 11:15:00 | 5093.00 | 5129.55 | 5132.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 14:15:00 | 5047.00 | 5099.58 | 5117.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 09:15:00 | 5124.50 | 5096.39 | 5112.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 5124.50 | 5096.39 | 5112.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 5124.50 | 5096.39 | 5112.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:00:00 | 5124.50 | 5096.39 | 5112.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 5110.50 | 5099.22 | 5111.97 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2025-08-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 13:15:00 | 5151.50 | 5122.41 | 5120.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-07 14:15:00 | 5175.00 | 5132.93 | 5125.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-08 09:15:00 | 5139.50 | 5142.57 | 5131.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-08 09:15:00 | 5139.50 | 5142.57 | 5131.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 5139.50 | 5142.57 | 5131.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:00:00 | 5139.50 | 5142.57 | 5131.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 5142.00 | 5142.46 | 5132.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 10:45:00 | 5135.00 | 5142.46 | 5132.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 11:15:00 | 5158.50 | 5145.67 | 5135.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:30:00 | 5144.00 | 5145.67 | 5135.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 5138.00 | 5144.13 | 5135.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 13:00:00 | 5138.00 | 5144.13 | 5135.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 13:15:00 | 5119.50 | 5139.21 | 5133.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 14:00:00 | 5119.50 | 5139.21 | 5133.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 5105.00 | 5132.37 | 5131.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 5105.00 | 5132.37 | 5131.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-08-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 15:15:00 | 5100.00 | 5125.89 | 5128.38 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 5156.00 | 5131.93 | 5130.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 11:15:00 | 5173.00 | 5140.14 | 5134.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 10:15:00 | 5220.00 | 5225.83 | 5200.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-13 11:00:00 | 5220.00 | 5225.83 | 5200.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 5227.50 | 5226.17 | 5203.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:45:00 | 5222.50 | 5226.17 | 5203.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 5257.50 | 5278.58 | 5255.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:30:00 | 5264.00 | 5278.58 | 5255.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 10:15:00 | 5277.00 | 5278.26 | 5257.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 10:30:00 | 5270.00 | 5278.26 | 5257.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 5234.00 | 5269.41 | 5255.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:00:00 | 5234.00 | 5269.41 | 5255.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 5250.00 | 5265.53 | 5254.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:30:00 | 5237.00 | 5265.53 | 5254.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 13:15:00 | 5252.00 | 5262.82 | 5254.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 13:30:00 | 5246.50 | 5262.82 | 5254.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 15:15:00 | 5231.00 | 5250.57 | 5250.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:15:00 | 5255.50 | 5250.57 | 5250.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 09:15:00 | 5201.50 | 5240.75 | 5245.69 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-08-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-20 10:15:00 | 5276.50 | 5245.01 | 5242.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 11:15:00 | 5346.00 | 5265.21 | 5251.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 5308.50 | 5312.76 | 5284.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 5308.50 | 5312.76 | 5284.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 5308.50 | 5312.76 | 5284.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 09:30:00 | 5378.50 | 5347.40 | 5318.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 10:30:00 | 5376.00 | 5349.52 | 5322.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 11:45:00 | 5375.00 | 5353.41 | 5326.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 12:15:00 | 5383.50 | 5353.41 | 5326.37 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 15:15:00 | 5350.00 | 5363.21 | 5340.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-25 09:15:00 | 5475.00 | 5363.21 | 5340.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 09:15:00 | 5335.50 | 5398.80 | 5401.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 09:15:00 | 5335.50 | 5398.80 | 5401.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 09:15:00 | 5281.00 | 5341.05 | 5368.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 15:15:00 | 5320.00 | 5319.09 | 5343.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 09:15:00 | 5371.00 | 5319.09 | 5343.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 5364.50 | 5328.17 | 5345.51 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-09-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 11:15:00 | 5424.00 | 5359.71 | 5357.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 10:15:00 | 5437.50 | 5406.17 | 5385.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 5387.00 | 5409.44 | 5392.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 5387.00 | 5409.44 | 5392.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 5387.00 | 5409.44 | 5392.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 5387.00 | 5409.44 | 5392.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 5409.50 | 5409.45 | 5394.32 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-09-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 10:15:00 | 5319.00 | 5382.78 | 5385.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 5288.00 | 5320.63 | 5343.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 5074.00 | 5069.27 | 5138.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-09 09:45:00 | 5090.00 | 5069.27 | 5138.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 5132.00 | 5089.09 | 5121.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 5132.00 | 5089.09 | 5121.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 5132.00 | 5097.67 | 5122.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 5309.00 | 5097.67 | 5122.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 5369.50 | 5152.04 | 5145.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 10:15:00 | 5386.50 | 5198.93 | 5167.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 5329.50 | 5336.19 | 5263.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-11 09:30:00 | 5359.00 | 5336.19 | 5263.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 5344.00 | 5381.27 | 5352.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:00:00 | 5344.00 | 5381.27 | 5352.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 10:15:00 | 5356.00 | 5376.22 | 5352.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 10:45:00 | 5352.00 | 5376.22 | 5352.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 11:15:00 | 5380.50 | 5377.07 | 5355.04 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 09:45:00 | 5391.00 | 5373.81 | 5361.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-16 14:00:00 | 5390.00 | 5373.99 | 5364.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 09:15:00 | 5280.00 | 5473.85 | 5494.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 09:15:00 | 5280.00 | 5473.85 | 5494.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 09:15:00 | 5140.50 | 5233.57 | 5302.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 09:15:00 | 5222.00 | 5197.67 | 5244.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 09:30:00 | 5226.50 | 5197.67 | 5244.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 4950.50 | 4867.32 | 4910.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 4950.50 | 4867.32 | 4910.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 4934.80 | 4880.81 | 4912.89 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 5040.00 | 4942.14 | 4934.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 09:15:00 | 5199.00 | 5058.43 | 5003.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 14:15:00 | 5254.80 | 5268.10 | 5217.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 15:00:00 | 5254.80 | 5268.10 | 5217.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 5262.20 | 5337.37 | 5313.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 10:00:00 | 5262.20 | 5337.37 | 5313.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 10:15:00 | 5220.40 | 5313.98 | 5305.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 11:00:00 | 5220.40 | 5313.98 | 5305.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 5237.40 | 5298.66 | 5298.88 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 14:15:00 | 5326.70 | 5299.51 | 5298.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-14 09:15:00 | 5354.60 | 5315.57 | 5306.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 15:15:00 | 5299.80 | 5338.88 | 5326.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 15:15:00 | 5299.80 | 5338.88 | 5326.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 5299.80 | 5338.88 | 5326.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 5676.80 | 5338.88 | 5326.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 11:15:00 | 5816.60 | 5849.54 | 5849.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 11:15:00 | 5816.60 | 5849.54 | 5849.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 13:15:00 | 5805.30 | 5835.49 | 5843.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 10:15:00 | 5847.00 | 5835.79 | 5840.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 10:15:00 | 5847.00 | 5835.79 | 5840.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 5847.00 | 5835.79 | 5840.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:30:00 | 5850.10 | 5835.79 | 5840.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 11:15:00 | 5849.70 | 5838.57 | 5841.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 12:00:00 | 5849.70 | 5838.57 | 5841.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2025-10-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 12:15:00 | 5863.60 | 5843.58 | 5843.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 14:15:00 | 5879.10 | 5854.91 | 5848.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 09:15:00 | 5907.20 | 5919.84 | 5893.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 10:00:00 | 5907.20 | 5919.84 | 5893.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 5887.70 | 5913.41 | 5893.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 5887.70 | 5913.41 | 5893.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 11:15:00 | 5878.00 | 5906.33 | 5891.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:00:00 | 5878.00 | 5906.33 | 5891.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 5885.90 | 5902.24 | 5891.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-31 14:00:00 | 5915.20 | 5904.83 | 5893.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-03 09:15:00 | 5856.00 | 5898.49 | 5893.75 | SL hit (close<static) qty=1.00 sl=5875.10 alert=retest2 |

### Cycle 34 — SELL (started 2025-11-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 11:15:00 | 5839.50 | 5891.43 | 5895.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 10:15:00 | 5816.00 | 5853.73 | 5872.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 5869.00 | 5853.07 | 5868.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 12:15:00 | 5869.00 | 5853.07 | 5868.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 5869.00 | 5853.07 | 5868.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:00:00 | 5869.00 | 5853.07 | 5868.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 5842.50 | 5850.96 | 5866.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 15:15:00 | 5831.50 | 5849.46 | 5863.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 13:15:00 | 5864.00 | 5829.88 | 5827.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 5864.00 | 5829.88 | 5827.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 09:15:00 | 6014.50 | 5878.93 | 5851.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 6131.50 | 6138.83 | 6080.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 6131.50 | 6138.83 | 6080.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 6096.50 | 6128.95 | 6085.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:15:00 | 6090.50 | 6128.95 | 6085.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 6103.50 | 6123.86 | 6087.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:45:00 | 6093.00 | 6123.86 | 6087.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 6040.50 | 6103.37 | 6084.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:00:00 | 6040.50 | 6103.37 | 6084.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 6049.00 | 6092.50 | 6081.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 14:30:00 | 6063.50 | 6095.60 | 6083.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:00:00 | 6108.00 | 6095.60 | 6083.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 15:15:00 | 6065.00 | 6094.50 | 6095.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 6065.00 | 6094.50 | 6095.94 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-11-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 09:15:00 | 6281.00 | 6131.80 | 6112.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 11:15:00 | 6318.00 | 6191.47 | 6144.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 15:15:00 | 6338.00 | 6349.02 | 6285.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-21 09:15:00 | 6348.00 | 6349.02 | 6285.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 6285.50 | 6336.31 | 6285.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:45:00 | 6296.50 | 6336.31 | 6285.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 6300.00 | 6329.05 | 6287.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 10:30:00 | 6291.50 | 6329.05 | 6287.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 6299.00 | 6323.04 | 6288.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:30:00 | 6279.50 | 6323.04 | 6288.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 14:15:00 | 6301.00 | 6315.29 | 6292.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 15:00:00 | 6301.00 | 6315.29 | 6292.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 15:15:00 | 6278.00 | 6307.83 | 6291.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-24 09:15:00 | 6389.00 | 6307.83 | 6291.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-28 13:15:00 | 6370.00 | 6391.50 | 6392.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2025-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 13:15:00 | 6370.00 | 6391.50 | 6392.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 10:15:00 | 6346.00 | 6370.48 | 6380.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-01 11:15:00 | 6395.00 | 6375.39 | 6382.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 11:15:00 | 6395.00 | 6375.39 | 6382.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 6395.00 | 6375.39 | 6382.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 6395.00 | 6375.39 | 6382.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 6383.00 | 6376.91 | 6382.24 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2025-12-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 15:15:00 | 6403.00 | 6388.49 | 6386.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 09:15:00 | 6425.00 | 6395.79 | 6390.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 10:15:00 | 6369.50 | 6390.53 | 6388.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 10:15:00 | 6369.50 | 6390.53 | 6388.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 6369.50 | 6390.53 | 6388.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 11:00:00 | 6369.50 | 6390.53 | 6388.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-12-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 11:15:00 | 6369.50 | 6386.33 | 6386.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 12:15:00 | 6342.50 | 6377.56 | 6382.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-02 14:15:00 | 6393.50 | 6376.66 | 6381.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 14:15:00 | 6393.50 | 6376.66 | 6381.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 14:15:00 | 6393.50 | 6376.66 | 6381.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-02 15:00:00 | 6393.50 | 6376.66 | 6381.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 15:15:00 | 6378.50 | 6377.03 | 6380.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 09:15:00 | 6371.00 | 6377.03 | 6380.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 6348.50 | 6371.32 | 6377.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 10:30:00 | 6316.50 | 6356.86 | 6370.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 12:30:00 | 6321.50 | 6347.53 | 6363.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 14:15:00 | 6321.50 | 6343.72 | 6360.47 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 09:15:00 | 6440.00 | 6360.95 | 6363.95 | SL hit (close>static) qty=1.00 sl=6408.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 10:15:00 | 6454.50 | 6379.66 | 6372.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 11:15:00 | 6482.50 | 6400.23 | 6382.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 09:15:00 | 6431.00 | 6484.28 | 6455.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 6431.00 | 6484.28 | 6455.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 6431.00 | 6484.28 | 6455.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 6431.00 | 6484.28 | 6455.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 6396.50 | 6466.72 | 6449.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:45:00 | 6400.00 | 6466.72 | 6449.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 6405.00 | 6438.18 | 6438.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 6340.00 | 6418.55 | 6429.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 13:15:00 | 6132.00 | 6123.98 | 6195.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 13:30:00 | 6132.50 | 6123.98 | 6195.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 14:15:00 | 6196.00 | 6138.38 | 6195.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 15:00:00 | 6196.00 | 6138.38 | 6195.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 15:15:00 | 6225.50 | 6155.80 | 6198.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:15:00 | 6287.00 | 6155.80 | 6198.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 6266.50 | 6177.94 | 6204.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:30:00 | 6271.00 | 6177.94 | 6204.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 6269.50 | 6196.25 | 6210.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:30:00 | 6263.50 | 6196.25 | 6210.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 12:15:00 | 6249.00 | 6219.16 | 6219.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 6344.50 | 6252.20 | 6234.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 13:15:00 | 6293.50 | 6295.99 | 6268.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 14:00:00 | 6293.50 | 6295.99 | 6268.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 6240.00 | 6284.28 | 6270.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-16 12:15:00 | 6276.50 | 6265.26 | 6263.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 12:15:00 | 6232.00 | 6258.61 | 6260.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — SELL (started 2025-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 12:15:00 | 6232.00 | 6258.61 | 6260.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 13:15:00 | 6222.00 | 6251.29 | 6257.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 15:15:00 | 6251.00 | 6249.26 | 6255.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-16 15:15:00 | 6251.00 | 6249.26 | 6255.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 6251.00 | 6249.26 | 6255.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 6316.50 | 6249.26 | 6255.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — BUY (started 2025-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 09:15:00 | 6304.50 | 6260.31 | 6259.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 13:15:00 | 6349.00 | 6321.45 | 6303.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 09:15:00 | 6507.50 | 6507.78 | 6435.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 14:15:00 | 6495.50 | 6503.78 | 6461.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 6495.50 | 6503.78 | 6461.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 6459.50 | 6503.78 | 6461.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 6442.50 | 6486.60 | 6460.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:45:00 | 6437.50 | 6486.60 | 6460.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 10:15:00 | 6420.00 | 6473.28 | 6456.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 11:00:00 | 6420.00 | 6473.28 | 6456.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 6396.00 | 6444.94 | 6446.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 14:15:00 | 6358.00 | 6427.55 | 6438.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 6235.00 | 6213.76 | 6256.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 15:15:00 | 6235.00 | 6213.76 | 6256.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 6235.00 | 6213.76 | 6256.84 | EMA400 retest candle locked (from downside) |

### Cycle 47 — BUY (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 14:15:00 | 6285.00 | 6255.45 | 6253.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 11:15:00 | 6315.50 | 6273.52 | 6262.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 6211.50 | 6273.00 | 6268.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 09:15:00 | 6211.50 | 6273.00 | 6268.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 6211.50 | 6273.00 | 6268.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:00:00 | 6211.50 | 6273.00 | 6268.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2026-01-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 10:15:00 | 6188.00 | 6256.00 | 6261.52 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 6388.00 | 6271.74 | 6256.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 10:15:00 | 6445.50 | 6306.49 | 6274.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 10:15:00 | 6448.00 | 6454.97 | 6382.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 11:00:00 | 6448.00 | 6454.97 | 6382.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 6410.00 | 6444.28 | 6410.70 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 09:30:00 | 6402.00 | 6444.28 | 6410.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 6440.00 | 6443.42 | 6413.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:30:00 | 6450.00 | 6443.42 | 6413.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 6412.50 | 6437.24 | 6413.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:45:00 | 6402.00 | 6437.24 | 6413.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 6404.00 | 6430.59 | 6412.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-09 12:45:00 | 6405.00 | 6430.59 | 6412.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 13:15:00 | 6423.50 | 6429.17 | 6413.45 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2026-01-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-12 10:15:00 | 6344.50 | 6400.85 | 6404.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 12:15:00 | 6321.00 | 6376.82 | 6392.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 10:15:00 | 6349.00 | 6342.89 | 6366.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 10:30:00 | 6313.50 | 6342.89 | 6366.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 6359.50 | 6340.50 | 6357.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 6359.50 | 6340.50 | 6357.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 6341.50 | 6340.70 | 6355.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 6329.00 | 6340.70 | 6355.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 6412.00 | 6320.72 | 6330.46 | SL hit (close>static) qty=1.00 sl=6360.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 10:15:00 | 6406.50 | 6337.88 | 6337.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 11:15:00 | 6475.50 | 6365.40 | 6349.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 6393.50 | 6418.14 | 6398.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 09:15:00 | 6393.50 | 6418.14 | 6398.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 6393.50 | 6418.14 | 6398.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 09:45:00 | 6361.50 | 6418.14 | 6398.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 6393.50 | 6413.21 | 6397.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 10:30:00 | 6415.00 | 6413.21 | 6397.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 6374.50 | 6405.47 | 6395.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 6374.50 | 6405.47 | 6395.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 6399.00 | 6404.17 | 6396.12 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 14:15:00 | 6340.00 | 6384.27 | 6388.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 6120.00 | 6322.73 | 6358.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 14:15:00 | 6239.00 | 6221.36 | 6284.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 15:00:00 | 6239.00 | 6221.36 | 6284.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 6365.00 | 6249.87 | 6286.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 6353.50 | 6249.87 | 6286.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 6235.50 | 6248.62 | 6279.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:45:00 | 6175.00 | 6248.73 | 6268.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:30:00 | 6171.00 | 6237.68 | 6261.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 13:15:00 | 6153.00 | 6229.25 | 6255.37 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 09:45:00 | 6172.00 | 6178.96 | 6219.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 6213.00 | 6157.41 | 6190.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 15:00:00 | 6213.00 | 6157.41 | 6190.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 6238.50 | 6173.63 | 6194.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:30:00 | 6278.00 | 6193.30 | 6201.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 6260.50 | 6206.74 | 6206.89 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-28 11:15:00 | 6242.50 | 6213.89 | 6210.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 6242.50 | 6213.89 | 6210.13 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 6027.00 | 6172.40 | 6191.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 09:15:00 | 6010.00 | 6068.09 | 6118.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 10:15:00 | 6062.50 | 6031.42 | 6067.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-01 11:00:00 | 6062.50 | 6031.42 | 6067.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 6055.00 | 6036.14 | 6066.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 6055.00 | 6036.14 | 6066.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 6157.00 | 6060.31 | 6074.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:00:00 | 6157.00 | 6060.31 | 6074.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 6110.00 | 6070.25 | 6078.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 15:15:00 | 6055.00 | 6078.70 | 6081.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 10:00:00 | 6079.00 | 6074.97 | 6078.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-02 10:45:00 | 6024.00 | 6062.47 | 6072.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 09:15:00 | 6279.00 | 6103.11 | 6085.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-02-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 09:15:00 | 6279.00 | 6103.11 | 6085.31 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 10:15:00 | 5845.00 | 6108.25 | 6122.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 5827.00 | 5945.42 | 5994.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 14:15:00 | 5857.00 | 5854.35 | 5922.13 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 15:00:00 | 5857.00 | 5854.35 | 5922.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 5856.50 | 5852.48 | 5909.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-09 10:15:00 | 5842.00 | 5852.48 | 5909.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-10 11:15:00 | 5940.00 | 5885.23 | 5895.08 | SL hit (close>static) qty=1.00 sl=5929.50 alert=retest2 |

### Cycle 57 — BUY (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 09:15:00 | 5697.00 | 5561.62 | 5555.97 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2026-02-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 10:15:00 | 5461.00 | 5565.02 | 5573.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 09:15:00 | 5412.50 | 5505.93 | 5537.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 11:15:00 | 4795.50 | 4747.58 | 4871.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 11:45:00 | 4794.00 | 4747.58 | 4871.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 4882.00 | 4771.76 | 4835.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:45:00 | 4878.00 | 4771.76 | 4835.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 4893.50 | 4796.11 | 4841.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:45:00 | 4887.50 | 4796.11 | 4841.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 12:15:00 | 4776.50 | 4796.57 | 4833.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 15:15:00 | 4769.00 | 4790.36 | 4824.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-27 12:45:00 | 4768.50 | 4780.85 | 4806.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 4530.55 | 4757.18 | 4785.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 4771.40 | 4757.18 | 4785.08 | SL hit (close>static) qty=0.50 sl=4757.18 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 4771.20 | 4707.24 | 4703.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-09 10:15:00 | 4806.00 | 4763.73 | 4737.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 12:15:00 | 4755.40 | 4766.52 | 4743.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-09 13:00:00 | 4755.40 | 4766.52 | 4743.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 4787.00 | 4774.76 | 4755.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:30:00 | 4772.00 | 4774.76 | 4755.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 4757.90 | 4775.46 | 4759.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-10 12:00:00 | 4757.90 | 4775.46 | 4759.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 12:15:00 | 4777.00 | 4775.77 | 4760.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 13:45:00 | 4789.80 | 4777.93 | 4763.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 14:30:00 | 4807.10 | 4786.25 | 4768.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 10:00:00 | 4808.00 | 4799.42 | 4777.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-11 12:00:00 | 4795.40 | 4799.33 | 4781.65 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 4776.00 | 4794.67 | 4781.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 4776.00 | 4794.67 | 4781.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 4770.00 | 4789.73 | 4780.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 14:00:00 | 4770.00 | 4789.73 | 4780.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 4734.00 | 4778.59 | 4775.93 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-03-11 14:15:00 | 4734.00 | 4778.59 | 4775.93 | SL hit (close<static) qty=1.00 sl=4750.00 alert=retest2 |

### Cycle 60 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 4747.90 | 4772.45 | 4773.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 4644.60 | 4746.88 | 4761.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-12 13:15:00 | 4726.70 | 4717.87 | 4740.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 13:15:00 | 4726.70 | 4717.87 | 4740.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 13:15:00 | 4726.70 | 4717.87 | 4740.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-12 13:30:00 | 4728.00 | 4717.87 | 4740.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 4715.00 | 4717.30 | 4738.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-13 09:15:00 | 4620.00 | 4719.84 | 4737.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-16 15:15:00 | 4848.90 | 4664.08 | 4667.77 | SL hit (close>static) qty=1.00 sl=4739.60 alert=retest2 |

### Cycle 61 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 4753.80 | 4625.77 | 4625.73 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 4600.00 | 4646.36 | 4646.87 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-03-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 10:15:00 | 4729.10 | 4657.53 | 4648.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 15:15:00 | 4739.00 | 4700.26 | 4675.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 15:15:00 | 4715.00 | 4730.81 | 4707.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-23 15:15:00 | 4715.00 | 4730.81 | 4707.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 15:15:00 | 4715.00 | 4730.81 | 4707.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 09:15:00 | 4758.20 | 4730.81 | 4707.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-24 11:15:00 | 4790.50 | 4732.81 | 4712.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-02 13:15:00 | 5234.02 | 5105.44 | 5030.92 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 13:15:00 | 5442.00 | 5456.25 | 5457.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 11:15:00 | 5419.40 | 5449.48 | 5453.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 15:15:00 | 5369.00 | 5348.44 | 5377.96 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-04-22 09:15:00 | 5117.30 | 5348.44 | 5377.96 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-24 11:15:00 | 4861.44 | 4990.23 | 5088.72 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2026-04-27 14:15:00 | 4820.00 | 4819.54 | 4906.54 | SL hit (close>ema200) qty=0.50 sl=4819.54 alert=retest1 |

### Cycle 65 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 4836.70 | 4810.14 | 4808.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 4949.80 | 4841.46 | 4823.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 09:15:00 | 4957.00 | 4959.06 | 4905.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 10:00:00 | 4957.00 | 4959.06 | 4905.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 4953.20 | 4953.67 | 4912.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 12:30:00 | 4978.10 | 4959.19 | 4918.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-14 15:15:00 | 5768.00 | 2025-05-15 09:15:00 | 5584.50 | STOP_HIT | 1.00 | -3.18% |
| SELL | retest2 | 2025-05-23 11:30:00 | 5680.00 | 2025-05-23 12:15:00 | 5687.00 | STOP_HIT | 1.00 | -0.12% |
| SELL | retest2 | 2025-06-02 09:15:00 | 5553.50 | 2025-06-05 11:15:00 | 5633.00 | STOP_HIT | 1.00 | -1.43% |
| BUY | retest2 | 2025-06-12 11:15:00 | 5934.00 | 2025-06-19 10:15:00 | 5877.00 | STOP_HIT | 1.00 | -0.96% |
| BUY | retest2 | 2025-06-12 14:15:00 | 5919.00 | 2025-06-19 10:15:00 | 5877.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-06-13 10:15:00 | 5940.00 | 2025-06-19 10:15:00 | 5877.00 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-06-13 12:15:00 | 5920.50 | 2025-06-19 10:15:00 | 5877.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-06-16 11:15:00 | 5964.00 | 2025-06-19 10:15:00 | 5877.00 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2025-06-25 09:15:00 | 6142.00 | 2025-06-26 09:15:00 | 6043.50 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-06-25 12:45:00 | 6109.00 | 2025-06-26 09:15:00 | 6043.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-07-02 13:15:00 | 5990.00 | 2025-07-10 09:15:00 | 5690.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-03 11:30:00 | 5991.00 | 2025-07-10 09:15:00 | 5691.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-03 12:00:00 | 5986.50 | 2025-07-10 09:15:00 | 5687.18 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-02 13:15:00 | 5990.00 | 2025-07-14 14:15:00 | 5612.00 | STOP_HIT | 0.50 | 6.31% |
| SELL | retest2 | 2025-07-03 11:30:00 | 5991.00 | 2025-07-14 14:15:00 | 5612.00 | STOP_HIT | 0.50 | 6.33% |
| SELL | retest2 | 2025-07-03 12:00:00 | 5986.50 | 2025-07-14 14:15:00 | 5612.00 | STOP_HIT | 0.50 | 6.26% |
| SELL | retest2 | 2025-07-30 10:30:00 | 5101.50 | 2025-08-04 15:15:00 | 5175.00 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-07-31 09:30:00 | 5109.00 | 2025-08-04 15:15:00 | 5175.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-07-31 10:00:00 | 5103.50 | 2025-08-04 15:15:00 | 5175.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-08-01 11:30:00 | 5097.50 | 2025-08-04 15:15:00 | 5175.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-08-22 09:30:00 | 5378.50 | 2025-08-28 09:15:00 | 5335.50 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-08-22 10:30:00 | 5376.00 | 2025-08-28 09:15:00 | 5335.50 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-08-22 11:45:00 | 5375.00 | 2025-08-28 09:15:00 | 5335.50 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-08-22 12:15:00 | 5383.50 | 2025-08-28 09:15:00 | 5335.50 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-08-25 09:15:00 | 5475.00 | 2025-08-28 09:15:00 | 5335.50 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-09-16 09:45:00 | 5391.00 | 2025-09-22 09:15:00 | 5280.00 | STOP_HIT | 1.00 | -2.06% |
| BUY | retest2 | 2025-09-16 14:00:00 | 5390.00 | 2025-09-22 09:15:00 | 5280.00 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2025-10-15 09:15:00 | 5676.80 | 2025-10-28 11:15:00 | 5816.60 | STOP_HIT | 1.00 | 2.46% |
| BUY | retest2 | 2025-10-31 14:00:00 | 5915.20 | 2025-11-03 09:15:00 | 5856.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-11-03 11:00:00 | 5899.50 | 2025-11-04 10:15:00 | 5874.50 | STOP_HIT | 1.00 | -0.42% |
| BUY | retest2 | 2025-11-03 12:30:00 | 5911.00 | 2025-11-04 10:15:00 | 5874.50 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-11-06 15:15:00 | 5831.50 | 2025-11-10 13:15:00 | 5864.00 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-11-14 14:30:00 | 6063.50 | 2025-11-18 15:15:00 | 6065.00 | STOP_HIT | 1.00 | 0.02% |
| BUY | retest2 | 2025-11-14 15:00:00 | 6108.00 | 2025-11-18 15:15:00 | 6065.00 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-11-24 09:15:00 | 6389.00 | 2025-11-28 13:15:00 | 6370.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-12-03 10:30:00 | 6316.50 | 2025-12-04 09:15:00 | 6440.00 | STOP_HIT | 1.00 | -1.96% |
| SELL | retest2 | 2025-12-03 12:30:00 | 6321.50 | 2025-12-04 09:15:00 | 6440.00 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-12-03 14:15:00 | 6321.50 | 2025-12-04 09:15:00 | 6440.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2025-12-16 12:15:00 | 6276.50 | 2025-12-16 12:15:00 | 6232.00 | STOP_HIT | 1.00 | -0.71% |
| SELL | retest2 | 2026-01-14 09:15:00 | 6329.00 | 2026-01-16 09:15:00 | 6412.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-01-23 10:45:00 | 6175.00 | 2026-01-28 11:15:00 | 6242.50 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2026-01-23 11:30:00 | 6171.00 | 2026-01-28 11:15:00 | 6242.50 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-01-23 13:15:00 | 6153.00 | 2026-01-28 11:15:00 | 6242.50 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2026-01-27 09:45:00 | 6172.00 | 2026-01-28 11:15:00 | 6242.50 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2026-02-01 15:15:00 | 6055.00 | 2026-02-03 09:15:00 | 6279.00 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2026-02-02 10:00:00 | 6079.00 | 2026-02-03 09:15:00 | 6279.00 | STOP_HIT | 1.00 | -3.29% |
| SELL | retest2 | 2026-02-02 10:45:00 | 6024.00 | 2026-02-03 09:15:00 | 6279.00 | STOP_HIT | 1.00 | -4.23% |
| SELL | retest2 | 2026-02-09 10:15:00 | 5842.00 | 2026-02-10 11:15:00 | 5940.00 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-02-11 09:15:00 | 5839.00 | 2026-02-12 09:15:00 | 5547.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 09:15:00 | 5839.00 | 2026-02-13 09:15:00 | 5255.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2026-02-26 15:15:00 | 4769.00 | 2026-03-02 09:15:00 | 4530.55 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 15:15:00 | 4769.00 | 2026-03-02 09:15:00 | 4771.40 | STOP_HIT | 0.50 | -0.05% |
| SELL | retest2 | 2026-02-27 12:45:00 | 4768.50 | 2026-03-02 09:15:00 | 4530.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-27 12:45:00 | 4768.50 | 2026-03-02 09:15:00 | 4771.40 | STOP_HIT | 0.50 | -0.06% |
| SELL | retest2 | 2026-03-02 10:15:00 | 4760.00 | 2026-03-06 09:15:00 | 4754.30 | STOP_HIT | 1.00 | 0.12% |
| SELL | retest2 | 2026-03-02 11:00:00 | 4722.70 | 2026-03-06 11:15:00 | 4771.20 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2026-03-05 09:30:00 | 4693.60 | 2026-03-06 11:15:00 | 4771.20 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2026-03-10 13:45:00 | 4789.80 | 2026-03-11 14:15:00 | 4734.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2026-03-10 14:30:00 | 4807.10 | 2026-03-11 14:15:00 | 4734.00 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2026-03-11 10:00:00 | 4808.00 | 2026-03-11 14:15:00 | 4734.00 | STOP_HIT | 1.00 | -1.54% |
| BUY | retest2 | 2026-03-11 12:00:00 | 4795.40 | 2026-03-11 14:15:00 | 4734.00 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2026-03-13 09:15:00 | 4620.00 | 2026-03-16 15:15:00 | 4848.90 | STOP_HIT | 1.00 | -4.95% |
| SELL | retest2 | 2026-03-17 09:15:00 | 4557.90 | 2026-03-18 10:15:00 | 4753.80 | STOP_HIT | 1.00 | -4.30% |
| BUY | retest2 | 2026-03-24 09:15:00 | 4758.20 | 2026-04-02 13:15:00 | 5234.02 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-24 11:15:00 | 4790.50 | 2026-04-06 09:15:00 | 5269.55 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2026-04-22 09:15:00 | 5117.30 | 2026-04-24 11:15:00 | 4861.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2026-04-22 09:15:00 | 5117.30 | 2026-04-27 14:15:00 | 4820.00 | STOP_HIT | 0.50 | 5.81% |
| SELL | retest2 | 2026-04-29 11:15:00 | 4814.00 | 2026-05-05 12:15:00 | 4836.70 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2026-04-29 12:00:00 | 4815.50 | 2026-05-05 12:15:00 | 4836.70 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2026-04-29 13:30:00 | 4799.00 | 2026-05-05 12:15:00 | 4836.70 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2026-05-05 10:15:00 | 4816.70 | 2026-05-05 12:15:00 | 4836.70 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2026-05-05 11:30:00 | 4801.30 | 2026-05-05 12:15:00 | 4836.70 | STOP_HIT | 1.00 | -0.74% |
