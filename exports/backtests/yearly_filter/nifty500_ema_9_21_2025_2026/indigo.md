# InterGlobe Aviation Ltd. (INDIGO)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 4522.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 71 |
| ALERT1 | 58 |
| ALERT2 | 56 |
| ALERT2_SKIP | 31 |
| ALERT3 | 118 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 57 |
| PARTIAL | 1 |
| TARGET_HIT | 2 |
| STOP_HIT | 55 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 58 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 43
- **Target hits / Stop hits / Partials:** 2 / 55 / 1
- **Avg / median % per leg:** -0.33% / -0.68%
- **Sum % (uncompounded):** -19.13%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 37 | 11 | 29.7% | 2 | 35 | 0 | -0.17% | -6.3% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 37 | 11 | 29.7% | 2 | 35 | 0 | -0.17% | -6.3% |
| SELL (all) | 21 | 4 | 19.0% | 0 | 20 | 1 | -0.61% | -12.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 21 | 4 | 19.0% | 0 | 20 | 1 | -0.61% | -12.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 58 | 15 | 25.9% | 2 | 55 | 1 | -0.33% | -19.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 5474.50 | 5236.15 | 5232.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 5516.00 | 5292.12 | 5257.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 12:15:00 | 5493.50 | 5498.30 | 5444.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 13:00:00 | 5493.50 | 5498.30 | 5444.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 14:15:00 | 5449.00 | 5481.51 | 5445.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:15:00 | 5479.50 | 5475.01 | 5445.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 12:15:00 | 5505.00 | 5550.70 | 5553.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 5505.00 | 5550.70 | 5553.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 13:15:00 | 5452.50 | 5531.06 | 5544.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 5473.00 | 5460.35 | 5491.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 15:00:00 | 5473.00 | 5460.35 | 5491.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 5457.50 | 5459.16 | 5485.92 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 5515.00 | 5497.43 | 5495.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-23 13:15:00 | 5526.00 | 5503.15 | 5498.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 15:15:00 | 5503.50 | 5506.47 | 5500.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 5489.00 | 5502.98 | 5499.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 5489.00 | 5502.98 | 5499.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-26 10:00:00 | 5489.00 | 5502.98 | 5499.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 10:15:00 | 5443.00 | 5490.98 | 5494.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-27 09:15:00 | 5289.00 | 5418.75 | 5454.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-28 12:15:00 | 5345.00 | 5328.80 | 5369.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-28 13:00:00 | 5345.00 | 5328.80 | 5369.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 5333.50 | 5315.91 | 5332.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-02 09:45:00 | 5302.00 | 5319.39 | 5328.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 15:15:00 | 5346.00 | 5331.69 | 5331.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 15:15:00 | 5346.00 | 5331.69 | 5331.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-03 09:15:00 | 5423.00 | 5349.95 | 5339.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 15:15:00 | 5379.00 | 5381.48 | 5364.16 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-04 09:15:00 | 5405.50 | 5381.48 | 5364.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 5402.50 | 5385.68 | 5367.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 10:45:00 | 5419.00 | 5392.54 | 5372.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 09:15:00 | 5460.00 | 5404.51 | 5387.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 10:15:00 | 5502.50 | 5582.11 | 5582.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 5502.50 | 5582.11 | 5582.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 5497.50 | 5565.19 | 5574.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 10:15:00 | 5320.00 | 5310.85 | 5389.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 11:00:00 | 5320.00 | 5310.85 | 5389.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 5404.00 | 5334.73 | 5375.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 5404.00 | 5334.73 | 5375.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 5389.50 | 5345.68 | 5376.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 5370.50 | 5345.68 | 5376.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 12:15:00 | 5359.50 | 5316.06 | 5312.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 12:15:00 | 5359.50 | 5316.06 | 5312.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 14:15:00 | 5388.50 | 5337.90 | 5323.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 09:15:00 | 5331.00 | 5344.86 | 5329.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 09:15:00 | 5331.00 | 5344.86 | 5329.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 09:15:00 | 5331.00 | 5344.86 | 5329.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 10:30:00 | 5363.00 | 5347.99 | 5332.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-23 11:00:00 | 5360.50 | 5347.99 | 5332.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-06-30 09:15:00 | 5899.30 | 5812.72 | 5733.84 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-03 11:15:00 | 5826.50 | 5887.61 | 5893.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-03 13:15:00 | 5783.50 | 5854.05 | 5876.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-07 09:15:00 | 5776.00 | 5774.21 | 5808.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 12:15:00 | 5783.00 | 5761.53 | 5777.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 5783.00 | 5761.53 | 5777.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:45:00 | 5782.00 | 5761.53 | 5777.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 5782.00 | 5765.62 | 5777.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:30:00 | 5785.00 | 5765.62 | 5777.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 5799.00 | 5772.30 | 5779.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 15:00:00 | 5799.00 | 5772.30 | 5779.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 5800.00 | 5777.84 | 5781.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 5801.50 | 5777.84 | 5781.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-07-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 09:15:00 | 5836.50 | 5789.57 | 5786.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 10:15:00 | 5840.50 | 5799.76 | 5791.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 12:15:00 | 5842.50 | 5849.52 | 5829.22 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-10 12:45:00 | 5846.50 | 5849.52 | 5829.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 5923.00 | 5860.77 | 5840.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 5940.00 | 5902.82 | 5873.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 11:00:00 | 5944.00 | 5913.16 | 5883.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 14:30:00 | 5942.00 | 5925.11 | 5899.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-15 09:30:00 | 5945.50 | 5930.21 | 5906.17 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 5917.00 | 5932.22 | 5919.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:15:00 | 5905.50 | 5932.22 | 5919.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 09:15:00 | 5929.00 | 5931.58 | 5919.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 10:15:00 | 5940.50 | 5931.58 | 5919.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:30:00 | 5932.00 | 5925.24 | 5920.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 14:15:00 | 5943.50 | 5925.24 | 5920.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 09:15:00 | 5806.00 | 5906.99 | 5913.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 09:15:00 | 5806.00 | 5906.99 | 5913.83 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2025-07-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-21 12:15:00 | 5885.50 | 5867.35 | 5866.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-22 10:15:00 | 5915.00 | 5884.69 | 5875.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 09:15:00 | 5869.00 | 5910.63 | 5895.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 09:15:00 | 5869.00 | 5910.63 | 5895.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 5869.00 | 5910.63 | 5895.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 09:45:00 | 5826.50 | 5910.63 | 5895.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 5848.00 | 5898.11 | 5891.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:00:00 | 5848.00 | 5898.11 | 5891.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 5882.00 | 5889.61 | 5888.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:00:00 | 5882.00 | 5889.61 | 5888.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 5895.00 | 5890.69 | 5889.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:30:00 | 5883.50 | 5890.69 | 5889.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 5890.50 | 5890.65 | 5889.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 5915.00 | 5890.65 | 5889.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 5918.00 | 5896.12 | 5892.01 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 11:15:00 | 5868.00 | 5887.12 | 5888.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-24 14:15:00 | 5839.00 | 5874.87 | 5882.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-28 09:15:00 | 5845.00 | 5817.36 | 5838.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 5845.00 | 5817.36 | 5838.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 5845.00 | 5817.36 | 5838.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 10:00:00 | 5845.00 | 5817.36 | 5838.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 5843.50 | 5822.59 | 5839.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 11:45:00 | 5809.50 | 5820.07 | 5836.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 12:00:00 | 5820.00 | 5768.64 | 5770.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 12:15:00 | 5872.50 | 5789.41 | 5779.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 12:15:00 | 5872.50 | 5789.41 | 5779.40 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-08-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 10:15:00 | 5759.00 | 5799.24 | 5804.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 09:15:00 | 5730.00 | 5775.38 | 5789.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-05 12:15:00 | 5778.00 | 5770.15 | 5782.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-05 12:15:00 | 5778.00 | 5770.15 | 5782.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 5778.00 | 5770.15 | 5782.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:45:00 | 5780.50 | 5770.15 | 5782.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 13:15:00 | 5762.00 | 5768.52 | 5781.07 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2025-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 09:15:00 | 5855.00 | 5787.20 | 5786.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-06 10:15:00 | 5885.00 | 5806.76 | 5795.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-07 09:15:00 | 5833.00 | 5850.86 | 5828.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 09:15:00 | 5833.00 | 5850.86 | 5828.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 5833.00 | 5850.86 | 5828.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 10:00:00 | 5833.00 | 5850.86 | 5828.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 10:15:00 | 5843.00 | 5849.29 | 5829.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 15:15:00 | 5884.00 | 5845.77 | 5833.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-08 09:15:00 | 5825.50 | 5847.83 | 5837.00 | SL hit (close<static) qty=1.00 sl=5826.50 alert=retest2 |

### Cycle 16 — SELL (started 2025-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 12:15:00 | 5792.00 | 5827.47 | 5829.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 5772.50 | 5810.64 | 5821.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 09:15:00 | 5849.00 | 5812.45 | 5819.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-11 09:15:00 | 5849.00 | 5812.45 | 5819.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 5849.00 | 5812.45 | 5819.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:00:00 | 5849.00 | 5812.45 | 5819.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 5875.50 | 5825.06 | 5824.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 11:15:00 | 5888.50 | 5837.75 | 5830.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 10:15:00 | 5965.00 | 5965.44 | 5936.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 11:00:00 | 5965.00 | 5965.44 | 5936.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 6055.00 | 6097.32 | 6051.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:45:00 | 6026.50 | 6097.32 | 6051.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 6050.50 | 6087.96 | 6051.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 6050.50 | 6087.96 | 6051.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 11:15:00 | 6059.50 | 6082.26 | 6051.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 09:30:00 | 6097.00 | 6073.92 | 6057.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:30:00 | 6086.00 | 6118.19 | 6092.67 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 15:15:00 | 6080.00 | 6093.28 | 6093.37 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-08-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 15:15:00 | 6080.00 | 6093.28 | 6093.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 6036.50 | 6081.93 | 6088.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 12:15:00 | 6092.00 | 6078.04 | 6084.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 12:15:00 | 6092.00 | 6078.04 | 6084.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 12:15:00 | 6092.00 | 6078.04 | 6084.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:00:00 | 6092.00 | 6078.04 | 6084.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 13:15:00 | 6103.00 | 6083.03 | 6085.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 13:45:00 | 6102.50 | 6083.03 | 6085.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 14:15:00 | 6115.00 | 6089.43 | 6088.62 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 10:15:00 | 6067.00 | 6086.45 | 6087.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 14:15:00 | 6036.00 | 6072.01 | 6080.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 10:15:00 | 5689.50 | 5686.23 | 5775.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 11:00:00 | 5689.50 | 5686.23 | 5775.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 5712.50 | 5683.66 | 5700.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 12:45:00 | 5711.00 | 5683.66 | 5700.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 5731.50 | 5693.23 | 5703.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:00:00 | 5731.50 | 5693.23 | 5703.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 21 — BUY (started 2025-09-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 15:15:00 | 5740.50 | 5710.25 | 5709.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-04 09:15:00 | 5772.00 | 5722.60 | 5715.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 13:15:00 | 5711.00 | 5722.10 | 5717.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 13:15:00 | 5711.00 | 5722.10 | 5717.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 13:15:00 | 5711.00 | 5722.10 | 5717.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 14:00:00 | 5711.00 | 5722.10 | 5717.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 5689.00 | 5715.48 | 5715.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 5689.00 | 5715.48 | 5715.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-09-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 15:15:00 | 5691.00 | 5710.59 | 5712.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 09:15:00 | 5678.50 | 5704.17 | 5709.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 5707.00 | 5681.78 | 5691.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 5707.00 | 5681.78 | 5691.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 5707.00 | 5681.78 | 5691.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 5707.00 | 5681.78 | 5691.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 5715.50 | 5688.52 | 5693.47 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2025-09-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 12:15:00 | 5735.00 | 5702.77 | 5699.40 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 09:15:00 | 5673.00 | 5694.76 | 5696.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-09 10:15:00 | 5659.50 | 5687.71 | 5693.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 5645.00 | 5633.30 | 5649.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 5645.00 | 5633.30 | 5649.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 5645.00 | 5633.30 | 5649.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:45:00 | 5642.00 | 5633.30 | 5649.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 5613.00 | 5629.24 | 5646.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 11:30:00 | 5605.00 | 5627.79 | 5644.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-11 13:15:00 | 5666.50 | 5638.69 | 5646.35 | SL hit (close>static) qty=1.00 sl=5650.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 09:15:00 | 5749.50 | 5668.73 | 5658.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 10:15:00 | 5765.00 | 5687.99 | 5668.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 14:15:00 | 5730.00 | 5734.42 | 5715.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 15:15:00 | 5721.00 | 5734.42 | 5715.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 5715.50 | 5728.49 | 5715.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:30:00 | 5750.50 | 5727.31 | 5720.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 11:00:00 | 5748.50 | 5731.55 | 5722.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 11:30:00 | 5746.00 | 5734.74 | 5725.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 12:30:00 | 5746.50 | 5736.59 | 5726.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 5733.50 | 5738.50 | 5731.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:45:00 | 5736.00 | 5738.50 | 5731.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 5725.50 | 5735.90 | 5731.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:00:00 | 5725.50 | 5735.90 | 5731.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 5718.00 | 5732.32 | 5729.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:45:00 | 5721.50 | 5732.32 | 5729.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-18 14:15:00 | 5716.00 | 5728.60 | 5728.68 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 14:15:00 | 5716.00 | 5728.60 | 5728.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 5693.50 | 5718.77 | 5724.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 09:15:00 | 5743.00 | 5698.30 | 5706.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 5743.00 | 5698.30 | 5706.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 5743.00 | 5698.30 | 5706.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:00:00 | 5743.00 | 5698.30 | 5706.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 10:15:00 | 5745.00 | 5707.64 | 5709.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:30:00 | 5740.00 | 5707.64 | 5709.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-09-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 11:15:00 | 5747.00 | 5715.51 | 5713.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 12:15:00 | 5768.50 | 5726.11 | 5718.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-23 13:15:00 | 5748.00 | 5753.41 | 5740.12 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-23 14:15:00 | 5745.50 | 5753.41 | 5740.12 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 5732.00 | 5749.13 | 5739.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 15:00:00 | 5732.00 | 5749.13 | 5739.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 15:15:00 | 5730.50 | 5745.40 | 5738.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 09:15:00 | 5725.00 | 5745.40 | 5738.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 10:15:00 | 5722.50 | 5740.20 | 5737.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-24 11:00:00 | 5722.50 | 5740.20 | 5737.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 28 — SELL (started 2025-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 11:15:00 | 5700.50 | 5732.26 | 5734.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 14:15:00 | 5691.00 | 5716.48 | 5725.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 13:15:00 | 5697.00 | 5693.08 | 5707.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 13:15:00 | 5697.00 | 5693.08 | 5707.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 5697.00 | 5693.08 | 5707.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:00:00 | 5697.00 | 5693.08 | 5707.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 5665.50 | 5687.57 | 5703.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 5654.00 | 5685.65 | 5701.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 10:00:00 | 5637.50 | 5676.02 | 5695.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 5756.50 | 5596.08 | 5615.96 | SL hit (close>static) qty=1.00 sl=5708.50 alert=retest2 |

### Cycle 29 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 5665.00 | 5602.66 | 5601.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 12:15:00 | 5688.00 | 5649.09 | 5633.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 09:15:00 | 5639.50 | 5665.80 | 5648.25 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-07 09:15:00 | 5639.50 | 5665.80 | 5648.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 5639.50 | 5665.80 | 5648.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 09:45:00 | 5634.00 | 5665.80 | 5648.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 10:15:00 | 5647.50 | 5662.14 | 5648.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 15:15:00 | 5665.50 | 5657.76 | 5650.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 10:15:00 | 5671.50 | 5656.25 | 5650.78 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 14:15:00 | 5633.00 | 5648.83 | 5649.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-10-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 14:15:00 | 5633.00 | 5648.83 | 5649.09 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 10:15:00 | 5714.00 | 5661.00 | 5654.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 12:15:00 | 5729.00 | 5684.68 | 5666.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 09:15:00 | 5772.00 | 5774.20 | 5749.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 10:15:00 | 5762.00 | 5774.20 | 5749.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 5750.50 | 5767.19 | 5750.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:00:00 | 5750.50 | 5767.19 | 5750.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 5771.00 | 5767.95 | 5752.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-14 13:30:00 | 5777.00 | 5770.06 | 5754.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 5797.50 | 5767.60 | 5756.11 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 12:15:00 | 5841.00 | 5881.89 | 5884.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 12:15:00 | 5841.00 | 5881.89 | 5884.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 13:15:00 | 5829.50 | 5871.41 | 5879.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 14:15:00 | 5774.50 | 5767.71 | 5810.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-24 15:00:00 | 5774.50 | 5767.71 | 5810.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 5810.00 | 5775.81 | 5806.74 | EMA400 retest candle locked (from downside) |

### Cycle 33 — BUY (started 2025-10-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 13:15:00 | 5861.00 | 5820.71 | 5820.60 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 09:15:00 | 5793.50 | 5820.41 | 5820.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 13:15:00 | 5750.00 | 5795.85 | 5808.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 14:15:00 | 5816.50 | 5799.98 | 5809.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 14:15:00 | 5816.50 | 5799.98 | 5809.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 5816.50 | 5799.98 | 5809.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 15:00:00 | 5816.50 | 5799.98 | 5809.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 15:15:00 | 5834.50 | 5806.88 | 5811.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 09:15:00 | 5791.00 | 5806.88 | 5811.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 14:30:00 | 5804.00 | 5810.08 | 5810.25 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-29 15:15:00 | 5822.00 | 5812.47 | 5811.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-10-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 15:15:00 | 5822.00 | 5812.47 | 5811.32 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 09:15:00 | 5776.50 | 5805.27 | 5808.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 11:15:00 | 5762.00 | 5791.94 | 5801.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 10:15:00 | 5675.50 | 5662.51 | 5701.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 11:00:00 | 5675.50 | 5662.51 | 5701.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 5692.00 | 5669.44 | 5692.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 5692.00 | 5669.44 | 5692.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 5700.00 | 5675.55 | 5692.81 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:15:00 | 5679.50 | 5675.55 | 5692.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 10:15:00 | 5736.50 | 5667.88 | 5672.32 | SL hit (close>static) qty=1.00 sl=5703.00 alert=retest2 |

### Cycle 37 — BUY (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 11:15:00 | 5767.00 | 5687.70 | 5680.93 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 10:15:00 | 5641.50 | 5677.29 | 5680.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-07 11:15:00 | 5608.00 | 5663.43 | 5674.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 11:15:00 | 5624.50 | 5612.67 | 5636.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 11:15:00 | 5624.50 | 5612.67 | 5636.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 11:15:00 | 5624.50 | 5612.67 | 5636.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:45:00 | 5631.00 | 5612.67 | 5636.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 12:15:00 | 5631.00 | 5616.34 | 5636.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 12:30:00 | 5636.50 | 5616.34 | 5636.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 13:15:00 | 5604.50 | 5613.97 | 5633.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 13:30:00 | 5614.00 | 5613.97 | 5633.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 5649.00 | 5613.35 | 5627.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 09:45:00 | 5638.00 | 5613.35 | 5627.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 5670.00 | 5624.68 | 5631.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 10:45:00 | 5672.50 | 5624.68 | 5631.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — BUY (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 11:15:00 | 5698.00 | 5639.35 | 5637.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 12:15:00 | 5722.00 | 5655.88 | 5645.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 10:15:00 | 5878.50 | 5883.25 | 5830.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-14 10:45:00 | 5884.50 | 5883.25 | 5830.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 5842.50 | 5880.18 | 5855.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 11:00:00 | 5842.50 | 5880.18 | 5855.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 5867.00 | 5877.55 | 5856.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 13:15:00 | 5875.00 | 5876.54 | 5858.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 14:00:00 | 5876.00 | 5876.43 | 5859.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 15:15:00 | 5878.00 | 5874.54 | 5860.45 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 5796.00 | 5859.39 | 5856.05 | SL hit (close<static) qty=1.00 sl=5841.50 alert=retest2 |

### Cycle 40 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 5767.50 | 5841.01 | 5848.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 5740.50 | 5792.12 | 5819.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 14:15:00 | 5764.50 | 5750.99 | 5779.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 15:00:00 | 5764.50 | 5750.99 | 5779.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 5798.50 | 5763.21 | 5780.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-20 09:30:00 | 5803.00 | 5763.21 | 5780.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 10:15:00 | 5791.00 | 5768.77 | 5781.40 | EMA400 retest candle locked (from downside) |

### Cycle 41 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 5830.00 | 5794.84 | 5791.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-21 12:15:00 | 5847.00 | 5812.68 | 5801.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-24 13:15:00 | 5846.50 | 5847.94 | 5830.05 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-24 14:00:00 | 5846.50 | 5847.94 | 5830.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 5807.00 | 5839.75 | 5827.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 5807.00 | 5839.75 | 5827.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 5805.50 | 5832.90 | 5825.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:15:00 | 5759.00 | 5832.90 | 5825.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — SELL (started 2025-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-25 10:15:00 | 5785.00 | 5814.06 | 5817.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 11:15:00 | 5763.50 | 5803.95 | 5813.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 5838.50 | 5798.56 | 5805.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 5838.50 | 5798.56 | 5805.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 5838.50 | 5798.56 | 5805.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 09:45:00 | 5841.50 | 5798.56 | 5805.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 5881.00 | 5815.05 | 5812.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 11:15:00 | 5890.50 | 5830.14 | 5819.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 5902.00 | 5908.69 | 5883.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 13:15:00 | 5891.00 | 5905.07 | 5890.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 5891.00 | 5905.07 | 5890.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:00:00 | 5891.00 | 5905.07 | 5890.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 5899.00 | 5903.86 | 5890.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-28 14:30:00 | 5895.00 | 5903.86 | 5890.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 5880.00 | 5898.95 | 5890.93 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 11:15:00 | 5792.50 | 5871.99 | 5879.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-01 15:15:00 | 5772.50 | 5824.15 | 5852.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 14:15:00 | 4974.00 | 4948.25 | 5089.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 15:00:00 | 4974.00 | 4948.25 | 5089.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 4894.00 | 4835.25 | 4889.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:45:00 | 4882.00 | 4835.25 | 4889.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 4881.00 | 4844.40 | 4888.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 11:45:00 | 4842.00 | 4842.62 | 4883.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 4941.00 | 4868.51 | 4880.52 | SL hit (close>static) qty=1.00 sl=4894.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 10:15:00 | 4994.00 | 4893.60 | 4890.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 10:15:00 | 5055.00 | 4991.33 | 4971.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 15:15:00 | 5113.00 | 5121.91 | 5079.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-22 09:15:00 | 5160.00 | 5121.91 | 5079.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 5155.00 | 5153.08 | 5135.68 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 5064.50 | 5115.04 | 5121.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 09:15:00 | 5059.50 | 5094.54 | 5109.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 14:15:00 | 5071.00 | 5068.93 | 5088.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-26 15:00:00 | 5071.00 | 5068.93 | 5088.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 5089.50 | 5072.57 | 5087.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 09:30:00 | 5092.00 | 5072.57 | 5087.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 5087.00 | 5075.46 | 5087.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 11:15:00 | 5083.00 | 5075.46 | 5087.01 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-29 14:45:00 | 5083.50 | 5079.99 | 5085.75 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 09:15:00 | 5066.50 | 5080.99 | 5085.69 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 09:15:00 | 5112.50 | 5054.81 | 5050.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2026-01-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 09:15:00 | 5112.50 | 5054.81 | 5050.66 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 5051.50 | 5092.21 | 5094.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 5026.00 | 5064.51 | 5079.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 10:15:00 | 4977.00 | 4968.68 | 5002.20 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-08 10:30:00 | 4984.50 | 4968.68 | 5002.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 4902.50 | 4772.19 | 4772.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 4902.50 | 4772.19 | 4772.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — BUY (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 10:15:00 | 4924.00 | 4802.55 | 4786.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 14:15:00 | 4943.00 | 4877.16 | 4831.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 09:15:00 | 4861.00 | 4883.50 | 4842.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 09:45:00 | 4854.00 | 4883.50 | 4842.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 10:15:00 | 4856.50 | 4878.10 | 4843.84 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2026-01-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 15:15:00 | 4795.00 | 4825.75 | 4828.30 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-01-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 12:15:00 | 4896.00 | 4840.58 | 4834.13 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 4762.00 | 4860.02 | 4861.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 4705.00 | 4812.53 | 4838.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 14:15:00 | 4761.00 | 4739.38 | 4772.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 14:15:00 | 4761.00 | 4739.38 | 4772.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 14:15:00 | 4761.00 | 4739.38 | 4772.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 14:45:00 | 4776.50 | 4739.38 | 4772.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 4763.50 | 4744.21 | 4771.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 4775.50 | 4744.21 | 4771.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 4769.00 | 4749.16 | 4771.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 13:15:00 | 4729.50 | 4757.55 | 4770.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 09:15:00 | 4676.50 | 4752.29 | 4764.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 14:15:00 | 4684.70 | 4638.82 | 4634.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 4684.70 | 4638.82 | 4634.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 4914.80 | 4702.20 | 4664.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 4854.00 | 4922.15 | 4863.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 4854.00 | 4922.15 | 4863.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 4854.00 | 4922.15 | 4863.35 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 13:30:00 | 4883.80 | 4896.13 | 4867.76 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 10:15:00 | 4883.90 | 4903.03 | 4878.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:00:00 | 4887.30 | 4899.88 | 4879.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:30:00 | 4902.50 | 4900.61 | 4881.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 09:15:00 | 4955.90 | 4913.27 | 4894.96 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 10:15:00 | 4963.60 | 4913.27 | 4894.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 12:00:00 | 4960.00 | 4931.49 | 4906.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 13:15:00 | 4964.00 | 4936.87 | 4911.64 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 14:15:00 | 4963.00 | 4940.22 | 4915.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 4963.50 | 4954.77 | 4937.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:30:00 | 4942.70 | 4954.77 | 4937.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 09:15:00 | 4998.50 | 4998.98 | 4975.96 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-13 10:15:00 | 4916.70 | 4963.66 | 4969.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — SELL (started 2026-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 10:15:00 | 4916.70 | 4963.66 | 4969.18 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2026-02-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 10:15:00 | 4968.40 | 4956.74 | 4955.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 12:15:00 | 4991.80 | 4966.06 | 4960.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 15:15:00 | 4966.40 | 4969.05 | 4963.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 09:15:00 | 4971.80 | 4969.60 | 4964.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 4971.80 | 4969.60 | 4964.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:45:00 | 4968.90 | 4969.60 | 4964.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 4980.00 | 4980.61 | 4972.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-18 14:45:00 | 4975.50 | 4980.61 | 4972.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 4972.00 | 4978.89 | 4972.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:15:00 | 4891.00 | 4978.89 | 4972.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 4892.40 | 4961.59 | 4965.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 13:15:00 | 4851.50 | 4914.07 | 4939.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 12:15:00 | 4861.00 | 4853.47 | 4891.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-20 13:00:00 | 4861.00 | 4853.47 | 4891.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 4866.20 | 4851.84 | 4867.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:00:00 | 4866.20 | 4851.84 | 4867.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 4855.00 | 4852.47 | 4866.47 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2026-02-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 09:15:00 | 4940.90 | 4869.30 | 4865.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 10:15:00 | 4955.90 | 4886.62 | 4874.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-26 11:15:00 | 4924.80 | 4931.90 | 4910.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-26 12:00:00 | 4924.80 | 4931.90 | 4910.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 14:15:00 | 4936.90 | 4936.92 | 4918.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 14:45:00 | 4927.50 | 4936.92 | 4918.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 09:15:00 | 4876.10 | 4924.61 | 4915.94 | EMA400 retest candle locked (from upside) |

### Cycle 58 — SELL (started 2026-02-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 10:15:00 | 4817.50 | 4903.19 | 4906.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 09:15:00 | 4640.80 | 4808.46 | 4854.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 14:15:00 | 4471.20 | 4399.30 | 4476.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 14:15:00 | 4471.20 | 4399.30 | 4476.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 4471.20 | 4399.30 | 4476.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 4540.30 | 4399.30 | 4476.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 4495.00 | 4418.44 | 4478.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 09:15:00 | 4402.60 | 4418.44 | 4478.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 4182.47 | 4355.51 | 4417.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 09:15:00 | 4360.00 | 4278.77 | 4333.47 | SL hit (close>ema200) qty=0.50 sl=4278.77 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 4383.50 | 4357.22 | 4356.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 4423.90 | 4370.55 | 4362.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 12:15:00 | 4352.30 | 4376.41 | 4367.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 12:15:00 | 4352.30 | 4376.41 | 4367.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 4352.30 | 4376.41 | 4367.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 4352.30 | 4376.41 | 4367.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 4351.90 | 4371.51 | 4366.39 | EMA400 retest candle locked (from upside) |

### Cycle 60 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 4350.20 | 4362.80 | 4363.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 4256.10 | 4341.46 | 4353.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 09:15:00 | 4207.50 | 4186.71 | 4232.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 09:15:00 | 4207.50 | 4186.71 | 4232.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 09:15:00 | 4207.50 | 4186.71 | 4232.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 09:45:00 | 4234.00 | 4186.71 | 4232.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 11:15:00 | 4208.60 | 4195.69 | 4229.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 12:15:00 | 4193.30 | 4195.69 | 4229.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 09:15:00 | 4289.40 | 4221.01 | 4228.11 | SL hit (close>static) qty=1.00 sl=4246.80 alert=retest2 |

### Cycle 61 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 4256.70 | 4234.24 | 4233.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 14:15:00 | 4292.30 | 4249.66 | 4240.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 4223.50 | 4312.58 | 4290.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 4223.50 | 4312.58 | 4290.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 4223.50 | 4312.58 | 4290.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 4223.50 | 4312.58 | 4290.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 4206.00 | 4291.27 | 4283.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 11:00:00 | 4206.00 | 4291.27 | 4283.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — SELL (started 2026-03-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 12:15:00 | 4202.60 | 4263.35 | 4271.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 4184.50 | 4247.58 | 4263.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 4040.00 | 4000.71 | 4076.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 4040.00 | 4000.71 | 4076.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 4040.00 | 4000.71 | 4076.35 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 4157.00 | 4100.04 | 4099.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 4240.70 | 4128.18 | 4112.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 4168.70 | 4231.22 | 4187.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 4168.70 | 4231.22 | 4187.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 4168.70 | 4231.22 | 4187.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 4168.70 | 4231.22 | 4187.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 4148.60 | 4214.69 | 4183.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 11:00:00 | 4148.60 | 4214.69 | 4183.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — SELL (started 2026-03-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 14:15:00 | 4097.40 | 4159.52 | 4164.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 4055.00 | 4130.98 | 4149.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 4276.00 | 4085.51 | 4108.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 4276.00 | 4085.51 | 4108.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 4276.00 | 4085.51 | 4108.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 4276.00 | 4085.51 | 4108.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 4240.00 | 4116.41 | 4120.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:30:00 | 4312.30 | 4116.41 | 4120.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 4210.40 | 4135.21 | 4128.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 4262.30 | 4160.63 | 4140.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 4018.80 | 4144.40 | 4141.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 4018.80 | 4144.40 | 4141.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 4018.80 | 4144.40 | 4141.47 | EMA400 retest candle locked (from upside) |

### Cycle 66 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 4015.30 | 4118.58 | 4130.00 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 15:15:00 | 4170.00 | 4134.66 | 4131.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 13:15:00 | 4251.10 | 4168.84 | 4149.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 10:15:00 | 4219.30 | 4220.81 | 4185.01 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-07 11:00:00 | 4219.30 | 4220.81 | 4185.01 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 4451.50 | 4504.55 | 4451.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 14:30:00 | 4445.80 | 4504.55 | 4451.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 4448.00 | 4493.24 | 4451.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:15:00 | 4486.00 | 4493.24 | 4451.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 4395.20 | 4503.81 | 4483.52 | SL hit (close<static) qty=1.00 sl=4443.30 alert=retest2 |

### Cycle 68 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 4439.10 | 4467.40 | 4470.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 14:15:00 | 4431.30 | 4460.18 | 4466.75 | Break + close below crossover candle low |

### Cycle 69 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 4643.00 | 4492.54 | 4480.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 10:15:00 | 4674.70 | 4622.59 | 4568.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 13:15:00 | 4623.30 | 4627.48 | 4584.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-17 09:15:00 | 4612.60 | 4620.69 | 4591.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 4612.60 | 4620.69 | 4591.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-17 10:00:00 | 4612.60 | 4620.69 | 4591.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 4626.20 | 4630.78 | 4612.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:45:00 | 4605.60 | 4630.78 | 4612.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 4641.60 | 4678.28 | 4664.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-22 10:00:00 | 4641.60 | 4678.28 | 4664.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 10:15:00 | 4663.00 | 4675.23 | 4663.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 11:45:00 | 4678.10 | 4675.58 | 4665.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 09:15:00 | 4578.40 | 4648.07 | 4655.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 09:15:00 | 4578.40 | 4648.07 | 4655.69 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 11:15:00 | 4458.90 | 4527.80 | 4549.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 14:15:00 | 4306.80 | 4293.11 | 4363.07 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-30 15:00:00 | 4306.80 | 4293.11 | 4363.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 4338.30 | 4303.25 | 4355.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 4288.70 | 4303.96 | 4347.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 11:15:00 | 4376.20 | 4309.44 | 4303.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 11:15:00 | 4376.20 | 4309.44 | 4303.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 4378.50 | 4328.96 | 4313.62 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 09:15:00 | 5479.50 | 2025-05-20 12:15:00 | 5505.00 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2025-06-02 09:45:00 | 5302.00 | 2025-06-02 15:15:00 | 5346.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-06-04 10:45:00 | 5419.00 | 2025-06-12 10:15:00 | 5502.50 | STOP_HIT | 1.00 | 1.54% |
| BUY | retest2 | 2025-06-05 09:15:00 | 5460.00 | 2025-06-12 10:15:00 | 5502.50 | STOP_HIT | 1.00 | 0.78% |
| SELL | retest2 | 2025-06-17 09:15:00 | 5370.50 | 2025-06-20 12:15:00 | 5359.50 | STOP_HIT | 1.00 | 0.20% |
| BUY | retest2 | 2025-06-23 10:30:00 | 5363.00 | 2025-06-30 09:15:00 | 5899.30 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-06-23 11:00:00 | 5360.50 | 2025-06-30 09:15:00 | 5896.55 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-07-14 09:15:00 | 5940.00 | 2025-07-17 09:15:00 | 5806.00 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-07-14 11:00:00 | 5944.00 | 2025-07-17 09:15:00 | 5806.00 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-07-14 14:30:00 | 5942.00 | 2025-07-17 09:15:00 | 5806.00 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-07-15 09:30:00 | 5945.50 | 2025-07-17 09:15:00 | 5806.00 | STOP_HIT | 1.00 | -2.35% |
| BUY | retest2 | 2025-07-16 10:15:00 | 5940.50 | 2025-07-17 09:15:00 | 5806.00 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2025-07-16 13:30:00 | 5932.00 | 2025-07-17 09:15:00 | 5806.00 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2025-07-16 14:15:00 | 5943.50 | 2025-07-17 09:15:00 | 5806.00 | STOP_HIT | 1.00 | -2.31% |
| SELL | retest2 | 2025-07-28 11:45:00 | 5809.50 | 2025-07-31 12:15:00 | 5872.50 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-07-31 12:00:00 | 5820.00 | 2025-07-31 12:15:00 | 5872.50 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-08-07 15:15:00 | 5884.00 | 2025-08-08 09:15:00 | 5825.50 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-08-20 09:30:00 | 6097.00 | 2025-08-22 15:15:00 | 6080.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-08-21 09:30:00 | 6086.00 | 2025-08-22 15:15:00 | 6080.00 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-09-11 11:30:00 | 5605.00 | 2025-09-11 13:15:00 | 5666.50 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-09-17 09:30:00 | 5750.50 | 2025-09-18 14:15:00 | 5716.00 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-09-17 11:00:00 | 5748.50 | 2025-09-18 14:15:00 | 5716.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-09-17 11:30:00 | 5746.00 | 2025-09-18 14:15:00 | 5716.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-09-17 12:30:00 | 5746.50 | 2025-09-18 14:15:00 | 5716.00 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-09-26 09:15:00 | 5654.00 | 2025-09-29 14:15:00 | 5756.50 | STOP_HIT | 1.00 | -1.81% |
| SELL | retest2 | 2025-09-26 10:00:00 | 5637.50 | 2025-09-29 14:15:00 | 5756.50 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-09-29 15:15:00 | 5572.50 | 2025-10-03 09:15:00 | 5665.00 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-10-07 15:15:00 | 5665.50 | 2025-10-08 14:15:00 | 5633.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-10-08 10:15:00 | 5671.50 | 2025-10-08 14:15:00 | 5633.00 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-10-14 13:30:00 | 5777.00 | 2025-10-23 12:15:00 | 5841.00 | STOP_HIT | 1.00 | 1.11% |
| BUY | retest2 | 2025-10-15 09:15:00 | 5797.50 | 2025-10-23 12:15:00 | 5841.00 | STOP_HIT | 1.00 | 0.75% |
| SELL | retest2 | 2025-10-29 09:15:00 | 5791.00 | 2025-10-29 15:15:00 | 5822.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-10-29 14:30:00 | 5804.00 | 2025-10-29 15:15:00 | 5822.00 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-11-04 09:15:00 | 5679.50 | 2025-11-06 10:15:00 | 5736.50 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-11-17 13:15:00 | 5875.00 | 2025-11-18 09:15:00 | 5796.00 | STOP_HIT | 1.00 | -1.34% |
| BUY | retest2 | 2025-11-17 14:00:00 | 5876.00 | 2025-11-18 09:15:00 | 5796.00 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2025-11-17 15:15:00 | 5878.00 | 2025-11-18 09:15:00 | 5796.00 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2025-12-12 11:45:00 | 4842.00 | 2025-12-15 09:15:00 | 4941.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-12-29 11:15:00 | 5083.00 | 2026-01-01 09:15:00 | 5112.50 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-12-29 14:45:00 | 5083.50 | 2026-01-01 09:15:00 | 5112.50 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-12-30 09:15:00 | 5066.50 | 2026-01-01 09:15:00 | 5112.50 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-01-28 13:15:00 | 4729.50 | 2026-02-02 14:15:00 | 4684.70 | STOP_HIT | 1.00 | 0.95% |
| SELL | retest2 | 2026-01-29 09:15:00 | 4676.50 | 2026-02-02 14:15:00 | 4684.70 | STOP_HIT | 1.00 | -0.18% |
| BUY | retest2 | 2026-02-05 13:30:00 | 4883.80 | 2026-02-13 10:15:00 | 4916.70 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2026-02-06 10:15:00 | 4883.90 | 2026-02-13 10:15:00 | 4916.70 | STOP_HIT | 1.00 | 0.67% |
| BUY | retest2 | 2026-02-06 11:00:00 | 4887.30 | 2026-02-13 10:15:00 | 4916.70 | STOP_HIT | 1.00 | 0.60% |
| BUY | retest2 | 2026-02-06 11:30:00 | 4902.50 | 2026-02-13 10:15:00 | 4916.70 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2026-02-09 10:15:00 | 4963.60 | 2026-02-13 10:15:00 | 4916.70 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-02-09 12:00:00 | 4960.00 | 2026-02-13 10:15:00 | 4916.70 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-02-09 13:15:00 | 4964.00 | 2026-02-13 10:15:00 | 4916.70 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2026-02-09 14:15:00 | 4963.00 | 2026-02-13 10:15:00 | 4916.70 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2026-03-06 09:15:00 | 4402.60 | 2026-03-09 09:15:00 | 4182.47 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 09:15:00 | 4402.60 | 2026-03-10 09:15:00 | 4360.00 | STOP_HIT | 0.50 | 0.97% |
| SELL | retest2 | 2026-03-16 12:15:00 | 4193.30 | 2026-03-17 09:15:00 | 4289.40 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2026-04-10 09:15:00 | 4486.00 | 2026-04-13 09:15:00 | 4395.20 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2026-04-13 12:15:00 | 4459.20 | 2026-04-13 13:15:00 | 4439.10 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2026-04-22 11:45:00 | 4678.10 | 2026-04-23 09:15:00 | 4578.40 | STOP_HIT | 1.00 | -2.13% |
| SELL | retest2 | 2026-05-04 12:00:00 | 4288.70 | 2026-05-06 11:15:00 | 4376.20 | STOP_HIT | 1.00 | -2.04% |
