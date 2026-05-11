# Alkem Laboratories Ltd. (ALKEM)

## Backtest Summary

- **Window:** 2025-12-22 09:15:00 → 2026-05-08 15:15:00 (644 bars)
- **Last close:** 5560.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 21 |
| ALERT1 | 16 |
| ALERT2 | 15 |
| ALERT2_SKIP | 7 |
| ALERT3 | 41 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 16 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 21 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 22 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 7 / 15
- **Target hits / Stop hits / Partials:** 0 / 21 / 1
- **Avg / median % per leg:** 0.00% / -0.90%
- **Sum % (uncompounded):** 0.08%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 9 | 5 | 55.6% | 0 | 9 | 0 | 1.03% | 9.2% |
| BUY @ 2nd Alert (retest1) | 4 | 0 | 0.0% | 0 | 4 | 0 | -1.00% | -4.0% |
| BUY @ 3rd Alert (retest2) | 5 | 5 | 100.0% | 0 | 5 | 0 | 2.64% | 13.2% |
| SELL (all) | 13 | 2 | 15.4% | 0 | 12 | 1 | -0.70% | -9.2% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.87% | -0.9% |
| SELL @ 3rd Alert (retest2) | 12 | 2 | 16.7% | 0 | 11 | 1 | -0.69% | -8.3% |
| retest1 (combined) | 5 | 0 | 0.0% | 0 | 5 | 0 | -0.97% | -4.9% |
| retest2 (combined) | 17 | 7 | 41.2% | 0 | 16 | 1 | 0.29% | 4.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2026-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 11:15:00 | 5557.50 | 5488.80 | 5483.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 5597.50 | 5525.20 | 5502.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 13:15:00 | 5546.50 | 5567.98 | 5538.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-05 13:15:00 | 5546.50 | 5567.98 | 5538.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 13:15:00 | 5546.50 | 5567.98 | 5538.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:30:00 | 5604.00 | 5571.29 | 5547.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 11:30:00 | 5610.50 | 5576.99 | 5554.35 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 12:15:00 | 5602.00 | 5576.99 | 5554.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 09:15:00 | 5770.50 | 5823.62 | 5824.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2026-01-19 09:15:00)

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

### Cycle 3 — BUY (started 2026-01-22 15:15:00)

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

### Cycle 4 — SELL (started 2026-01-28 11:15:00)

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

### Cycle 5 — BUY (started 2026-02-03 11:15:00)

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

### Cycle 6 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 5580.50 | 5640.70 | 5643.60 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 5674.00 | 5645.03 | 5644.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 14:15:00 | 5693.00 | 5654.62 | 5648.56 | Break + close above crossover candle high |

### Cycle 8 — SELL (started 2026-02-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 15:15:00 | 5587.50 | 5641.20 | 5643.01 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2026-02-09 09:15:00)

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

### Cycle 10 — SELL (started 2026-02-13 11:15:00)

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

### Cycle 11 — BUY (started 2026-02-23 13:15:00)

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

### Cycle 12 — SELL (started 2026-03-02 10:15:00)

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

### Cycle 13 — BUY (started 2026-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 13:15:00 | 5555.50 | 5525.67 | 5524.97 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 5461.00 | 5513.51 | 5519.72 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 5620.00 | 5520.94 | 5512.71 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 5467.50 | 5547.94 | 5550.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 13:15:00 | 5407.00 | 5486.23 | 5518.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 10:15:00 | 5322.00 | 5315.30 | 5361.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-18 09:15:00 | 5385.00 | 5340.27 | 5354.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 5385.00 | 5340.27 | 5354.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 5253.50 | 5358.86 | 5359.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 14:15:00 | 5294.50 | 5230.04 | 5228.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2026-03-24 14:15:00)

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

### Cycle 18 — SELL (started 2026-03-30 10:15:00)

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

### Cycle 19 — BUY (started 2026-04-08 11:15:00)

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

### Cycle 20 — SELL (started 2026-04-23 09:15:00)

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

### Cycle 21 — BUY (started 2026-05-05 14:15:00)

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
| BUY | retest2 | 2026-01-06 09:30:00 | 5604.00 | 2026-01-19 09:15:00 | 5770.50 | STOP_HIT | 1.00 | 2.97% |
| BUY | retest2 | 2026-01-06 11:30:00 | 5610.50 | 2026-01-19 09:15:00 | 5770.50 | STOP_HIT | 1.00 | 2.85% |
| BUY | retest2 | 2026-01-06 12:15:00 | 5602.00 | 2026-01-19 09:15:00 | 5770.50 | STOP_HIT | 1.00 | 3.01% |
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
