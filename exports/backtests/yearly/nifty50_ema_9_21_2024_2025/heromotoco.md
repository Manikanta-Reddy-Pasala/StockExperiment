# HEROMOTOCO (HEROMOTOCO)

## Backtest Summary

- **Window:** 2024-03-12 09:15:00 → 2026-05-08 15:15:00 (3717 bars)
- **Last close:** 5325.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 142 |
| ALERT1 | 104 |
| ALERT2 | 103 |
| ALERT2_SKIP | 90 |
| ALERT3 | 157 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 34 |
| PARTIAL | 8 |
| TARGET_HIT | 0 |
| STOP_HIT | 34 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 42 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 21 / 21
- **Target hits / Stop hits / Partials:** 0 / 34 / 8
- **Avg / median % per leg:** 1.81% / 1.88%
- **Sum % (uncompounded):** 76.15%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 21 | 4 | 19.0% | 0 | 21 | 0 | -0.31% | -6.6% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 21 | 4 | 19.0% | 0 | 21 | 0 | -0.31% | -6.6% |
| SELL (all) | 21 | 17 | 81.0% | 0 | 13 | 8 | 3.94% | 82.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 21 | 17 | 81.0% | 0 | 13 | 8 | 3.94% | 82.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 42 | 21 | 50.0% | 0 | 34 | 8 | 1.81% | 76.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 15:15:00 | 5045.00 | 5071.04 | 5073.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-22 09:15:00 | 4990.10 | 5054.86 | 5066.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-23 11:15:00 | 5025.05 | 5005.81 | 5025.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 11:15:00 | 5025.05 | 5005.81 | 5025.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 11:15:00 | 5025.05 | 5005.81 | 5025.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 12:00:00 | 5025.05 | 5005.81 | 5025.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 12:15:00 | 5033.05 | 5011.26 | 5026.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 12:45:00 | 5051.05 | 5011.26 | 5026.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 13:15:00 | 5095.20 | 5028.05 | 5032.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-23 14:00:00 | 5095.20 | 5028.05 | 5032.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 14:15:00 | 5151.90 | 5052.82 | 5043.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-28 10:15:00 | 5168.50 | 5108.69 | 5095.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-29 09:15:00 | 5146.35 | 5155.09 | 5129.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-29 09:15:00 | 5146.35 | 5155.09 | 5129.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 09:15:00 | 5146.35 | 5155.09 | 5129.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 11:45:00 | 5181.40 | 5160.68 | 5136.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 09:15:00 | 5176.15 | 5151.82 | 5139.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 09:45:00 | 5177.65 | 5154.25 | 5141.97 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-30 10:30:00 | 5172.10 | 5154.40 | 5143.15 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-30 11:15:00 | 5110.70 | 5145.66 | 5140.20 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-05-30 11:15:00 | 5110.70 | 5145.66 | 5140.20 | SL hit (close<static) qty=1.00 sl=5119.05 alert=retest2 |

### Cycle 3 — SELL (started 2024-05-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-31 10:15:00 | 5097.85 | 5131.46 | 5135.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-31 11:15:00 | 5019.40 | 5109.05 | 5124.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-31 14:15:00 | 5123.90 | 5100.41 | 5115.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 14:15:00 | 5123.90 | 5100.41 | 5115.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 14:15:00 | 5123.90 | 5100.41 | 5115.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 15:00:00 | 5123.90 | 5100.41 | 5115.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 15:15:00 | 5125.20 | 5105.37 | 5116.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 09:15:00 | 5248.10 | 5105.37 | 5116.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 09:15:00 | 5241.15 | 5132.53 | 5128.05 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 5035.25 | 5134.59 | 5140.01 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 12:15:00 | 5250.00 | 5157.67 | 5150.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-04 13:15:00 | 5304.40 | 5187.02 | 5164.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 09:15:00 | 5562.55 | 5564.09 | 5430.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-06 09:30:00 | 5512.05 | 5564.09 | 5430.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 10:15:00 | 5743.00 | 5784.41 | 5756.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-13 11:00:00 | 5743.00 | 5784.41 | 5756.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-13 11:15:00 | 5795.00 | 5786.53 | 5760.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 12:15:00 | 5806.00 | 5786.53 | 5760.00 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-13 13:00:00 | 5819.10 | 5793.04 | 5765.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-14 10:15:00 | 5804.80 | 5797.99 | 5777.14 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-18 09:15:00 | 5846.00 | 5809.92 | 5794.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 11:15:00 | 5773.35 | 5818.04 | 5803.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-18 12:00:00 | 5773.35 | 5818.04 | 5803.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 12:15:00 | 5794.20 | 5813.27 | 5802.62 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-18 14:15:00 | 5749.85 | 5790.92 | 5793.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-18 14:15:00 | 5749.85 | 5790.92 | 5793.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-19 09:15:00 | 5699.80 | 5769.03 | 5783.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-21 10:15:00 | 5595.30 | 5574.03 | 5631.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-21 10:30:00 | 5582.35 | 5574.03 | 5631.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 5568.90 | 5518.43 | 5542.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-25 09:30:00 | 5571.00 | 5518.43 | 5542.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 10:15:00 | 5544.50 | 5523.64 | 5542.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 13:30:00 | 5531.30 | 5530.71 | 5541.94 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 10:15:00 | 5549.15 | 5499.49 | 5495.98 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2024-06-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 10:15:00 | 5549.15 | 5499.49 | 5495.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-28 15:15:00 | 5590.00 | 5540.80 | 5519.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-02 11:15:00 | 5558.65 | 5601.32 | 5578.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-02 11:15:00 | 5558.65 | 5601.32 | 5578.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 11:15:00 | 5558.65 | 5601.32 | 5578.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:00:00 | 5558.65 | 5601.32 | 5578.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 5537.80 | 5588.62 | 5575.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 13:00:00 | 5537.80 | 5588.62 | 5575.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 14:15:00 | 5572.00 | 5585.83 | 5576.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 14:30:00 | 5563.20 | 5585.83 | 5576.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 15:15:00 | 5557.15 | 5580.10 | 5574.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 09:15:00 | 5585.05 | 5580.10 | 5574.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 11:45:00 | 5577.55 | 5577.73 | 5574.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-03 15:00:00 | 5579.95 | 5575.60 | 5574.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-04 09:15:00 | 5551.65 | 5571.68 | 5572.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2024-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 09:15:00 | 5551.65 | 5571.68 | 5572.84 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-08 09:15:00 | 5598.00 | 5569.03 | 5568.13 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2024-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 11:15:00 | 5530.00 | 5560.30 | 5564.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 14:15:00 | 5494.00 | 5539.44 | 5553.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 09:15:00 | 5541.45 | 5536.41 | 5549.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-09 10:00:00 | 5541.45 | 5536.41 | 5549.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 5536.55 | 5536.44 | 5548.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:30:00 | 5569.10 | 5536.44 | 5548.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 11:15:00 | 5542.50 | 5537.65 | 5547.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 12:00:00 | 5542.50 | 5537.65 | 5547.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 12:15:00 | 5588.50 | 5547.82 | 5551.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 13:00:00 | 5588.50 | 5547.82 | 5551.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 5574.35 | 5553.13 | 5553.34 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2024-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 14:15:00 | 5584.75 | 5559.45 | 5556.20 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-07-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 10:15:00 | 5496.00 | 5548.23 | 5552.22 | EMA200 below EMA400 |

### Cycle 14 — BUY (started 2024-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 09:15:00 | 5573.40 | 5534.64 | 5534.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-15 11:15:00 | 5620.05 | 5564.47 | 5549.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-16 09:15:00 | 5584.40 | 5586.40 | 5568.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 5584.40 | 5586.40 | 5568.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 5584.40 | 5586.40 | 5568.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 10:00:00 | 5584.40 | 5586.40 | 5568.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 12:15:00 | 5600.85 | 5593.85 | 5577.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-16 12:45:00 | 5580.60 | 5593.85 | 5577.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 15:15:00 | 5567.95 | 5590.01 | 5579.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 09:15:00 | 5500.80 | 5590.01 | 5579.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 09:15:00 | 5535.55 | 5579.12 | 5575.75 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2024-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 10:15:00 | 5453.70 | 5554.04 | 5564.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 11:15:00 | 5442.50 | 5531.73 | 5553.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-22 09:15:00 | 5474.30 | 5442.78 | 5475.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-22 09:15:00 | 5474.30 | 5442.78 | 5475.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 5474.30 | 5442.78 | 5475.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:00:00 | 5474.30 | 5442.78 | 5475.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 5461.00 | 5446.42 | 5474.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:30:00 | 5484.20 | 5446.42 | 5474.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 5459.95 | 5449.13 | 5473.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 11:45:00 | 5472.80 | 5449.13 | 5473.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 5479.00 | 5455.10 | 5473.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-22 13:00:00 | 5479.00 | 5455.10 | 5473.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 13:15:00 | 5449.95 | 5454.07 | 5471.49 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-23 10:15:00 | 5439.45 | 5457.19 | 5468.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-23 10:15:00 | 5509.45 | 5467.64 | 5472.59 | SL hit (close>static) qty=1.00 sl=5485.00 alert=retest2 |

### Cycle 16 — BUY (started 2024-07-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 11:15:00 | 5569.70 | 5488.05 | 5481.42 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-07-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 11:15:00 | 5474.35 | 5486.95 | 5487.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-24 13:15:00 | 5415.10 | 5467.11 | 5477.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 09:15:00 | 5469.40 | 5420.75 | 5437.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 09:15:00 | 5469.40 | 5420.75 | 5437.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 09:15:00 | 5469.40 | 5420.75 | 5437.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 10:00:00 | 5469.40 | 5420.75 | 5437.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 10:15:00 | 5509.65 | 5438.53 | 5443.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 11:00:00 | 5509.65 | 5438.53 | 5443.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2024-07-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 11:15:00 | 5510.95 | 5453.01 | 5449.97 | EMA200 above EMA400 |

### Cycle 19 — SELL (started 2024-07-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 11:15:00 | 5418.35 | 5455.32 | 5456.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-29 12:15:00 | 5407.00 | 5445.66 | 5452.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-30 09:15:00 | 5446.50 | 5436.07 | 5444.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 09:15:00 | 5446.50 | 5436.07 | 5444.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 5446.50 | 5436.07 | 5444.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 09:30:00 | 5456.95 | 5436.07 | 5444.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 10:15:00 | 5466.80 | 5442.22 | 5446.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 11:00:00 | 5466.80 | 5442.22 | 5446.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — BUY (started 2024-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 11:15:00 | 5506.30 | 5455.03 | 5451.87 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2024-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 10:15:00 | 5413.65 | 5450.97 | 5452.76 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2024-07-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 12:15:00 | 5499.95 | 5460.61 | 5456.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 15:15:00 | 5521.80 | 5479.45 | 5467.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 09:15:00 | 5410.30 | 5465.62 | 5461.88 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 09:15:00 | 5410.30 | 5465.62 | 5461.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 09:15:00 | 5410.30 | 5465.62 | 5461.88 | EMA400 retest candle locked (from upside) |

### Cycle 23 — SELL (started 2024-08-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 10:15:00 | 5417.00 | 5455.90 | 5457.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-01 13:15:00 | 5355.95 | 5422.37 | 5440.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 11:15:00 | 5217.75 | 5209.01 | 5257.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 12:00:00 | 5217.75 | 5209.01 | 5257.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 09:15:00 | 5275.05 | 5207.87 | 5236.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 09:30:00 | 5277.00 | 5207.87 | 5236.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 10:15:00 | 5264.95 | 5219.28 | 5239.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-07 10:30:00 | 5280.20 | 5219.28 | 5239.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 5244.30 | 5237.59 | 5243.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-07 15:15:00 | 5229.00 | 5237.59 | 5243.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 10:30:00 | 5215.40 | 5225.16 | 5235.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 11:15:00 | 5292.20 | 5221.43 | 5216.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — BUY (started 2024-08-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-12 11:15:00 | 5292.20 | 5221.43 | 5216.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-12 12:15:00 | 5308.20 | 5238.78 | 5225.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-13 11:15:00 | 5300.00 | 5306.29 | 5270.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-13 12:00:00 | 5300.00 | 5306.29 | 5270.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 12:15:00 | 5271.90 | 5299.41 | 5270.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 12:30:00 | 5267.20 | 5299.41 | 5270.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 5249.05 | 5289.34 | 5268.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 5249.05 | 5289.34 | 5268.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 5249.35 | 5281.34 | 5267.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-13 15:15:00 | 5229.55 | 5281.34 | 5267.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2024-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 09:15:00 | 5017.70 | 5220.33 | 5241.37 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2024-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 14:15:00 | 5191.00 | 5150.83 | 5149.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 5297.00 | 5186.33 | 5166.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-26 10:15:00 | 5379.05 | 5383.56 | 5350.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-26 11:00:00 | 5379.05 | 5383.56 | 5350.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 5356.95 | 5378.33 | 5353.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:00:00 | 5356.95 | 5378.33 | 5353.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 5356.80 | 5374.03 | 5353.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:45:00 | 5345.00 | 5374.03 | 5353.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 5345.00 | 5368.22 | 5353.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 15:00:00 | 5345.00 | 5368.22 | 5353.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 15:15:00 | 5344.85 | 5363.55 | 5352.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:15:00 | 5340.75 | 5363.55 | 5352.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 09:15:00 | 5361.45 | 5363.13 | 5353.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 09:30:00 | 5337.50 | 5363.13 | 5353.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 11:15:00 | 5359.45 | 5365.80 | 5356.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 11:45:00 | 5353.55 | 5365.80 | 5356.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 14:15:00 | 5363.95 | 5364.18 | 5357.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-27 14:45:00 | 5350.00 | 5364.18 | 5357.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-27 15:15:00 | 5335.00 | 5358.35 | 5355.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-28 09:15:00 | 5318.90 | 5358.35 | 5355.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — SELL (started 2024-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-28 09:15:00 | 5289.60 | 5344.60 | 5349.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 10:15:00 | 5270.30 | 5329.74 | 5342.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 13:15:00 | 5328.20 | 5317.13 | 5332.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 13:15:00 | 5328.20 | 5317.13 | 5332.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 13:15:00 | 5328.20 | 5317.13 | 5332.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-28 14:00:00 | 5328.20 | 5317.13 | 5332.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 14:15:00 | 5314.20 | 5316.54 | 5330.81 | EMA400 retest candle locked (from downside) |

### Cycle 28 — BUY (started 2024-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 14:15:00 | 5374.90 | 5335.12 | 5333.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-30 09:15:00 | 5481.05 | 5371.33 | 5350.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-04 10:15:00 | 5634.00 | 5634.24 | 5576.75 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-04 11:00:00 | 5634.00 | 5634.24 | 5576.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 10:15:00 | 5656.70 | 5703.53 | 5670.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-06 11:00:00 | 5656.70 | 5703.53 | 5670.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-06 11:15:00 | 5697.30 | 5702.28 | 5673.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-06 12:15:00 | 5718.15 | 5702.28 | 5673.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-09 09:30:00 | 5708.60 | 5729.13 | 5699.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-10 13:00:00 | 5711.00 | 5719.96 | 5713.92 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-10 14:15:00 | 5662.80 | 5708.05 | 5709.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2024-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-10 14:15:00 | 5662.80 | 5708.05 | 5709.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-11 13:15:00 | 5649.85 | 5675.98 | 5690.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-12 09:15:00 | 5702.80 | 5675.53 | 5686.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-12 09:15:00 | 5702.80 | 5675.53 | 5686.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-12 09:15:00 | 5702.80 | 5675.53 | 5686.11 | EMA400 retest candle locked (from downside) |

### Cycle 30 — BUY (started 2024-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 11:15:00 | 5779.50 | 5706.96 | 5699.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-12 13:15:00 | 5802.00 | 5737.61 | 5715.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-16 11:15:00 | 5788.40 | 5791.55 | 5770.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-16 12:00:00 | 5788.40 | 5791.55 | 5770.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 12:15:00 | 5777.05 | 5788.65 | 5771.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 13:45:00 | 5794.55 | 5789.89 | 5773.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 09:15:00 | 5803.30 | 5788.21 | 5775.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 09:45:00 | 5803.35 | 5791.22 | 5777.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-26 09:15:00 | 5932.00 | 6053.98 | 6070.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2024-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-26 09:15:00 | 5932.00 | 6053.98 | 6070.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 5779.90 | 5944.70 | 5992.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 13:15:00 | 5763.85 | 5745.40 | 5817.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-01 13:45:00 | 5784.15 | 5745.40 | 5817.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 5529.75 | 5498.08 | 5535.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 5529.75 | 5498.08 | 5535.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 5534.95 | 5505.45 | 5535.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 5542.35 | 5505.45 | 5535.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 5550.90 | 5514.54 | 5537.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:00:00 | 5550.90 | 5514.54 | 5537.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 10:15:00 | 5580.00 | 5527.64 | 5541.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 10:45:00 | 5575.65 | 5527.64 | 5541.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 11:15:00 | 5548.00 | 5531.71 | 5541.71 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2024-10-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 15:15:00 | 5554.00 | 5547.54 | 5546.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 5572.30 | 5552.49 | 5549.26 | Break + close above crossover candle high |

### Cycle 33 — SELL (started 2024-10-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 10:15:00 | 5517.05 | 5545.41 | 5546.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 11:15:00 | 5508.95 | 5538.11 | 5542.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 09:15:00 | 5561.70 | 5494.17 | 5501.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 09:15:00 | 5561.70 | 5494.17 | 5501.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 5561.70 | 5494.17 | 5501.86 | EMA400 retest candle locked (from downside) |

### Cycle 34 — BUY (started 2024-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-14 11:15:00 | 5552.20 | 5511.57 | 5508.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-14 13:15:00 | 5555.70 | 5525.72 | 5516.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-15 09:15:00 | 5526.50 | 5534.70 | 5523.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-15 10:00:00 | 5526.50 | 5534.70 | 5523.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 10:15:00 | 5529.45 | 5533.65 | 5523.88 | EMA400 retest candle locked (from upside) |

### Cycle 35 — SELL (started 2024-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 12:15:00 | 5468.00 | 5517.04 | 5517.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 09:15:00 | 5427.10 | 5494.76 | 5506.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-21 10:15:00 | 5210.50 | 5205.88 | 5260.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-21 10:45:00 | 5219.95 | 5205.88 | 5260.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 14:15:00 | 5239.10 | 5208.67 | 5243.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-21 14:30:00 | 5243.90 | 5208.67 | 5243.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 15:15:00 | 5240.00 | 5214.94 | 5243.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-22 09:15:00 | 5216.10 | 5214.94 | 5243.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-22 09:15:00 | 5200.10 | 5211.97 | 5239.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 10:15:00 | 5162.85 | 5211.97 | 5239.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 11:15:00 | 5176.35 | 5208.95 | 5235.57 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 13:15:00 | 5188.00 | 5200.70 | 5226.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-22 14:45:00 | 5179.70 | 5195.28 | 5219.98 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 09:15:00 | 5216.70 | 5197.28 | 5216.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 09:45:00 | 5218.35 | 5197.28 | 5216.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 10:15:00 | 5237.00 | 5205.22 | 5218.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 10:45:00 | 5236.65 | 5205.22 | 5218.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 11:15:00 | 5223.30 | 5208.84 | 5218.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:15:00 | 5264.00 | 5208.84 | 5218.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 12:15:00 | 5250.35 | 5217.14 | 5221.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 12:30:00 | 5262.95 | 5217.14 | 5221.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 13:15:00 | 5212.85 | 5216.28 | 5220.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-23 13:30:00 | 5245.60 | 5216.28 | 5220.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-23 14:15:00 | 5152.35 | 5203.50 | 5214.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-23 15:15:00 | 5133.95 | 5203.50 | 5214.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-24 12:15:00 | 5122.45 | 5176.66 | 5196.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 4904.71 | 5067.67 | 5130.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 4917.53 | 5067.67 | 5130.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 4928.60 | 5067.67 | 5130.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 4920.71 | 5067.67 | 5130.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-25 10:15:00 | 4877.25 | 5067.67 | 5130.46 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-29 09:15:00 | 4866.33 | 4922.50 | 4986.11 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-30 10:15:00 | 4850.10 | 4823.83 | 4886.22 | SL hit (close>ema200) qty=0.50 sl=4823.83 alert=retest2 |

### Cycle 36 — BUY (started 2024-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-31 10:15:00 | 5003.90 | 4913.58 | 4905.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-01 17:15:00 | 5028.00 | 4979.89 | 4947.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 4813.95 | 4953.44 | 4941.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 4813.95 | 4953.44 | 4941.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 4813.95 | 4953.44 | 4941.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 4813.95 | 4953.44 | 4941.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 4755.90 | 4913.93 | 4924.70 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2024-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 12:15:00 | 4892.90 | 4863.05 | 4859.38 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 10:15:00 | 4820.80 | 4854.55 | 4857.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 09:15:00 | 4780.40 | 4817.40 | 4835.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 14:15:00 | 4759.60 | 4759.20 | 4781.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-11 14:30:00 | 4760.75 | 4759.20 | 4781.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 09:15:00 | 4772.05 | 4760.30 | 4778.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 12:15:00 | 4748.50 | 4761.16 | 4775.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 14:15:00 | 4511.07 | 4602.86 | 4673.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 10:15:00 | 4582.90 | 4574.18 | 4640.50 | SL hit (close>ema200) qty=0.50 sl=4574.18 alert=retest2 |

### Cycle 40 — BUY (started 2024-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 10:15:00 | 4807.00 | 4669.96 | 4657.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 4848.75 | 4797.44 | 4779.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-26 09:15:00 | 4829.70 | 4847.69 | 4820.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-26 09:15:00 | 4829.70 | 4847.69 | 4820.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 4829.70 | 4847.69 | 4820.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-26 09:45:00 | 4820.55 | 4847.69 | 4820.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 10:15:00 | 4860.70 | 4850.29 | 4823.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 11:15:00 | 4867.65 | 4842.85 | 4831.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 12:45:00 | 4868.50 | 4851.64 | 4837.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 13:15:00 | 4806.70 | 4837.21 | 4840.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — SELL (started 2024-11-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 13:15:00 | 4806.70 | 4837.21 | 4840.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-28 15:15:00 | 4777.00 | 4816.35 | 4829.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 09:15:00 | 4766.05 | 4753.02 | 4769.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-03 09:15:00 | 4766.05 | 4753.02 | 4769.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-03 09:15:00 | 4766.05 | 4753.02 | 4769.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-03 11:30:00 | 4733.05 | 4748.78 | 4765.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 13:15:00 | 4644.20 | 4618.20 | 4615.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — BUY (started 2024-12-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 13:15:00 | 4644.20 | 4618.20 | 4615.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 14:15:00 | 4647.70 | 4624.10 | 4618.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 09:15:00 | 4589.30 | 4620.97 | 4618.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 09:15:00 | 4589.30 | 4620.97 | 4618.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 4589.30 | 4620.97 | 4618.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:00:00 | 4589.30 | 4620.97 | 4618.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2024-12-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 10:15:00 | 4569.85 | 4610.75 | 4614.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 11:15:00 | 4556.50 | 4599.90 | 4608.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 12:15:00 | 4560.40 | 4554.58 | 4575.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-13 13:00:00 | 4560.40 | 4554.58 | 4575.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 13:15:00 | 4577.05 | 4559.07 | 4575.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 14:00:00 | 4577.05 | 4559.07 | 4575.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 14:15:00 | 4572.90 | 4561.84 | 4575.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 09:15:00 | 4546.25 | 4564.44 | 4575.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-23 09:15:00 | 4318.94 | 4366.84 | 4398.00 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-26 10:15:00 | 4291.45 | 4276.74 | 4305.44 | SL hit (close>ema200) qty=0.50 sl=4276.74 alert=retest2 |

### Cycle 44 — BUY (started 2025-01-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 11:15:00 | 4244.05 | 4195.53 | 4192.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-02 12:15:00 | 4266.95 | 4209.81 | 4199.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 09:15:00 | 4225.95 | 4251.57 | 4226.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 09:15:00 | 4225.95 | 4251.57 | 4226.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 09:15:00 | 4225.95 | 4251.57 | 4226.59 | EMA400 retest candle locked (from upside) |

### Cycle 45 — SELL (started 2025-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 14:15:00 | 4192.80 | 4228.37 | 4228.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 12:15:00 | 4185.35 | 4206.19 | 4216.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 11:15:00 | 4140.80 | 4138.36 | 4160.00 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 13:15:00 | 4160.55 | 4144.64 | 4159.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 13:15:00 | 4160.55 | 4144.64 | 4159.22 | EMA400 retest candle locked (from downside) |

### Cycle 46 — BUY (started 2025-01-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 10:15:00 | 4119.00 | 4104.36 | 4104.08 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2025-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-16 10:15:00 | 4073.60 | 4101.60 | 4103.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-16 12:15:00 | 4063.65 | 4089.98 | 4097.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-17 09:15:00 | 4084.00 | 4083.19 | 4091.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-17 09:15:00 | 4084.00 | 4083.19 | 4091.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 09:15:00 | 4084.00 | 4083.19 | 4091.46 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2025-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-23 10:15:00 | 4110.35 | 4068.27 | 4066.19 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-24 14:15:00 | 4053.30 | 4077.78 | 4078.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 10:15:00 | 4019.05 | 4059.51 | 4069.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 10:15:00 | 4028.75 | 4027.35 | 4044.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-28 11:15:00 | 4040.35 | 4029.95 | 4044.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 11:15:00 | 4040.35 | 4029.95 | 4044.22 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 4092.00 | 4052.41 | 4048.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 4118.70 | 4080.95 | 4066.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 09:15:00 | 4281.00 | 4289.74 | 4219.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 10:15:00 | 4289.05 | 4352.79 | 4300.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 4289.05 | 4352.79 | 4300.06 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2025-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-04 10:15:00 | 4236.75 | 4282.81 | 4283.29 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 09:15:00 | 4303.90 | 4272.15 | 4269.22 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 13:15:00 | 4227.75 | 4264.56 | 4267.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-06 14:15:00 | 4221.00 | 4255.85 | 4263.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 10:15:00 | 4254.00 | 4251.27 | 4258.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-07 10:15:00 | 4254.00 | 4251.27 | 4258.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 4254.00 | 4251.27 | 4258.83 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-10 09:15:00 | 4284.35 | 4263.93 | 4261.82 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-02-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 11:15:00 | 4235.65 | 4259.08 | 4260.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 12:15:00 | 4190.50 | 4245.36 | 4253.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-17 12:15:00 | 3858.00 | 3856.12 | 3918.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-19 09:15:00 | 3883.05 | 3850.29 | 3872.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 3883.05 | 3850.29 | 3872.54 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2025-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 09:15:00 | 3913.60 | 3881.43 | 3879.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 10:15:00 | 3935.00 | 3892.14 | 3884.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 3844.00 | 3893.01 | 3890.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 3844.00 | 3893.01 | 3890.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 3844.00 | 3893.01 | 3890.55 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 3848.55 | 3884.12 | 3886.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 3807.75 | 3851.67 | 3867.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 11:15:00 | 3853.70 | 3851.43 | 3864.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 11:15:00 | 3853.70 | 3851.43 | 3864.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 3853.70 | 3851.43 | 3864.80 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2025-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 13:15:00 | 3649.05 | 3614.21 | 3611.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 15:15:00 | 3658.00 | 3628.50 | 3618.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 12:15:00 | 3652.55 | 3659.10 | 3639.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-10 09:15:00 | 3633.25 | 3652.78 | 3642.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 3633.25 | 3652.78 | 3642.76 | EMA400 retest candle locked (from upside) |

### Cycle 59 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 11:15:00 | 3586.50 | 3634.49 | 3635.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 3581.20 | 3615.24 | 3626.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 10:15:00 | 3621.95 | 3611.31 | 3621.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 10:15:00 | 3621.95 | 3611.31 | 3621.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 3621.95 | 3611.31 | 3621.01 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2025-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 15:15:00 | 3651.00 | 3625.22 | 3623.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 10:15:00 | 3655.75 | 3632.39 | 3627.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 11:15:00 | 3615.95 | 3629.10 | 3626.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 11:15:00 | 3615.95 | 3629.10 | 3626.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 3615.95 | 3629.10 | 3626.44 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2025-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 12:15:00 | 3605.10 | 3624.30 | 3624.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 10:15:00 | 3586.90 | 3610.40 | 3617.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 3528.00 | 3509.28 | 3539.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 11:15:00 | 3548.10 | 3519.10 | 3538.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 11:15:00 | 3548.10 | 3519.10 | 3538.77 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2025-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 09:15:00 | 3568.95 | 3544.60 | 3544.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 11:15:00 | 3615.00 | 3567.28 | 3555.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 13:15:00 | 3636.65 | 3640.16 | 3608.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-24 09:15:00 | 3608.65 | 3631.75 | 3612.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 3608.65 | 3631.75 | 3612.05 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 3635.35 | 3731.06 | 3743.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 3455.00 | 3637.54 | 3687.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 12:15:00 | 3566.00 | 3544.27 | 3587.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-08 14:15:00 | 3571.90 | 3554.73 | 3585.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 3571.90 | 3554.73 | 3585.15 | EMA400 retest candle locked (from downside) |

### Cycle 64 — BUY (started 2025-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 12:15:00 | 3638.15 | 3595.64 | 3594.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 3673.00 | 3620.52 | 3607.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 12:15:00 | 3752.00 | 3756.94 | 3719.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-17 09:15:00 | 3680.20 | 3751.24 | 3729.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 3680.20 | 3751.24 | 3729.85 | EMA400 retest candle locked (from upside) |

### Cycle 65 — SELL (started 2025-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-28 10:15:00 | 3881.00 | 3892.32 | 3893.05 | EMA200 below EMA400 |

### Cycle 66 — BUY (started 2025-04-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 12:15:00 | 3922.30 | 3898.90 | 3895.95 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 11:15:00 | 3866.00 | 3896.06 | 3897.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-30 09:15:00 | 3834.00 | 3865.46 | 3880.22 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-02 09:15:00 | 3896.10 | 3848.20 | 3859.94 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-02 09:15:00 | 3896.10 | 3848.20 | 3859.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 09:15:00 | 3896.10 | 3848.20 | 3859.94 | EMA400 retest candle locked (from downside) |

### Cycle 68 — BUY (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-06 11:15:00 | 3854.20 | 3812.97 | 3810.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-06 12:15:00 | 3887.00 | 3827.78 | 3817.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-07 09:15:00 | 3848.20 | 3849.15 | 3832.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 09:15:00 | 3848.20 | 3849.15 | 3832.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 09:15:00 | 3848.20 | 3849.15 | 3832.61 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 12:15:00 | 3810.80 | 3841.27 | 3842.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 3779.00 | 3828.81 | 3836.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 09:15:00 | 3843.00 | 3823.88 | 3831.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-09 09:15:00 | 3843.00 | 3823.88 | 3831.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 3843.00 | 3823.88 | 3831.91 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2025-05-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 11:15:00 | 3874.90 | 3837.78 | 3837.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 3920.80 | 3861.09 | 3848.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 4329.00 | 4368.07 | 4312.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 4329.00 | 4368.07 | 4312.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 4329.00 | 4368.07 | 4312.74 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 4249.00 | 4296.63 | 4297.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 4220.50 | 4272.50 | 4285.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 4265.00 | 4262.98 | 4277.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 12:15:00 | 4268.00 | 4255.08 | 4266.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 4268.00 | 4255.08 | 4266.74 | EMA400 retest candle locked (from downside) |

### Cycle 72 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 4319.20 | 4273.72 | 4272.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 4398.60 | 4316.39 | 4295.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 15:15:00 | 4358.00 | 4358.95 | 4331.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 4302.80 | 4347.72 | 4328.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 4302.80 | 4347.72 | 4328.65 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 14:15:00 | 4302.80 | 4343.83 | 4345.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 4216.50 | 4312.97 | 4331.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 13:15:00 | 4244.50 | 4240.73 | 4265.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 09:15:00 | 4229.30 | 4203.10 | 4217.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 4229.30 | 4203.10 | 4217.44 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 4280.80 | 4235.42 | 4230.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 4286.30 | 4260.34 | 4244.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 14:15:00 | 4413.50 | 4417.95 | 4378.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 09:15:00 | 4391.20 | 4412.93 | 4382.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 09:15:00 | 4391.20 | 4412.93 | 4382.89 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2025-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 09:15:00 | 4309.50 | 4366.09 | 4371.69 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 4386.80 | 4357.25 | 4356.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 09:15:00 | 4427.00 | 4382.98 | 4372.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 11:15:00 | 4381.10 | 4384.45 | 4375.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 12:15:00 | 4391.10 | 4385.78 | 4376.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 4391.10 | 4385.78 | 4376.82 | EMA400 retest candle locked (from upside) |

### Cycle 77 — SELL (started 2025-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 09:15:00 | 4282.40 | 4358.83 | 4367.70 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2025-06-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-27 10:15:00 | 4320.50 | 4295.27 | 4295.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-27 11:15:00 | 4344.00 | 4305.01 | 4299.48 | Break + close above crossover candle high |

### Cycle 79 — SELL (started 2025-06-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-30 09:15:00 | 4231.90 | 4299.41 | 4300.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 10:15:00 | 4202.30 | 4246.23 | 4268.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 09:15:00 | 4240.00 | 4230.11 | 4248.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 09:15:00 | 4240.00 | 4230.11 | 4248.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 4240.00 | 4230.11 | 4248.10 | EMA400 retest candle locked (from downside) |

### Cycle 80 — BUY (started 2025-07-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 09:15:00 | 4286.00 | 4251.96 | 4251.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 11:15:00 | 4307.20 | 4268.25 | 4259.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-07 10:15:00 | 4328.10 | 4334.92 | 4315.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-07 12:15:00 | 4304.00 | 4327.31 | 4315.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 12:15:00 | 4304.00 | 4327.31 | 4315.39 | EMA400 retest candle locked (from upside) |

### Cycle 81 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 4274.90 | 4307.69 | 4309.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 4248.00 | 4295.75 | 4303.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 11:15:00 | 4297.00 | 4296.00 | 4302.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 12:15:00 | 4305.40 | 4297.88 | 4303.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 4305.40 | 4297.88 | 4303.16 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 4320.00 | 4305.17 | 4304.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 11:15:00 | 4327.50 | 4309.64 | 4306.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 11:15:00 | 4324.60 | 4326.30 | 4318.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 11:15:00 | 4324.60 | 4326.30 | 4318.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 4324.60 | 4326.30 | 4318.69 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 4275.30 | 4312.32 | 4315.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 12:15:00 | 4232.30 | 4289.99 | 4304.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 14:15:00 | 4253.90 | 4242.43 | 4262.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 4326.00 | 4259.52 | 4266.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 4326.00 | 4259.52 | 4266.92 | EMA400 retest candle locked (from downside) |

### Cycle 84 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 4399.70 | 4287.56 | 4278.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 4402.20 | 4310.49 | 4290.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 4393.20 | 4393.57 | 4345.67 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 10:15:00 | 4413.10 | 4429.90 | 4409.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 4413.10 | 4429.90 | 4409.96 | EMA400 retest candle locked (from upside) |

### Cycle 85 — SELL (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 10:15:00 | 4383.50 | 4405.44 | 4406.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 11:15:00 | 4360.60 | 4396.47 | 4402.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 4374.20 | 4366.18 | 4382.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 15:15:00 | 4366.50 | 4358.18 | 4369.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 4366.50 | 4358.18 | 4369.74 | EMA400 retest candle locked (from downside) |

### Cycle 86 — BUY (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 10:15:00 | 4321.50 | 4290.25 | 4286.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 12:15:00 | 4330.00 | 4303.41 | 4293.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 10:15:00 | 4257.80 | 4304.42 | 4299.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 10:15:00 | 4257.80 | 4304.42 | 4299.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 4257.80 | 4304.42 | 4299.37 | EMA400 retest candle locked (from upside) |

### Cycle 87 — SELL (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 11:15:00 | 4227.50 | 4289.04 | 4292.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 4218.00 | 4253.70 | 4271.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 11:15:00 | 4264.80 | 4254.77 | 4269.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 11:15:00 | 4264.80 | 4254.77 | 4269.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 4264.80 | 4254.77 | 4269.05 | EMA400 retest candle locked (from downside) |

### Cycle 88 — BUY (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 11:15:00 | 4297.00 | 4273.18 | 4272.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 09:15:00 | 4426.40 | 4320.17 | 4296.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 4491.40 | 4522.10 | 4466.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 14:15:00 | 4475.00 | 4496.41 | 4473.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 4475.00 | 4496.41 | 4473.22 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2025-08-22 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 11:15:00 | 5015.30 | 5058.65 | 5060.74 | EMA200 below EMA400 |

### Cycle 90 — BUY (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 14:15:00 | 5058.10 | 5048.43 | 5047.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-26 09:15:00 | 5115.00 | 5064.72 | 5055.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 14:15:00 | 5060.70 | 5080.72 | 5069.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 14:15:00 | 5060.70 | 5080.72 | 5069.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 5060.70 | 5080.72 | 5069.08 | EMA400 retest candle locked (from upside) |

### Cycle 91 — SELL (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 11:15:00 | 5348.50 | 5402.02 | 5407.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 09:15:00 | 5306.00 | 5355.80 | 5380.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 5347.00 | 5324.19 | 5347.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 5347.00 | 5324.19 | 5347.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 5347.00 | 5324.19 | 5347.67 | EMA400 retest candle locked (from downside) |

### Cycle 92 — BUY (started 2025-09-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 13:15:00 | 5333.00 | 5322.68 | 5321.63 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 14:15:00 | 5313.00 | 5320.74 | 5320.84 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2025-09-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 15:15:00 | 5322.50 | 5321.09 | 5321.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 5353.00 | 5327.48 | 5323.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 14:15:00 | 5352.50 | 5352.57 | 5340.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 12:15:00 | 5351.00 | 5358.47 | 5348.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 5351.00 | 5358.47 | 5348.17 | EMA400 retest candle locked (from upside) |

### Cycle 95 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 5374.00 | 5404.25 | 5407.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 15:15:00 | 5362.50 | 5395.90 | 5402.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 10:15:00 | 5308.50 | 5307.58 | 5340.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 13:15:00 | 5338.00 | 5309.24 | 5333.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 5338.00 | 5309.24 | 5333.04 | EMA400 retest candle locked (from downside) |

### Cycle 96 — BUY (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-26 09:15:00 | 5422.50 | 5344.03 | 5344.01 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-09-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 10:15:00 | 5327.00 | 5347.51 | 5349.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-29 11:15:00 | 5282.00 | 5334.41 | 5343.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 13:15:00 | 5384.50 | 5341.80 | 5345.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 13:15:00 | 5384.50 | 5341.80 | 5345.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 13:15:00 | 5384.50 | 5341.80 | 5345.31 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 10:15:00 | 5400.00 | 5353.20 | 5348.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 11:15:00 | 5423.00 | 5367.16 | 5355.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 14:15:00 | 5427.50 | 5434.18 | 5410.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 09:15:00 | 5392.50 | 5426.38 | 5410.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 5392.50 | 5426.38 | 5410.92 | EMA400 retest candle locked (from upside) |

### Cycle 99 — SELL (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 15:15:00 | 5520.00 | 5554.52 | 5558.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 5487.50 | 5541.12 | 5552.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 12:15:00 | 5526.50 | 5515.81 | 5527.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 12:15:00 | 5526.50 | 5515.81 | 5527.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 5526.50 | 5515.81 | 5527.07 | EMA400 retest candle locked (from downside) |

### Cycle 100 — BUY (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 11:15:00 | 5552.00 | 5529.98 | 5529.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 12:15:00 | 5577.50 | 5539.48 | 5533.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 15:15:00 | 5560.00 | 5561.61 | 5552.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 13:15:00 | 5518.00 | 5563.95 | 5558.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 5518.00 | 5563.95 | 5558.48 | EMA400 retest candle locked (from upside) |

### Cycle 101 — SELL (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-15 15:15:00 | 5535.00 | 5553.61 | 5554.44 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 5583.00 | 5559.49 | 5557.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 10:15:00 | 5613.00 | 5584.05 | 5573.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 15:15:00 | 5627.50 | 5635.43 | 5614.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 09:15:00 | 5646.50 | 5639.12 | 5621.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 5646.50 | 5639.12 | 5621.57 | EMA400 retest candle locked (from upside) |

### Cycle 103 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 5584.00 | 5619.01 | 5619.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 10:15:00 | 5560.00 | 5604.48 | 5613.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 5596.50 | 5563.61 | 5583.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 5596.50 | 5563.61 | 5583.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 5596.50 | 5563.61 | 5583.10 | EMA400 retest candle locked (from downside) |

### Cycle 104 — BUY (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 12:15:00 | 5651.00 | 5601.49 | 5597.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 5655.00 | 5630.44 | 5614.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 11:15:00 | 5612.50 | 5627.66 | 5615.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 11:15:00 | 5612.50 | 5627.66 | 5615.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 5612.50 | 5627.66 | 5615.77 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 5566.50 | 5606.17 | 5608.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 11:15:00 | 5509.00 | 5548.97 | 5572.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 09:15:00 | 5616.00 | 5544.75 | 5559.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 5616.00 | 5544.75 | 5559.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 5616.00 | 5544.75 | 5559.47 | EMA400 retest candle locked (from downside) |

### Cycle 106 — BUY (started 2025-10-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 13:15:00 | 5572.00 | 5568.32 | 5568.14 | EMA200 above EMA400 |

### Cycle 107 — SELL (started 2025-10-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 14:15:00 | 5546.50 | 5563.95 | 5566.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-03 09:15:00 | 5534.00 | 5555.73 | 5561.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 5349.00 | 5342.02 | 5401.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-10 09:15:00 | 5374.00 | 5317.65 | 5343.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 5374.00 | 5317.65 | 5343.26 | EMA400 retest candle locked (from downside) |

### Cycle 108 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 5396.00 | 5360.95 | 5358.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 11:15:00 | 5411.00 | 5370.36 | 5363.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 5520.00 | 5523.96 | 5485.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 15:15:00 | 5498.00 | 5518.77 | 5486.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 5498.00 | 5518.77 | 5486.73 | EMA400 retest candle locked (from upside) |

### Cycle 109 — SELL (started 2025-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 13:15:00 | 6209.50 | 6231.65 | 6232.99 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 6315.50 | 6246.12 | 6238.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 09:15:00 | 6376.00 | 6323.76 | 6288.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 13:15:00 | 6339.00 | 6339.94 | 6308.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 09:15:00 | 6231.00 | 6322.91 | 6309.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 6231.00 | 6322.91 | 6309.23 | EMA400 retest candle locked (from upside) |

### Cycle 111 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 6199.50 | 6281.36 | 6291.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 6173.00 | 6245.71 | 6272.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 6004.00 | 5977.17 | 6043.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 5999.50 | 5979.29 | 6011.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 5999.50 | 5979.29 | 6011.44 | EMA400 retest candle locked (from downside) |

### Cycle 112 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 5721.50 | 5662.89 | 5655.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 10:15:00 | 5746.00 | 5694.76 | 5672.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 15:15:00 | 5990.00 | 5992.03 | 5956.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 13:15:00 | 5950.00 | 5986.68 | 5967.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 5950.00 | 5986.68 | 5967.93 | EMA400 retest candle locked (from upside) |

### Cycle 113 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 5901.50 | 5961.88 | 5962.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 5863.00 | 5942.10 | 5953.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 5749.00 | 5717.53 | 5752.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 14:15:00 | 5749.00 | 5717.53 | 5752.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 5749.00 | 5717.53 | 5752.39 | EMA400 retest candle locked (from downside) |

### Cycle 114 — BUY (started 2026-01-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 13:15:00 | 5744.50 | 5704.85 | 5703.19 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2026-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 13:15:00 | 5620.00 | 5699.71 | 5705.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 5583.50 | 5676.47 | 5694.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 5615.00 | 5578.25 | 5618.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 5615.00 | 5578.25 | 5618.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 5615.00 | 5578.25 | 5618.09 | EMA400 retest candle locked (from downside) |

### Cycle 116 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 5510.00 | 5425.90 | 5425.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 09:15:00 | 5550.50 | 5450.82 | 5436.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 11:15:00 | 5534.00 | 5539.21 | 5503.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 12:15:00 | 5481.50 | 5527.67 | 5501.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 5481.50 | 5527.67 | 5501.42 | EMA400 retest candle locked (from upside) |

### Cycle 117 — SELL (started 2026-02-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-02 10:15:00 | 5430.50 | 5510.11 | 5518.43 | EMA200 below EMA400 |

### Cycle 118 — BUY (started 2026-02-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 13:15:00 | 5566.50 | 5526.81 | 5524.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 14:15:00 | 5617.00 | 5544.85 | 5532.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 5801.00 | 5814.30 | 5743.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 14:15:00 | 5764.50 | 5791.60 | 5757.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 5764.50 | 5791.60 | 5757.91 | EMA400 retest candle locked (from upside) |

### Cycle 119 — SELL (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 09:15:00 | 5725.50 | 5760.97 | 5761.27 | EMA200 below EMA400 |

### Cycle 120 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 10:15:00 | 5785.50 | 5759.49 | 5757.17 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 5668.00 | 5745.55 | 5753.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 10:15:00 | 5527.50 | 5600.11 | 5650.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 10:15:00 | 5536.00 | 5525.07 | 5577.73 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 13:15:00 | 5554.00 | 5535.67 | 5569.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 5554.00 | 5535.67 | 5569.78 | EMA400 retest candle locked (from downside) |

### Cycle 122 — BUY (started 2026-02-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 15:15:00 | 5594.50 | 5575.04 | 5574.99 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2026-02-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-19 09:15:00 | 5550.00 | 5570.03 | 5572.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 10:15:00 | 5526.50 | 5561.33 | 5568.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-20 13:15:00 | 5436.50 | 5435.23 | 5480.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 09:15:00 | 5475.50 | 5451.06 | 5477.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 5475.50 | 5451.06 | 5477.09 | EMA400 retest candle locked (from downside) |

### Cycle 124 — BUY (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 10:15:00 | 5505.50 | 5483.05 | 5481.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 15:15:00 | 5520.00 | 5498.49 | 5490.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 5747.00 | 5749.79 | 5691.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 13:15:00 | 5743.00 | 5739.83 | 5704.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 5743.00 | 5739.83 | 5704.92 | EMA400 retest candle locked (from upside) |

### Cycle 125 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 5561.00 | 5669.99 | 5682.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 5530.50 | 5642.09 | 5668.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 5525.50 | 5514.43 | 5563.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 5525.50 | 5514.43 | 5563.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 5525.50 | 5514.43 | 5563.27 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 5685.00 | 5525.70 | 5517.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 5704.00 | 5609.22 | 5563.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 5615.00 | 5632.73 | 5588.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-11 11:15:00 | 5618.50 | 5629.88 | 5590.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 5618.50 | 5629.88 | 5590.79 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 5419.50 | 5560.45 | 5570.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 15:15:00 | 5390.50 | 5459.72 | 5508.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 12:15:00 | 5247.50 | 5236.20 | 5315.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 5282.00 | 5246.85 | 5306.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 5282.00 | 5246.85 | 5306.73 | EMA400 retest candle locked (from downside) |

### Cycle 128 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 5353.00 | 5326.38 | 5322.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 5420.50 | 5345.21 | 5331.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 5297.50 | 5392.29 | 5373.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 5297.50 | 5392.29 | 5373.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 5297.50 | 5392.29 | 5373.34 | EMA400 retest candle locked (from upside) |

### Cycle 129 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 5280.50 | 5353.32 | 5357.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 5190.00 | 5306.29 | 5334.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 5283.00 | 5266.83 | 5306.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 10:15:00 | 5350.50 | 5283.56 | 5310.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 5350.50 | 5283.56 | 5310.34 | EMA400 retest candle locked (from downside) |

### Cycle 130 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 5258.00 | 5214.22 | 5210.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 5332.50 | 5237.88 | 5221.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 5289.50 | 5293.02 | 5259.70 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 5158.00 | 5263.61 | 5251.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 5158.00 | 5263.61 | 5251.97 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 5154.50 | 5241.79 | 5243.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 5140.00 | 5221.43 | 5233.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 5169.50 | 5116.69 | 5151.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 5169.50 | 5116.69 | 5151.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 5169.50 | 5116.69 | 5151.56 | EMA400 retest candle locked (from downside) |

### Cycle 132 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 5335.50 | 5097.26 | 5075.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 09:15:00 | 5365.50 | 5297.68 | 5243.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 5343.50 | 5395.65 | 5331.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 5343.50 | 5395.65 | 5331.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 5343.50 | 5395.65 | 5331.58 | EMA400 retest candle locked (from upside) |

### Cycle 133 — SELL (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 14:15:00 | 5253.00 | 5296.37 | 5300.68 | EMA200 below EMA400 |

### Cycle 134 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 11:15:00 | 5318.50 | 5302.93 | 5301.82 | EMA200 above EMA400 |

### Cycle 135 — SELL (started 2026-04-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-15 13:15:00 | 5290.00 | 5300.04 | 5300.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-15 14:15:00 | 5289.50 | 5297.93 | 5299.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 09:15:00 | 5233.50 | 5198.86 | 5233.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-17 09:15:00 | 5233.50 | 5198.86 | 5233.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 5233.50 | 5198.86 | 5233.55 | EMA400 retest candle locked (from downside) |

### Cycle 136 — BUY (started 2026-04-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 10:15:00 | 5281.50 | 5240.56 | 5238.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 11:15:00 | 5295.00 | 5251.45 | 5243.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 15:15:00 | 5254.50 | 5265.64 | 5254.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 15:15:00 | 5254.50 | 5265.64 | 5254.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 5254.50 | 5265.64 | 5254.08 | EMA400 retest candle locked (from upside) |

### Cycle 137 — SELL (started 2026-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 12:15:00 | 5221.00 | 5260.96 | 5262.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 13:15:00 | 5215.00 | 5251.77 | 5257.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 5071.00 | 5015.24 | 5068.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 5071.00 | 5015.24 | 5068.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 5071.00 | 5015.24 | 5068.72 | EMA400 retest candle locked (from downside) |

### Cycle 138 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 5170.50 | 5080.78 | 5072.37 | EMA200 above EMA400 |

### Cycle 139 — SELL (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 10:15:00 | 5032.00 | 5086.34 | 5087.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 11:15:00 | 5029.50 | 5074.97 | 5082.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 13:15:00 | 5073.50 | 5069.72 | 5078.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 13:15:00 | 5073.50 | 5069.72 | 5078.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 5073.50 | 5069.72 | 5078.38 | EMA400 retest candle locked (from downside) |

### Cycle 140 — BUY (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 15:15:00 | 5110.00 | 5083.50 | 5083.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 5148.50 | 5096.50 | 5089.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 11:15:00 | 5086.50 | 5098.50 | 5091.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 11:15:00 | 5086.50 | 5098.50 | 5091.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 5086.50 | 5098.50 | 5091.80 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2026-05-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 12:15:00 | 5024.00 | 5083.60 | 5085.63 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2026-05-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 15:15:00 | 5127.00 | 5087.80 | 5082.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 5255.50 | 5121.34 | 5098.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-06 10:15:00 | 4985.00 | 5094.07 | 5088.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-06 10:15:00 | 4985.00 | 5094.07 | 5088.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 4985.00 | 5094.07 | 5088.10 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-13 13:15:00 | 4818.20 | 2024-05-21 15:15:00 | 5045.00 | STOP_HIT | 1.00 | 4.71% |
| BUY | retest2 | 2024-05-29 11:45:00 | 5181.40 | 2024-05-30 11:15:00 | 5110.70 | STOP_HIT | 1.00 | -1.36% |
| BUY | retest2 | 2024-05-30 09:15:00 | 5176.15 | 2024-05-30 11:15:00 | 5110.70 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-05-30 09:45:00 | 5177.65 | 2024-05-30 11:15:00 | 5110.70 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-05-30 10:30:00 | 5172.10 | 2024-05-30 11:15:00 | 5110.70 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-05-31 09:30:00 | 5178.50 | 2024-05-31 10:15:00 | 5097.85 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2024-06-13 12:15:00 | 5806.00 | 2024-06-18 14:15:00 | 5749.85 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-06-13 13:00:00 | 5819.10 | 2024-06-18 14:15:00 | 5749.85 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-06-14 10:15:00 | 5804.80 | 2024-06-18 14:15:00 | 5749.85 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2024-06-18 09:15:00 | 5846.00 | 2024-06-18 14:15:00 | 5749.85 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2024-06-25 13:30:00 | 5531.30 | 2024-06-28 10:15:00 | 5549.15 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2024-07-03 09:15:00 | 5585.05 | 2024-07-04 09:15:00 | 5551.65 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2024-07-03 11:45:00 | 5577.55 | 2024-07-04 09:15:00 | 5551.65 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-07-03 15:00:00 | 5579.95 | 2024-07-04 09:15:00 | 5551.65 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-07-23 10:15:00 | 5439.45 | 2024-07-23 10:15:00 | 5509.45 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-08-07 15:15:00 | 5229.00 | 2024-08-12 11:15:00 | 5292.20 | STOP_HIT | 1.00 | -1.21% |
| SELL | retest2 | 2024-08-08 10:30:00 | 5215.40 | 2024-08-12 11:15:00 | 5292.20 | STOP_HIT | 1.00 | -1.47% |
| BUY | retest2 | 2024-09-06 12:15:00 | 5718.15 | 2024-09-10 14:15:00 | 5662.80 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest2 | 2024-09-09 09:30:00 | 5708.60 | 2024-09-10 14:15:00 | 5662.80 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2024-09-10 13:00:00 | 5711.00 | 2024-09-10 14:15:00 | 5662.80 | STOP_HIT | 1.00 | -0.84% |
| BUY | retest2 | 2024-09-16 13:45:00 | 5794.55 | 2024-09-26 09:15:00 | 5932.00 | STOP_HIT | 1.00 | 2.37% |
| BUY | retest2 | 2024-09-17 09:15:00 | 5803.30 | 2024-09-26 09:15:00 | 5932.00 | STOP_HIT | 1.00 | 2.22% |
| BUY | retest2 | 2024-09-17 09:45:00 | 5803.35 | 2024-09-26 09:15:00 | 5932.00 | STOP_HIT | 1.00 | 2.22% |
| SELL | retest2 | 2024-10-22 10:15:00 | 5162.85 | 2024-10-25 10:15:00 | 4904.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 11:15:00 | 5176.35 | 2024-10-25 10:15:00 | 4917.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 13:15:00 | 5188.00 | 2024-10-25 10:15:00 | 4928.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 14:45:00 | 5179.70 | 2024-10-25 10:15:00 | 4920.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-23 15:15:00 | 5133.95 | 2024-10-25 10:15:00 | 4877.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-24 12:15:00 | 5122.45 | 2024-10-29 09:15:00 | 4866.33 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-22 10:15:00 | 5162.85 | 2024-10-30 10:15:00 | 4850.10 | STOP_HIT | 0.50 | 6.06% |
| SELL | retest2 | 2024-10-22 11:15:00 | 5176.35 | 2024-10-30 10:15:00 | 4850.10 | STOP_HIT | 0.50 | 6.30% |
| SELL | retest2 | 2024-10-22 13:15:00 | 5188.00 | 2024-10-30 10:15:00 | 4850.10 | STOP_HIT | 0.50 | 6.51% |
| SELL | retest2 | 2024-10-22 14:45:00 | 5179.70 | 2024-10-30 10:15:00 | 4850.10 | STOP_HIT | 0.50 | 6.36% |
| SELL | retest2 | 2024-10-23 15:15:00 | 5133.95 | 2024-10-30 10:15:00 | 4850.10 | STOP_HIT | 0.50 | 5.53% |
| SELL | retest2 | 2024-10-24 12:15:00 | 5122.45 | 2024-10-30 10:15:00 | 4850.10 | STOP_HIT | 0.50 | 5.32% |
| SELL | retest2 | 2024-11-12 12:15:00 | 4748.50 | 2024-11-13 14:15:00 | 4511.07 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 12:15:00 | 4748.50 | 2024-11-14 10:15:00 | 4582.90 | STOP_HIT | 0.50 | 3.49% |
| BUY | retest2 | 2024-11-27 11:15:00 | 4867.65 | 2024-11-28 13:15:00 | 4806.70 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2024-11-27 12:45:00 | 4868.50 | 2024-11-28 13:15:00 | 4806.70 | STOP_HIT | 1.00 | -1.27% |
| SELL | retest2 | 2024-12-03 11:30:00 | 4733.05 | 2024-12-11 13:15:00 | 4644.20 | STOP_HIT | 1.00 | 1.88% |
| SELL | retest2 | 2024-12-16 09:15:00 | 4546.25 | 2024-12-23 09:15:00 | 4318.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-16 09:15:00 | 4546.25 | 2024-12-26 10:15:00 | 4291.45 | STOP_HIT | 0.50 | 5.60% |
