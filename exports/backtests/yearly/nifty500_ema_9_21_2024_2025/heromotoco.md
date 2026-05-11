# Hero MotoCorp Ltd. (HEROMOTOCO)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
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
| ALERT2_SKIP | 53 |
| ALERT3 | 253 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 112 |
| PARTIAL | 11 |
| TARGET_HIT | 5 |
| STOP_HIT | 107 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 123 (incl. partial bookings)
- **Trades open at end:** 3
- **Winners / losers:** 47 / 76
- **Target hits / Stop hits / Partials:** 5 / 107 / 11
- **Avg / median % per leg:** 0.80% / -0.60%
- **Sum % (uncompounded):** 98.29%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 53 | 15 | 28.3% | 5 | 48 | 0 | 0.47% | 24.9% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.17% | -6.5% |
| BUY @ 3rd Alert (retest2) | 50 | 15 | 30.0% | 5 | 45 | 0 | 0.63% | 31.4% |
| SELL (all) | 70 | 32 | 45.7% | 0 | 59 | 11 | 1.05% | 73.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 70 | 32 | 45.7% | 0 | 59 | 11 | 1.05% | 73.4% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -2.17% | -6.5% |
| retest2 (combined) | 120 | 47 | 39.2% | 5 | 104 | 11 | 0.87% | 104.8% |

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
| ALERT2_SIDEWAYS | 2025-01-09 11:30:00 | 4134.70 | 4138.36 | 4160.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 13:15:00 | 4160.55 | 4144.64 | 4159.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 13:45:00 | 4159.90 | 4144.64 | 4159.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 14:15:00 | 4139.00 | 4143.51 | 4157.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 09:30:00 | 4121.45 | 4136.84 | 4151.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 11:30:00 | 4128.95 | 4137.09 | 4149.49 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-10 14:00:00 | 4121.45 | 4137.73 | 4147.86 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-15 10:00:00 | 4117.65 | 4100.70 | 4102.59 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-15 10:15:00 | 4119.00 | 4104.36 | 4104.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

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
| ALERT3_SIDEWAYS | 2025-01-17 09:30:00 | 4085.00 | 4083.19 | 4091.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 10:15:00 | 4083.00 | 4083.15 | 4090.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 10:30:00 | 4097.60 | 4083.15 | 4090.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 11:15:00 | 4109.35 | 4088.39 | 4092.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-17 12:00:00 | 4109.35 | 4088.39 | 4092.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-17 12:15:00 | 4109.00 | 4092.51 | 4093.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 13:15:00 | 4099.10 | 4092.51 | 4093.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-17 15:00:00 | 4099.75 | 4094.12 | 4094.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-23 10:15:00 | 4110.35 | 4068.27 | 4066.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

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
| ALERT2_SIDEWAYS | 2025-01-28 11:00:00 | 4028.75 | 4027.35 | 4044.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 11:15:00 | 4040.35 | 4029.95 | 4044.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-28 12:00:00 | 4040.35 | 4029.95 | 4044.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-28 12:15:00 | 4016.10 | 4027.18 | 4041.66 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 4092.00 | 4052.41 | 4048.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-30 09:15:00 | 4118.70 | 4080.95 | 4066.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 09:15:00 | 4281.00 | 4289.74 | 4219.75 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-02-01 12:15:00 | 4456.85 | 4298.81 | 4236.20 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 10:15:00 | 4289.05 | 4352.79 | 4300.06 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-02-03 10:15:00 | 4289.05 | 4352.79 | 4300.06 | SL hit (close<ema400) qty=1.00 sl=4300.06 alert=retest1 |

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
| ALERT3_SIDEWAYS | 2025-02-07 10:30:00 | 4248.00 | 4251.27 | 4258.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 4258.85 | 4252.78 | 4258.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-07 11:45:00 | 4278.55 | 4252.78 | 4258.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 12:15:00 | 4247.15 | 4251.66 | 4257.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 14:30:00 | 4244.00 | 4253.03 | 4257.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 15:15:00 | 4282.00 | 4258.82 | 4259.57 | SL hit (close>static) qty=1.00 sl=4266.20 alert=retest2 |

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
| ALERT2_SIDEWAYS | 2025-02-17 13:00:00 | 3858.00 | 3856.12 | 3918.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 3883.05 | 3850.29 | 3872.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:00:00 | 3883.05 | 3850.29 | 3872.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 3874.50 | 3855.13 | 3872.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 3879.05 | 3855.13 | 3872.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 11:15:00 | 3881.35 | 3860.37 | 3873.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 11:30:00 | 3880.15 | 3860.37 | 3873.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 12:15:00 | 3880.25 | 3864.35 | 3874.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-19 14:15:00 | 3865.35 | 3868.42 | 3875.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-20 09:15:00 | 3913.60 | 3881.43 | 3879.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2025-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 09:15:00 | 3913.60 | 3881.43 | 3879.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 10:15:00 | 3935.00 | 3892.14 | 3884.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 3844.00 | 3893.01 | 3890.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 3844.00 | 3893.01 | 3890.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 3844.00 | 3893.01 | 3890.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 3844.00 | 3893.01 | 3890.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — SELL (started 2025-02-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 10:15:00 | 3848.55 | 3884.12 | 3886.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 3807.75 | 3851.67 | 3867.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 11:15:00 | 3853.70 | 3851.43 | 3864.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 11:15:00 | 3853.70 | 3851.43 | 3864.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 3853.70 | 3851.43 | 3864.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 11:45:00 | 3857.45 | 3851.43 | 3864.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 3875.45 | 3857.81 | 3865.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 14:00:00 | 3875.45 | 3857.81 | 3865.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 14:15:00 | 3888.90 | 3864.03 | 3867.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 15:00:00 | 3888.90 | 3864.03 | 3867.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 11:15:00 | 3871.30 | 3864.78 | 3867.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 12:00:00 | 3871.30 | 3864.78 | 3867.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 12:15:00 | 3866.65 | 3865.16 | 3866.97 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 14:00:00 | 3850.10 | 3862.15 | 3865.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-03 09:15:00 | 3657.59 | 3694.74 | 3738.83 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-05 09:15:00 | 3563.00 | 3554.65 | 3610.16 | SL hit (close>ema200) qty=0.50 sl=3554.65 alert=retest2 |

### Cycle 58 — BUY (started 2025-03-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 13:15:00 | 3649.05 | 3614.21 | 3611.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 15:15:00 | 3658.00 | 3628.50 | 3618.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 12:15:00 | 3652.55 | 3659.10 | 3639.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 12:45:00 | 3653.00 | 3659.10 | 3639.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 3633.25 | 3652.78 | 3642.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:30:00 | 3635.00 | 3652.78 | 3642.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 3621.30 | 3646.48 | 3640.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:30:00 | 3624.45 | 3646.48 | 3640.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2025-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 11:15:00 | 3586.50 | 3634.49 | 3635.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 3581.20 | 3615.24 | 3626.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 10:15:00 | 3621.95 | 3611.31 | 3621.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 10:15:00 | 3621.95 | 3611.31 | 3621.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 10:15:00 | 3621.95 | 3611.31 | 3621.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-11 11:00:00 | 3621.95 | 3611.31 | 3621.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 11:15:00 | 3613.25 | 3611.70 | 3620.30 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-11 12:45:00 | 3595.00 | 3608.27 | 3617.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-11 13:15:00 | 3626.70 | 3611.96 | 3618.76 | SL hit (close>static) qty=1.00 sl=3623.65 alert=retest2 |

### Cycle 60 — BUY (started 2025-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-11 15:15:00 | 3651.00 | 3625.22 | 3623.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-12 10:15:00 | 3655.75 | 3632.39 | 3627.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-12 11:15:00 | 3615.95 | 3629.10 | 3626.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-12 11:15:00 | 3615.95 | 3629.10 | 3626.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 11:15:00 | 3615.95 | 3629.10 | 3626.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-12 12:00:00 | 3615.95 | 3629.10 | 3626.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2025-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-12 12:15:00 | 3605.10 | 3624.30 | 3624.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-13 10:15:00 | 3586.90 | 3610.40 | 3617.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-18 09:15:00 | 3528.00 | 3509.28 | 3539.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-18 11:15:00 | 3548.10 | 3519.10 | 3538.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 11:15:00 | 3548.10 | 3519.10 | 3538.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 12:00:00 | 3548.10 | 3519.10 | 3538.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 12:15:00 | 3558.30 | 3526.94 | 3540.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 13:00:00 | 3558.30 | 3526.94 | 3540.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 10:15:00 | 3532.80 | 3542.96 | 3545.17 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2025-03-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-20 09:15:00 | 3568.95 | 3544.60 | 3544.08 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-20 11:15:00 | 3615.00 | 3567.28 | 3555.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-21 13:15:00 | 3636.65 | 3640.16 | 3608.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-21 13:45:00 | 3634.80 | 3640.16 | 3608.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 3608.65 | 3631.75 | 3612.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 09:45:00 | 3610.00 | 3631.75 | 3612.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 10:15:00 | 3595.00 | 3624.40 | 3610.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-24 11:15:00 | 3608.35 | 3624.40 | 3610.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 11:15:00 | 3613.05 | 3622.13 | 3610.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 12:15:00 | 3626.70 | 3622.13 | 3610.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 09:45:00 | 3615.15 | 3631.56 | 3626.07 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 10:30:00 | 3619.90 | 3629.73 | 3625.73 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-04 09:15:00 | 3635.35 | 3731.06 | 3743.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 3635.35 | 3731.06 | 3743.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-07 09:15:00 | 3455.00 | 3637.54 | 3687.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 12:15:00 | 3566.00 | 3544.27 | 3587.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 13:00:00 | 3566.00 | 3544.27 | 3587.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 3571.90 | 3554.73 | 3585.15 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 3562.85 | 3559.99 | 3584.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-09 10:15:00 | 3597.90 | 3564.29 | 3582.22 | SL hit (close>static) qty=1.00 sl=3591.95 alert=retest2 |

### Cycle 64 — BUY (started 2025-04-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-09 12:15:00 | 3638.15 | 3595.64 | 3594.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-11 09:15:00 | 3673.00 | 3620.52 | 3607.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 12:15:00 | 3752.00 | 3756.94 | 3719.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-16 13:00:00 | 3752.00 | 3756.94 | 3719.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 3680.20 | 3751.24 | 3729.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 09:30:00 | 3815.00 | 3764.58 | 3744.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-28 10:15:00 | 3881.00 | 3892.32 | 3893.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

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
| ALERT3_SIDEWAYS | 2025-05-02 09:45:00 | 3900.50 | 3848.20 | 3859.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 10:15:00 | 3846.00 | 3847.76 | 3858.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 11:45:00 | 3831.20 | 3841.03 | 3854.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:15:00 | 3833.50 | 3802.66 | 3805.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-06 11:15:00 | 3854.20 | 3812.97 | 3810.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

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
| ALERT3_SIDEWAYS | 2025-05-09 10:00:00 | 3843.00 | 3823.88 | 3831.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 10:15:00 | 3847.00 | 3828.50 | 3833.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 11:15:00 | 3880.00 | 3828.50 | 3833.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — BUY (started 2025-05-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 11:15:00 | 3874.90 | 3837.78 | 3837.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 3920.80 | 3861.09 | 3848.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 4329.00 | 4368.07 | 4312.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 4329.00 | 4368.07 | 4312.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 4329.00 | 4368.07 | 4312.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:00:00 | 4329.00 | 4368.07 | 4312.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 4316.00 | 4357.65 | 4313.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 10:45:00 | 4309.70 | 4357.65 | 4313.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 4312.80 | 4348.68 | 4313.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:30:00 | 4298.10 | 4348.68 | 4313.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 12:15:00 | 4304.00 | 4339.75 | 4312.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:45:00 | 4309.90 | 4339.75 | 4312.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 13:15:00 | 4265.50 | 4324.90 | 4307.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:00:00 | 4265.50 | 4324.90 | 4307.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 14:15:00 | 4243.10 | 4308.54 | 4302.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 14:30:00 | 4240.00 | 4308.54 | 4302.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 4249.00 | 4296.63 | 4297.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 4220.50 | 4272.50 | 4285.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 4265.00 | 4262.98 | 4277.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-21 14:45:00 | 4271.10 | 4262.98 | 4277.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 12:15:00 | 4268.00 | 4255.08 | 4266.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 13:00:00 | 4268.00 | 4255.08 | 4266.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 13:15:00 | 4258.40 | 4255.75 | 4265.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 14:15:00 | 4253.00 | 4255.75 | 4265.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 09:15:00 | 4319.20 | 4273.72 | 4272.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 72 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 4319.20 | 4273.72 | 4272.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 4398.60 | 4316.39 | 4295.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 15:15:00 | 4358.00 | 4358.95 | 4331.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 09:15:00 | 4323.90 | 4358.95 | 4331.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 4302.80 | 4347.72 | 4328.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 4302.80 | 4347.72 | 4328.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 4327.00 | 4343.58 | 4328.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 13:30:00 | 4354.00 | 4338.23 | 4333.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 15:15:00 | 4375.00 | 4340.22 | 4334.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 09:45:00 | 4364.00 | 4351.26 | 4341.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-29 10:45:00 | 4355.80 | 4352.03 | 4342.38 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 4358.80 | 4353.31 | 4344.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:30:00 | 4352.10 | 4353.31 | 4344.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 4370.50 | 4357.59 | 4349.43 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-30 14:15:00 | 4302.80 | 4343.83 | 4345.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2025-05-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 14:15:00 | 4302.80 | 4343.83 | 4345.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 4216.50 | 4312.97 | 4331.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 13:15:00 | 4244.50 | 4240.73 | 4265.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-03 13:45:00 | 4246.90 | 4240.73 | 4265.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 09:15:00 | 4229.30 | 4203.10 | 4217.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:00:00 | 4229.30 | 4203.10 | 4217.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 10:15:00 | 4256.80 | 4213.84 | 4221.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 10:30:00 | 4275.00 | 4213.84 | 4221.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 4280.80 | 4235.42 | 4230.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 4286.30 | 4260.34 | 4244.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-11 14:15:00 | 4413.50 | 4417.95 | 4378.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-11 15:00:00 | 4413.50 | 4417.95 | 4378.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
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
| ALERT2_SIDEWAYS | 2025-06-18 11:30:00 | 4395.00 | 4384.45 | 4375.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 4391.10 | 4385.78 | 4376.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 12:30:00 | 4384.00 | 4385.78 | 4376.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 4365.40 | 4381.70 | 4375.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:45:00 | 4372.30 | 4381.70 | 4375.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 4358.30 | 4377.02 | 4374.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 15:15:00 | 4358.00 | 4377.02 | 4374.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 15:15:00 | 4358.00 | 4373.22 | 4372.72 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 09:15:00 | 4373.20 | 4373.22 | 4372.72 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-19 13:00:00 | 4370.10 | 4376.92 | 4375.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 09:15:00 | 4282.40 | 4358.83 | 4367.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

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
| ALERT3_SIDEWAYS | 2025-07-07 13:00:00 | 4304.00 | 4327.31 | 4315.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 13:15:00 | 4302.40 | 4322.33 | 4314.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-07 13:45:00 | 4303.00 | 4322.33 | 4314.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2025-07-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 09:15:00 | 4274.90 | 4307.69 | 4309.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 10:15:00 | 4248.00 | 4295.75 | 4303.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 11:15:00 | 4297.00 | 4296.00 | 4302.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 12:00:00 | 4297.00 | 4296.00 | 4302.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 12:15:00 | 4305.40 | 4297.88 | 4303.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 12:45:00 | 4302.00 | 4297.88 | 4303.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 4283.50 | 4295.00 | 4301.37 | EMA400 retest candle locked (from downside) |

### Cycle 82 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 4320.00 | 4305.17 | 4304.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 11:15:00 | 4327.50 | 4309.64 | 4306.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-10 11:15:00 | 4324.60 | 4326.30 | 4318.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-10 11:15:00 | 4324.60 | 4326.30 | 4318.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 11:15:00 | 4324.60 | 4326.30 | 4318.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 11:45:00 | 4323.30 | 4326.30 | 4318.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 4329.90 | 4327.02 | 4319.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 13:00:00 | 4329.90 | 4327.02 | 4319.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 14:15:00 | 4322.60 | 4326.61 | 4320.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-10 15:00:00 | 4322.60 | 4326.61 | 4320.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 15:15:00 | 4315.80 | 4324.45 | 4320.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 09:15:00 | 4334.80 | 4324.45 | 4320.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 09:15:00 | 4310.10 | 4321.58 | 4319.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 10:00:00 | 4310.10 | 4321.58 | 4319.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 4275.30 | 4312.32 | 4315.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 12:15:00 | 4232.30 | 4289.99 | 4304.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 14:15:00 | 4253.90 | 4242.43 | 4262.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 15:00:00 | 4253.90 | 4242.43 | 4262.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 4326.00 | 4259.52 | 4266.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 09:30:00 | 4300.40 | 4259.52 | 4266.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 84 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 4399.70 | 4287.56 | 4278.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 4402.20 | 4310.49 | 4290.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 4393.20 | 4393.57 | 4345.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 09:45:00 | 4400.30 | 4393.57 | 4345.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 4413.10 | 4429.90 | 4409.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:30:00 | 4407.30 | 4429.90 | 4409.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 4403.50 | 4424.62 | 4409.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:30:00 | 4400.90 | 4424.62 | 4409.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 12:15:00 | 4406.10 | 4420.91 | 4409.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 12:45:00 | 4406.20 | 4420.91 | 4409.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 13:15:00 | 4406.90 | 4418.11 | 4408.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 11:45:00 | 4418.00 | 4408.23 | 4406.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 12:30:00 | 4418.50 | 4409.70 | 4407.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 09:45:00 | 4414.10 | 4410.92 | 4408.48 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 10:15:00 | 4383.50 | 4405.44 | 4406.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 10:15:00 | 4383.50 | 4405.44 | 4406.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 11:15:00 | 4360.60 | 4396.47 | 4402.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 09:15:00 | 4374.20 | 4366.18 | 4382.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 09:45:00 | 4373.20 | 4366.18 | 4382.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 4366.50 | 4358.18 | 4369.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 4296.10 | 4358.18 | 4369.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-29 10:15:00 | 4321.50 | 4290.25 | 4286.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2025-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 10:15:00 | 4321.50 | 4290.25 | 4286.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-29 12:15:00 | 4330.00 | 4303.41 | 4293.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-30 10:15:00 | 4257.80 | 4304.42 | 4299.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-30 10:15:00 | 4257.80 | 4304.42 | 4299.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 4257.80 | 4304.42 | 4299.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:45:00 | 4265.40 | 4304.42 | 4299.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — SELL (started 2025-07-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 11:15:00 | 4227.50 | 4289.04 | 4292.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-31 09:15:00 | 4218.00 | 4253.70 | 4271.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 11:15:00 | 4264.80 | 4254.77 | 4269.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 11:15:00 | 4264.80 | 4254.77 | 4269.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 4264.80 | 4254.77 | 4269.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 11:45:00 | 4270.00 | 4254.77 | 4269.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 4282.40 | 4260.29 | 4270.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 12:45:00 | 4293.50 | 4260.29 | 4270.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 13:15:00 | 4293.60 | 4266.96 | 4272.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:30:00 | 4288.00 | 4266.96 | 4272.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 4259.10 | 4261.48 | 4268.20 | EMA400 retest candle locked (from downside) |

### Cycle 88 — BUY (started 2025-08-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 11:15:00 | 4297.00 | 4273.18 | 4272.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-04 09:15:00 | 4426.40 | 4320.17 | 4296.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 09:15:00 | 4491.40 | 4522.10 | 4466.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-06 10:00:00 | 4491.40 | 4522.10 | 4466.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 14:15:00 | 4475.00 | 4496.41 | 4473.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:30:00 | 4470.20 | 4496.41 | 4473.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 15:15:00 | 4470.10 | 4491.15 | 4472.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-07 09:15:00 | 4556.60 | 4491.15 | 4472.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 4533.50 | 4499.62 | 4478.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-07 12:45:00 | 4607.60 | 4531.81 | 4499.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-12 09:15:00 | 4604.00 | 4573.12 | 4571.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-08-18 09:15:00 | 5068.36 | 4790.95 | 4735.72 | Target hit (10%) qty=1.00 alert=retest2 |

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
| ALERT3_SIDEWAYS | 2025-08-26 15:00:00 | 5060.70 | 5080.72 | 5069.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 15:15:00 | 5074.90 | 5079.56 | 5069.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-28 09:15:00 | 5151.00 | 5079.56 | 5069.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-29 09:45:00 | 5102.90 | 5113.66 | 5099.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 11:15:00 | 5348.50 | 5402.02 | 5407.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — SELL (started 2025-09-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 11:15:00 | 5348.50 | 5402.02 | 5407.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-11 09:15:00 | 5306.00 | 5355.80 | 5380.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 5347.00 | 5324.19 | 5347.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-12 09:15:00 | 5347.00 | 5324.19 | 5347.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 09:15:00 | 5347.00 | 5324.19 | 5347.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 09:30:00 | 5374.50 | 5324.19 | 5347.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 10:15:00 | 5352.00 | 5329.76 | 5348.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-12 11:00:00 | 5352.00 | 5329.76 | 5348.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 11:15:00 | 5315.00 | 5326.80 | 5345.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-12 13:30:00 | 5293.00 | 5318.04 | 5337.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-15 11:30:00 | 5291.00 | 5306.10 | 5324.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 09:30:00 | 5293.00 | 5303.33 | 5314.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 13:15:00 | 5333.00 | 5322.68 | 5321.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

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
| ALERT2_SIDEWAYS | 2025-09-17 14:30:00 | 5354.00 | 5352.57 | 5340.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 12:15:00 | 5351.00 | 5358.47 | 5348.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-18 12:45:00 | 5352.00 | 5358.47 | 5348.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 13:15:00 | 5357.50 | 5358.27 | 5349.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 15:00:00 | 5375.50 | 5361.72 | 5351.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 14:15:00 | 5374.00 | 5404.25 | 5407.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — SELL (started 2025-09-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 14:15:00 | 5374.00 | 5404.25 | 5407.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 15:15:00 | 5362.50 | 5395.90 | 5402.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 10:15:00 | 5308.50 | 5307.58 | 5340.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-25 11:15:00 | 5330.00 | 5307.58 | 5340.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 5338.00 | 5309.24 | 5333.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:00:00 | 5338.00 | 5309.24 | 5333.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 5345.00 | 5316.39 | 5334.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:45:00 | 5369.00 | 5316.39 | 5334.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 5356.50 | 5324.41 | 5336.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 5358.00 | 5324.41 | 5336.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

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
| ALERT3_SIDEWAYS | 2025-09-29 14:00:00 | 5384.50 | 5341.80 | 5345.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 5283.00 | 5330.04 | 5339.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 14:45:00 | 5424.00 | 5330.04 | 5339.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 5345.50 | 5333.13 | 5340.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:15:00 | 5370.00 | 5333.13 | 5340.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 5375.00 | 5341.51 | 5343.35 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2025-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 10:15:00 | 5400.00 | 5353.20 | 5348.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 11:15:00 | 5423.00 | 5367.16 | 5355.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 14:15:00 | 5427.50 | 5434.18 | 5410.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-01 15:00:00 | 5427.50 | 5434.18 | 5410.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 5392.50 | 5426.38 | 5410.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 11:00:00 | 5526.00 | 5446.30 | 5421.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 15:15:00 | 5520.00 | 5554.52 | 5558.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 15:15:00 | 5520.00 | 5554.52 | 5558.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 5487.50 | 5541.12 | 5552.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 12:15:00 | 5526.50 | 5515.81 | 5527.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-10 12:15:00 | 5526.50 | 5515.81 | 5527.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 12:15:00 | 5526.50 | 5515.81 | 5527.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 12:45:00 | 5523.50 | 5515.81 | 5527.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 13:15:00 | 5521.00 | 5516.85 | 5526.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 14:30:00 | 5498.00 | 5514.18 | 5524.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-13 09:15:00 | 5552.50 | 5521.09 | 5525.74 | SL hit (close>static) qty=1.00 sl=5532.50 alert=retest2 |

### Cycle 100 — BUY (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 11:15:00 | 5552.00 | 5529.98 | 5529.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 12:15:00 | 5577.50 | 5539.48 | 5533.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 15:15:00 | 5560.00 | 5561.61 | 5552.08 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-15 09:15:00 | 5594.50 | 5561.61 | 5552.08 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 5518.00 | 5563.95 | 5558.48 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-15 13:15:00 | 5518.00 | 5563.95 | 5558.48 | SL hit (close<ema400) qty=1.00 sl=5558.48 alert=retest1 |

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
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-21 13:45:00 | 5650.00 | 5638.84 | 5617.89 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 5646.50 | 5639.12 | 5621.57 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-23 14:15:00 | 5571.50 | 5627.76 | 5623.56 | SL hit (close<ema400) qty=1.00 sl=5623.56 alert=retest1 |

### Cycle 103 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 5584.00 | 5619.01 | 5619.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 10:15:00 | 5560.00 | 5604.48 | 5613.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 5596.50 | 5563.61 | 5583.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 5596.50 | 5563.61 | 5583.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 5596.50 | 5563.61 | 5583.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 5596.50 | 5563.61 | 5583.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 5647.50 | 5580.39 | 5588.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:45:00 | 5652.00 | 5580.39 | 5588.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 12:15:00 | 5651.00 | 5601.49 | 5597.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 09:15:00 | 5655.00 | 5630.44 | 5614.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 11:15:00 | 5612.50 | 5627.66 | 5615.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 11:15:00 | 5612.50 | 5627.66 | 5615.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 5612.50 | 5627.66 | 5615.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 12:00:00 | 5612.50 | 5627.66 | 5615.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 12:15:00 | 5605.00 | 5623.13 | 5614.79 | EMA400 retest candle locked (from upside) |

### Cycle 105 — SELL (started 2025-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 09:15:00 | 5566.50 | 5606.17 | 5608.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 11:15:00 | 5509.00 | 5548.97 | 5572.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-31 09:15:00 | 5616.00 | 5544.75 | 5559.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-31 09:15:00 | 5616.00 | 5544.75 | 5559.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 5616.00 | 5544.75 | 5559.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 5616.00 | 5544.75 | 5559.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 5613.00 | 5558.40 | 5564.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 11:00:00 | 5613.00 | 5558.40 | 5564.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

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
| ALERT2_SIDEWAYS | 2025-11-06 12:45:00 | 5357.00 | 5342.02 | 5401.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 5374.00 | 5317.65 | 5343.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:30:00 | 5362.00 | 5317.65 | 5343.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 5389.50 | 5332.02 | 5347.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:45:00 | 5386.50 | 5332.02 | 5347.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 5396.00 | 5360.95 | 5358.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 11:15:00 | 5411.00 | 5370.36 | 5363.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 5520.00 | 5523.96 | 5485.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 5520.00 | 5523.96 | 5485.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 15:15:00 | 5498.00 | 5518.77 | 5486.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 5549.00 | 5518.77 | 5486.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 10:30:00 | 5603.00 | 5534.79 | 5499.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 15:00:00 | 5537.50 | 5552.47 | 5521.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-11-26 09:15:00 | 6103.90 | 6065.83 | 6029.25 | Target hit (10%) qty=1.00 alert=retest2 |

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
| ALERT2_SIDEWAYS | 2025-12-05 13:45:00 | 6332.50 | 6339.94 | 6308.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 6231.00 | 6322.91 | 6309.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 10:00:00 | 6231.00 | 6322.91 | 6309.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 10:15:00 | 6217.50 | 6301.83 | 6300.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 11:00:00 | 6217.50 | 6301.83 | 6300.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 111 — SELL (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 11:15:00 | 6199.50 | 6281.36 | 6291.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 6173.00 | 6245.71 | 6272.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 09:15:00 | 6004.00 | 5977.17 | 6043.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-11 10:00:00 | 6004.00 | 5977.17 | 6043.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 5999.50 | 5979.29 | 6011.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 09:45:00 | 6033.00 | 5979.29 | 6011.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 5994.50 | 5982.33 | 6009.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 11:45:00 | 5975.50 | 5981.27 | 6006.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 12:45:00 | 5974.50 | 5979.01 | 6003.55 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:15:00 | 5676.72 | 5811.90 | 5873.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 09:15:00 | 5675.77 | 5811.90 | 5873.06 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 10:15:00 | 5768.50 | 5753.63 | 5800.83 | SL hit (close>ema200) qty=0.50 sl=5753.63 alert=retest2 |

### Cycle 112 — BUY (started 2025-12-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 14:15:00 | 5721.50 | 5662.89 | 5655.27 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 10:15:00 | 5746.00 | 5694.76 | 5672.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-06 15:15:00 | 5990.00 | 5992.03 | 5956.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 09:15:00 | 5981.00 | 5992.03 | 5956.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 5950.00 | 5986.68 | 5967.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 13:30:00 | 5951.00 | 5986.68 | 5967.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 14:15:00 | 5989.00 | 5987.15 | 5969.85 | EMA400 retest candle locked (from upside) |

### Cycle 113 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 5901.50 | 5961.88 | 5962.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 11:15:00 | 5863.00 | 5942.10 | 5953.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 5749.00 | 5717.53 | 5752.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 14:15:00 | 5749.00 | 5717.53 | 5752.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 5749.00 | 5717.53 | 5752.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 5749.00 | 5717.53 | 5752.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 5705.00 | 5715.02 | 5748.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:45:00 | 5698.00 | 5708.82 | 5742.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:00:00 | 5700.00 | 5708.28 | 5736.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:45:00 | 5701.00 | 5695.20 | 5712.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 12:15:00 | 5700.00 | 5695.20 | 5712.82 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 12:15:00 | 5668.00 | 5689.76 | 5708.75 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:15:00 | 5650.50 | 5689.76 | 5708.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-19 12:15:00 | 5729.00 | 5694.94 | 5699.05 | SL hit (close>static) qty=1.00 sl=5720.00 alert=retest2 |

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
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 5637.50 | 5578.25 | 5618.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 5583.50 | 5579.30 | 5614.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:45:00 | 5566.50 | 5579.54 | 5611.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 12:30:00 | 5569.00 | 5571.53 | 5605.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 15:15:00 | 5510.00 | 5425.90 | 5425.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 5510.00 | 5425.90 | 5425.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 09:15:00 | 5550.50 | 5450.82 | 5436.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 11:15:00 | 5534.00 | 5539.21 | 5503.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-30 12:00:00 | 5534.00 | 5539.21 | 5503.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 5481.50 | 5527.67 | 5501.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 13:00:00 | 5481.50 | 5527.67 | 5501.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 5515.50 | 5525.23 | 5502.70 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 14:45:00 | 5536.00 | 5525.99 | 5505.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 12:30:00 | 5542.00 | 5563.35 | 5535.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:15:00 | 5531.00 | 5543.52 | 5532.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-02 10:15:00 | 5430.50 | 5510.11 | 5518.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

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
| ALERT2_SIDEWAYS | 2026-02-05 09:45:00 | 5794.50 | 5814.30 | 5743.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 14:15:00 | 5764.50 | 5791.60 | 5757.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 14:45:00 | 5759.00 | 5791.60 | 5757.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 5800.00 | 5791.51 | 5763.60 | EMA400 retest candle locked (from upside) |

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
| ALERT2_SIDEWAYS | 2026-02-17 10:30:00 | 5528.00 | 5525.07 | 5577.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 13:15:00 | 5554.00 | 5535.67 | 5569.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 14:00:00 | 5554.00 | 5535.67 | 5569.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 14:15:00 | 5577.50 | 5544.03 | 5570.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 15:00:00 | 5577.50 | 5544.03 | 5570.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 5565.00 | 5548.23 | 5569.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:15:00 | 5609.50 | 5548.23 | 5569.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 09:15:00 | 5575.50 | 5553.68 | 5570.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 09:30:00 | 5624.50 | 5553.68 | 5570.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 10:15:00 | 5583.00 | 5559.55 | 5571.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 10:45:00 | 5599.50 | 5559.55 | 5571.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 11:15:00 | 5563.00 | 5560.24 | 5570.84 | EMA400 retest candle locked (from downside) |

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
| ALERT2_SIDEWAYS | 2026-02-20 14:00:00 | 5436.50 | 5435.23 | 5480.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 09:15:00 | 5475.50 | 5451.06 | 5477.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 09:30:00 | 5504.50 | 5451.06 | 5477.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 10:15:00 | 5498.50 | 5460.55 | 5479.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 11:30:00 | 5465.50 | 5459.74 | 5476.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 10:15:00 | 5505.50 | 5483.05 | 5481.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 124 — BUY (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 10:15:00 | 5505.50 | 5483.05 | 5481.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 15:15:00 | 5520.00 | 5498.49 | 5490.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 09:15:00 | 5747.00 | 5749.79 | 5691.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-27 10:15:00 | 5738.50 | 5749.79 | 5691.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 13:15:00 | 5743.00 | 5739.83 | 5704.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 13:30:00 | 5703.50 | 5739.83 | 5704.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 5705.50 | 5732.96 | 5704.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 5705.50 | 5732.96 | 5704.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 5715.00 | 5729.37 | 5705.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 5679.00 | 5729.37 | 5705.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 09:15:00 | 5681.50 | 5719.80 | 5703.67 | EMA400 retest candle locked (from upside) |

### Cycle 125 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 5561.00 | 5669.99 | 5682.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 12:15:00 | 5530.50 | 5642.09 | 5668.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 5525.50 | 5514.43 | 5563.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 09:15:00 | 5525.50 | 5514.43 | 5563.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 5525.50 | 5514.43 | 5563.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 11:15:00 | 5497.00 | 5514.04 | 5558.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-05 14:15:00 | 5571.50 | 5531.01 | 5552.80 | SL hit (close>static) qty=1.00 sl=5570.00 alert=retest2 |

### Cycle 126 — BUY (started 2026-03-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 10:15:00 | 5685.00 | 5525.70 | 5517.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-10 14:15:00 | 5704.00 | 5609.22 | 5563.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 10:15:00 | 5615.00 | 5632.73 | 5588.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 10:45:00 | 5620.00 | 5632.73 | 5588.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 11:15:00 | 5618.50 | 5629.88 | 5590.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 11:45:00 | 5599.50 | 5629.88 | 5590.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 5578.50 | 5619.61 | 5589.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:00:00 | 5578.50 | 5619.61 | 5589.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 5583.50 | 5612.38 | 5589.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:45:00 | 5569.00 | 5612.38 | 5589.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 5561.00 | 5602.11 | 5586.56 | EMA400 retest candle locked (from upside) |

### Cycle 127 — SELL (started 2026-03-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 09:15:00 | 5419.50 | 5560.45 | 5570.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 15:15:00 | 5390.50 | 5459.72 | 5508.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 12:15:00 | 5247.50 | 5236.20 | 5315.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 13:00:00 | 5247.50 | 5236.20 | 5315.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 5282.00 | 5246.85 | 5306.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:30:00 | 5303.50 | 5246.85 | 5306.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 5360.50 | 5271.92 | 5307.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 5360.50 | 5271.92 | 5307.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 5322.00 | 5281.94 | 5309.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 5296.00 | 5281.94 | 5309.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 15:15:00 | 5353.00 | 5326.38 | 5322.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 128 — BUY (started 2026-03-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 15:15:00 | 5353.00 | 5326.38 | 5322.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 5420.50 | 5345.21 | 5331.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 5297.50 | 5392.29 | 5373.34 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 5297.50 | 5392.29 | 5373.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 5297.50 | 5392.29 | 5373.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 5297.50 | 5392.29 | 5373.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 5280.50 | 5353.32 | 5357.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 5190.00 | 5306.29 | 5334.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 5283.00 | 5266.83 | 5306.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 09:45:00 | 5295.50 | 5266.83 | 5306.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 5350.50 | 5283.56 | 5310.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 10:45:00 | 5337.50 | 5283.56 | 5310.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 5293.00 | 5285.45 | 5308.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 12:15:00 | 5272.50 | 5285.45 | 5308.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 13:30:00 | 5267.50 | 5280.15 | 5302.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 13:45:00 | 5286.50 | 5192.72 | 5201.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 14:30:00 | 5282.00 | 5203.28 | 5205.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 15:15:00 | 5258.00 | 5214.22 | 5210.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 5258.00 | 5214.22 | 5210.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 5332.50 | 5237.88 | 5221.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 5289.50 | 5293.02 | 5259.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-25 14:45:00 | 5299.00 | 5293.02 | 5259.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 5158.00 | 5263.61 | 5251.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:45:00 | 5167.00 | 5263.61 | 5251.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 5154.50 | 5241.79 | 5243.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 11:15:00 | 5140.00 | 5221.43 | 5233.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 5169.50 | 5116.69 | 5151.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 5169.50 | 5116.69 | 5151.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 5169.50 | 5116.69 | 5151.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 11:00:00 | 5136.50 | 5120.65 | 5150.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 13:30:00 | 5125.00 | 5137.62 | 5151.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 5335.50 | 5097.26 | 5075.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

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
| ALERT3_SIDEWAYS | 2026-04-17 09:30:00 | 5199.00 | 5198.86 | 5233.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 5227.00 | 5204.49 | 5232.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 12:15:00 | 5216.50 | 5208.29 | 5232.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 13:00:00 | 5216.00 | 5209.83 | 5230.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-17 14:00:00 | 5215.50 | 5210.97 | 5229.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 09:15:00 | 5276.50 | 5230.32 | 5234.10 | SL hit (close>static) qty=1.00 sl=5250.00 alert=retest2 |

### Cycle 136 — BUY (started 2026-04-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-20 10:15:00 | 5281.50 | 5240.56 | 5238.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-20 11:15:00 | 5295.00 | 5251.45 | 5243.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 15:15:00 | 5254.50 | 5265.64 | 5254.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 15:15:00 | 5254.50 | 5265.64 | 5254.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 5254.50 | 5265.64 | 5254.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 10:00:00 | 5295.00 | 5271.51 | 5257.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 11:00:00 | 5295.00 | 5276.21 | 5261.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 11:45:00 | 5297.00 | 5281.37 | 5264.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 12:15:00 | 5221.00 | 5260.96 | 5262.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2026-04-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 12:15:00 | 5221.00 | 5260.96 | 5262.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 13:15:00 | 5215.00 | 5251.77 | 5257.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 5071.00 | 5015.24 | 5068.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 5071.00 | 5015.24 | 5068.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 5071.00 | 5015.24 | 5068.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 5071.00 | 5015.24 | 5068.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 5062.00 | 5024.59 | 5068.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 13:30:00 | 5052.00 | 5046.29 | 5068.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-27 14:00:00 | 5054.50 | 5046.29 | 5068.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:45:00 | 5035.50 | 5052.18 | 5063.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 13:30:00 | 5055.50 | 5050.85 | 5060.81 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 5057.50 | 5052.18 | 5060.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 5057.50 | 5052.18 | 5060.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 5083.00 | 5058.35 | 5062.55 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-28 15:15:00 | 5083.00 | 5058.35 | 5062.55 | SL hit (close>static) qty=1.00 sl=5082.50 alert=retest2 |

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
| ALERT3_SIDEWAYS | 2026-04-30 14:00:00 | 5073.50 | 5069.72 | 5078.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 5105.50 | 5076.88 | 5080.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:45:00 | 5112.00 | 5076.88 | 5080.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — BUY (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 15:15:00 | 5110.00 | 5083.50 | 5083.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 5148.50 | 5096.50 | 5089.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 11:15:00 | 5086.50 | 5098.50 | 5091.80 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 11:15:00 | 5086.50 | 5098.50 | 5091.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 5086.50 | 5098.50 | 5091.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-04 12:00:00 | 5086.50 | 5098.50 | 5091.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

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
| ALERT3_SIDEWAYS | 2026-05-06 11:00:00 | 4985.00 | 5094.07 | 5088.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 11:15:00 | 5080.00 | 5091.26 | 5087.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 12:15:00 | 5091.00 | 5091.26 | 5087.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 13:00:00 | 5089.50 | 5090.91 | 5087.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 14:00:00 | 5092.00 | 5091.13 | 5087.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |


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
| SELL | retest2 | 2025-01-10 09:30:00 | 4121.45 | 2025-01-15 10:15:00 | 4119.00 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-01-10 11:30:00 | 4128.95 | 2025-01-15 10:15:00 | 4119.00 | STOP_HIT | 1.00 | 0.24% |
| SELL | retest2 | 2025-01-10 14:00:00 | 4121.45 | 2025-01-15 10:15:00 | 4119.00 | STOP_HIT | 1.00 | 0.06% |
| SELL | retest2 | 2025-01-15 10:00:00 | 4117.65 | 2025-01-15 10:15:00 | 4119.00 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-01-17 13:15:00 | 4099.10 | 2025-01-23 10:15:00 | 4110.35 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-01-17 15:00:00 | 4099.75 | 2025-01-23 10:15:00 | 4110.35 | STOP_HIT | 1.00 | -0.26% |
| BUY | retest1 | 2025-02-01 12:15:00 | 4456.85 | 2025-02-03 10:15:00 | 4289.05 | STOP_HIT | 1.00 | -3.76% |
| BUY | retest2 | 2025-02-04 09:15:00 | 4309.85 | 2025-02-04 10:15:00 | 4236.75 | STOP_HIT | 1.00 | -1.70% |
| SELL | retest2 | 2025-02-07 14:30:00 | 4244.00 | 2025-02-07 15:15:00 | 4282.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-02-19 14:15:00 | 3865.35 | 2025-02-20 09:15:00 | 3913.60 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-02-25 14:00:00 | 3850.10 | 2025-03-03 09:15:00 | 3657.59 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 14:00:00 | 3850.10 | 2025-03-05 09:15:00 | 3563.00 | STOP_HIT | 0.50 | 7.46% |
| SELL | retest2 | 2025-03-11 12:45:00 | 3595.00 | 2025-03-11 13:15:00 | 3626.70 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-03-24 12:15:00 | 3626.70 | 2025-04-04 09:15:00 | 3635.35 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest2 | 2025-03-26 09:45:00 | 3615.15 | 2025-04-04 09:15:00 | 3635.35 | STOP_HIT | 1.00 | 0.56% |
| BUY | retest2 | 2025-03-26 10:30:00 | 3619.90 | 2025-04-04 09:15:00 | 3635.35 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2025-04-09 09:15:00 | 3562.85 | 2025-04-09 10:15:00 | 3597.90 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-04-21 09:30:00 | 3815.00 | 2025-04-28 10:15:00 | 3881.00 | STOP_HIT | 1.00 | 1.73% |
| SELL | retest2 | 2025-05-02 11:45:00 | 3831.20 | 2025-05-06 11:15:00 | 3854.20 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-05-06 11:15:00 | 3833.50 | 2025-05-06 11:15:00 | 3854.20 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2025-05-22 14:15:00 | 4253.00 | 2025-05-23 09:15:00 | 4319.20 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-05-28 13:30:00 | 4354.00 | 2025-05-30 14:15:00 | 4302.80 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest2 | 2025-05-28 15:15:00 | 4375.00 | 2025-05-30 14:15:00 | 4302.80 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-05-29 09:45:00 | 4364.00 | 2025-05-30 14:15:00 | 4302.80 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2025-05-29 10:45:00 | 4355.80 | 2025-05-30 14:15:00 | 4302.80 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-06-19 09:15:00 | 4373.20 | 2025-06-20 09:15:00 | 4282.40 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2025-06-19 13:00:00 | 4370.10 | 2025-06-20 09:15:00 | 4282.40 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-07-21 11:45:00 | 4418.00 | 2025-07-22 10:15:00 | 4383.50 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-07-21 12:30:00 | 4418.50 | 2025-07-22 10:15:00 | 4383.50 | STOP_HIT | 1.00 | -0.79% |
| BUY | retest2 | 2025-07-22 09:45:00 | 4414.10 | 2025-07-22 10:15:00 | 4383.50 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-07-24 09:15:00 | 4296.10 | 2025-07-29 10:15:00 | 4321.50 | STOP_HIT | 1.00 | -0.59% |
| BUY | retest2 | 2025-08-07 12:45:00 | 4607.60 | 2025-08-18 09:15:00 | 5068.36 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-12 09:15:00 | 4604.00 | 2025-08-18 09:15:00 | 5064.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-08-28 09:15:00 | 5151.00 | 2025-09-10 11:15:00 | 5348.50 | STOP_HIT | 1.00 | 3.83% |
| BUY | retest2 | 2025-08-29 09:45:00 | 5102.90 | 2025-09-10 11:15:00 | 5348.50 | STOP_HIT | 1.00 | 4.81% |
| SELL | retest2 | 2025-09-12 13:30:00 | 5293.00 | 2025-09-16 13:15:00 | 5333.00 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2025-09-15 11:30:00 | 5291.00 | 2025-09-16 13:15:00 | 5333.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-09-16 09:30:00 | 5293.00 | 2025-09-16 13:15:00 | 5333.00 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-09-18 15:00:00 | 5375.50 | 2025-09-23 14:15:00 | 5374.00 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-10-03 11:00:00 | 5526.00 | 2025-10-08 15:15:00 | 5520.00 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-10-10 14:30:00 | 5498.00 | 2025-10-13 09:15:00 | 5552.50 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest1 | 2025-10-15 09:15:00 | 5594.50 | 2025-10-15 13:15:00 | 5518.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest1 | 2025-10-21 13:45:00 | 5650.00 | 2025-10-23 14:15:00 | 5571.50 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2025-11-14 09:15:00 | 5549.00 | 2025-11-26 09:15:00 | 6103.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-11-14 10:30:00 | 5603.00 | 2025-11-26 09:15:00 | 6091.25 | TARGET_HIT | 1.00 | 8.71% |
| BUY | retest2 | 2025-11-14 15:00:00 | 5537.50 | 2025-11-27 09:15:00 | 6163.30 | TARGET_HIT | 1.00 | 11.30% |
| SELL | retest2 | 2025-12-12 11:45:00 | 5975.50 | 2025-12-18 09:15:00 | 5676.72 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 12:45:00 | 5974.50 | 2025-12-18 09:15:00 | 5675.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-12 11:45:00 | 5975.50 | 2025-12-19 10:15:00 | 5768.50 | STOP_HIT | 0.50 | 3.46% |
| SELL | retest2 | 2025-12-12 12:45:00 | 5974.50 | 2025-12-19 10:15:00 | 5768.50 | STOP_HIT | 0.50 | 3.45% |
| SELL | retest2 | 2026-01-14 09:45:00 | 5698.00 | 2026-01-19 12:15:00 | 5729.00 | STOP_HIT | 1.00 | -0.54% |
| SELL | retest2 | 2026-01-14 12:00:00 | 5700.00 | 2026-01-19 13:15:00 | 5744.50 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2026-01-16 11:45:00 | 5701.00 | 2026-01-19 13:15:00 | 5744.50 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2026-01-16 12:15:00 | 5700.00 | 2026-01-19 13:15:00 | 5744.50 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2026-01-16 13:15:00 | 5650.50 | 2026-01-19 13:15:00 | 5744.50 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-01-22 11:45:00 | 5566.50 | 2026-01-28 15:15:00 | 5510.00 | STOP_HIT | 1.00 | 1.02% |
| SELL | retest2 | 2026-01-22 12:30:00 | 5569.00 | 2026-01-28 15:15:00 | 5510.00 | STOP_HIT | 1.00 | 1.06% |
| BUY | retest2 | 2026-01-30 14:45:00 | 5536.00 | 2026-02-02 10:15:00 | 5430.50 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2026-02-01 12:30:00 | 5542.00 | 2026-02-02 10:15:00 | 5430.50 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2026-02-02 09:15:00 | 5531.00 | 2026-02-02 10:15:00 | 5430.50 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2026-02-23 11:30:00 | 5465.50 | 2026-02-24 10:15:00 | 5505.50 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2026-03-05 11:15:00 | 5497.00 | 2026-03-05 14:15:00 | 5571.50 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-03-06 14:45:00 | 5497.50 | 2026-03-10 10:15:00 | 5685.00 | STOP_HIT | 1.00 | -3.41% |
| SELL | retest2 | 2026-03-09 09:15:00 | 5351.50 | 2026-03-10 10:15:00 | 5685.00 | STOP_HIT | 1.00 | -6.23% |
| SELL | retest2 | 2026-03-17 11:15:00 | 5296.00 | 2026-03-17 15:15:00 | 5353.00 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2026-03-20 12:15:00 | 5272.50 | 2026-03-24 15:15:00 | 5258.00 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest2 | 2026-03-20 13:30:00 | 5267.50 | 2026-03-24 15:15:00 | 5258.00 | STOP_HIT | 1.00 | 0.18% |
| SELL | retest2 | 2026-03-24 13:45:00 | 5286.50 | 2026-03-24 15:15:00 | 5258.00 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2026-03-24 14:30:00 | 5282.00 | 2026-03-24 15:15:00 | 5258.00 | STOP_HIT | 1.00 | 0.45% |
| SELL | retest2 | 2026-04-01 11:00:00 | 5136.50 | 2026-04-08 09:15:00 | 5335.50 | STOP_HIT | 1.00 | -3.87% |
| SELL | retest2 | 2026-04-01 13:30:00 | 5125.00 | 2026-04-08 09:15:00 | 5335.50 | STOP_HIT | 1.00 | -4.11% |
| SELL | retest2 | 2026-04-17 12:15:00 | 5216.50 | 2026-04-20 09:15:00 | 5276.50 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2026-04-17 13:00:00 | 5216.00 | 2026-04-20 09:15:00 | 5276.50 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2026-04-17 14:00:00 | 5215.50 | 2026-04-20 09:15:00 | 5276.50 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2026-04-21 10:00:00 | 5295.00 | 2026-04-22 12:15:00 | 5221.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2026-04-21 11:00:00 | 5295.00 | 2026-04-22 12:15:00 | 5221.00 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2026-04-21 11:45:00 | 5297.00 | 2026-04-22 12:15:00 | 5221.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2026-04-27 13:30:00 | 5052.00 | 2026-04-28 15:15:00 | 5083.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2026-04-27 14:00:00 | 5054.50 | 2026-04-28 15:15:00 | 5083.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2026-04-28 11:45:00 | 5035.50 | 2026-04-28 15:15:00 | 5083.00 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2026-04-28 13:30:00 | 5055.50 | 2026-04-28 15:15:00 | 5083.00 | STOP_HIT | 1.00 | -0.54% |
