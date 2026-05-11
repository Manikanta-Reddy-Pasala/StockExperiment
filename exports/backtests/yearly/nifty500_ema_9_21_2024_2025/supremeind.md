# Supreme Industries Ltd. (SUPREMEIND)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 3654.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 159 |
| ALERT1 | 94 |
| ALERT2 | 93 |
| ALERT2_SKIP | 52 |
| ALERT3 | 252 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 112 |
| PARTIAL | 21 |
| TARGET_HIT | 9 |
| STOP_HIT | 106 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 136 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 56 / 80
- **Target hits / Stop hits / Partials:** 9 / 106 / 21
- **Avg / median % per leg:** 1.01% / -0.80%
- **Sum % (uncompounded):** 137.16%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 35 | 7 | 20.0% | 3 | 32 | 0 | -0.39% | -13.7% |
| BUY @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.41% | 1.4% |
| BUY @ 3rd Alert (retest2) | 34 | 6 | 17.6% | 3 | 31 | 0 | -0.45% | -15.2% |
| SELL (all) | 101 | 49 | 48.5% | 6 | 74 | 21 | 1.49% | 150.9% |
| SELL @ 2nd Alert (retest1) | 2 | 0 | 0.0% | 0 | 2 | 0 | -1.09% | -2.2% |
| SELL @ 3rd Alert (retest2) | 99 | 49 | 49.5% | 6 | 72 | 21 | 1.55% | 153.1% |
| retest1 (combined) | 3 | 1 | 33.3% | 0 | 3 | 0 | -0.26% | -0.8% |
| retest2 (combined) | 133 | 55 | 41.4% | 9 | 103 | 21 | 1.04% | 137.9% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-13 09:15:00 | 5176.50 | 5255.08 | 5255.10 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2024-05-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 13:15:00 | 5283.70 | 5251.34 | 5251.06 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 11:15:00 | 5243.50 | 5252.38 | 5252.59 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2024-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 12:15:00 | 5266.10 | 5255.12 | 5253.82 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2024-05-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 14:15:00 | 5181.80 | 5239.57 | 5246.92 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2024-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-15 10:15:00 | 5287.25 | 5251.98 | 5250.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 13:15:00 | 5330.65 | 5281.12 | 5265.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-21 09:15:00 | 5514.90 | 5554.44 | 5495.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 09:15:00 | 5514.90 | 5554.44 | 5495.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 5514.90 | 5554.44 | 5495.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:45:00 | 5439.40 | 5554.44 | 5495.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 09:15:00 | 5543.75 | 5549.50 | 5520.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:30:00 | 5532.90 | 5549.50 | 5520.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 10:15:00 | 5515.60 | 5542.72 | 5519.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:00:00 | 5515.60 | 5542.72 | 5519.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-22 11:15:00 | 5475.80 | 5529.33 | 5515.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-22 11:30:00 | 5468.00 | 5529.33 | 5515.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2024-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-22 14:15:00 | 5461.95 | 5503.07 | 5506.00 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2024-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-23 09:15:00 | 5568.00 | 5509.49 | 5508.00 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-05-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-24 09:15:00 | 5468.75 | 5512.38 | 5515.05 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2024-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-27 09:15:00 | 5724.75 | 5536.01 | 5521.44 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2024-05-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 10:15:00 | 5502.45 | 5578.48 | 5580.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 15:15:00 | 5480.15 | 5503.46 | 5525.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 5520.00 | 5404.25 | 5451.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 5520.00 | 5404.25 | 5451.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 5520.00 | 5404.25 | 5451.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 10:00:00 | 5520.00 | 5404.25 | 5451.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 10:15:00 | 5458.50 | 5415.10 | 5452.15 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2024-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 13:15:00 | 5565.70 | 5480.71 | 5476.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-04 09:15:00 | 5589.35 | 5525.09 | 5499.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-04 10:15:00 | 5398.55 | 5499.79 | 5490.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 10:15:00 | 5398.55 | 5499.79 | 5490.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 5398.55 | 5499.79 | 5490.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:00:00 | 5398.55 | 5499.79 | 5490.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 11:15:00 | 5330.00 | 5465.83 | 5475.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 14:15:00 | 5289.00 | 5404.83 | 5442.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 09:15:00 | 5498.65 | 5405.02 | 5435.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-05 09:15:00 | 5498.65 | 5405.02 | 5435.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-05 09:15:00 | 5498.65 | 5405.02 | 5435.30 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2024-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-05 11:15:00 | 5532.00 | 5459.70 | 5456.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 09:15:00 | 5694.00 | 5533.35 | 5495.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 15:15:00 | 5691.05 | 5692.28 | 5608.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-07 09:15:00 | 5693.05 | 5692.28 | 5608.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 5640.65 | 5681.96 | 5611.84 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 15:00:00 | 5763.00 | 5664.15 | 5624.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-06-12 09:15:00 | 6339.30 | 6161.06 | 6013.08 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2024-06-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 09:15:00 | 6054.80 | 6092.34 | 6096.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-14 10:15:00 | 5973.00 | 6068.47 | 6085.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-14 14:15:00 | 6050.00 | 6027.59 | 6056.99 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-14 14:15:00 | 6050.00 | 6027.59 | 6056.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-14 14:15:00 | 6050.00 | 6027.59 | 6056.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-14 15:00:00 | 6050.00 | 6027.59 | 6056.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 6197.70 | 6063.60 | 6068.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 10:00:00 | 6197.70 | 6063.60 | 6068.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2024-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-18 10:15:00 | 6245.85 | 6100.05 | 6084.52 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-06-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 09:15:00 | 5996.55 | 6123.41 | 6133.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 10:15:00 | 5938.10 | 6086.35 | 6115.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 11:15:00 | 5998.75 | 5960.18 | 6016.40 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-24 11:15:00 | 5998.75 | 5960.18 | 6016.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 5998.75 | 5960.18 | 6016.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:45:00 | 5977.30 | 5960.18 | 6016.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-25 09:15:00 | 5966.10 | 5957.34 | 5993.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-25 12:45:00 | 5945.80 | 5953.12 | 5982.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 10:30:00 | 5887.15 | 5845.62 | 5879.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 14:15:00 | 5950.10 | 5878.67 | 5871.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2024-06-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-28 14:15:00 | 5950.10 | 5878.67 | 5871.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-01 09:15:00 | 5989.15 | 5906.42 | 5885.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-01 14:15:00 | 5956.15 | 5958.59 | 5923.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-01 15:00:00 | 5956.15 | 5958.59 | 5923.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 12:15:00 | 5882.70 | 5946.84 | 5932.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-02 12:45:00 | 5868.55 | 5946.84 | 5932.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-02 13:15:00 | 5895.75 | 5936.62 | 5928.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-02 14:15:00 | 5970.00 | 5936.62 | 5928.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-08 10:15:00 | 5939.40 | 6012.04 | 6021.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2024-07-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-08 10:15:00 | 5939.40 | 6012.04 | 6021.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-08 11:15:00 | 5906.85 | 5991.00 | 6011.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 09:15:00 | 6029.90 | 5951.23 | 5978.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-09 09:15:00 | 6029.90 | 5951.23 | 5978.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 09:15:00 | 6029.90 | 5951.23 | 5978.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 09:45:00 | 6035.25 | 5951.23 | 5978.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 10:15:00 | 5953.75 | 5951.73 | 5975.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 10:30:00 | 6013.70 | 5951.73 | 5975.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 12:15:00 | 6002.00 | 5961.65 | 5976.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-09 13:00:00 | 6002.00 | 5961.65 | 5976.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-09 13:15:00 | 5984.55 | 5966.23 | 5976.99 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2024-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-09 15:15:00 | 6012.00 | 5987.53 | 5985.56 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2024-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-10 09:15:00 | 5931.35 | 5976.30 | 5980.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-11 09:15:00 | 5802.75 | 5917.52 | 5947.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-12 09:15:00 | 5897.60 | 5863.35 | 5897.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 09:15:00 | 5897.60 | 5863.35 | 5897.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 5897.60 | 5863.35 | 5897.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-12 12:30:00 | 5795.60 | 5854.17 | 5885.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 12:00:00 | 5820.05 | 5830.01 | 5855.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 13:15:00 | 5819.45 | 5831.93 | 5854.18 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-15 14:00:00 | 5824.05 | 5830.36 | 5851.44 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 14:15:00 | 5845.35 | 5833.36 | 5850.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-15 15:00:00 | 5845.35 | 5833.36 | 5850.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-15 15:15:00 | 5847.70 | 5836.23 | 5850.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 09:15:00 | 5833.75 | 5836.23 | 5850.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 5860.00 | 5840.98 | 5851.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 12:45:00 | 5814.00 | 5843.97 | 5850.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 13:15:00 | 5815.95 | 5843.97 | 5850.85 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-16 13:45:00 | 5793.40 | 5835.29 | 5846.28 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-19 09:30:00 | 5775.70 | 5737.30 | 5772.79 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 10:15:00 | 5743.95 | 5738.63 | 5770.17 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 09:15:00 | 5720.00 | 5753.55 | 5767.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-22 10:45:00 | 5719.95 | 5741.21 | 5759.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 5505.82 | 5621.00 | 5679.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 5529.05 | 5621.00 | 5679.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 5528.48 | 5621.00 | 5679.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 5532.85 | 5621.00 | 5679.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 5523.30 | 5621.00 | 5679.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 5525.15 | 5621.00 | 5679.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 5503.73 | 5621.00 | 5679.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-23 12:15:00 | 5486.91 | 5621.00 | 5679.51 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-24 09:15:00 | 5434.00 | 5536.21 | 5617.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-07-24 09:15:00 | 5433.95 | 5536.21 | 5617.44 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-07-25 09:15:00 | 5238.05 | 5393.94 | 5496.53 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 22 — BUY (started 2024-07-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 13:15:00 | 5523.25 | 5448.55 | 5443.77 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2024-07-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-26 15:15:00 | 5419.40 | 5437.97 | 5439.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-29 10:15:00 | 5343.85 | 5413.44 | 5427.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-30 13:15:00 | 5365.00 | 5361.01 | 5383.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-30 14:00:00 | 5365.00 | 5361.01 | 5383.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-31 09:15:00 | 5369.60 | 5352.12 | 5372.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 12:30:00 | 5345.00 | 5355.46 | 5369.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-31 15:15:00 | 5350.00 | 5357.90 | 5368.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 12:15:00 | 5348.30 | 5366.58 | 5369.71 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 13:45:00 | 5337.00 | 5361.16 | 5366.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 5077.75 | 5183.29 | 5249.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 5082.50 | 5183.29 | 5249.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 5080.89 | 5183.29 | 5249.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-08-05 09:15:00 | 5070.15 | 5183.29 | 5249.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-08-06 09:15:00 | 5036.80 | 5035.81 | 5125.93 | SL hit (close>ema200) qty=0.50 sl=5035.81 alert=retest2 |

### Cycle 24 — BUY (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 11:15:00 | 5283.80 | 5126.39 | 5120.90 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2024-08-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 14:15:00 | 5076.50 | 5134.40 | 5142.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-09 10:15:00 | 5064.00 | 5104.67 | 5125.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-09 12:15:00 | 5132.95 | 5109.99 | 5124.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-09 12:15:00 | 5132.95 | 5109.99 | 5124.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 12:15:00 | 5132.95 | 5109.99 | 5124.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 12:45:00 | 5138.70 | 5109.99 | 5124.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 13:15:00 | 5129.00 | 5113.79 | 5124.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 14:15:00 | 5115.00 | 5113.79 | 5124.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 14:15:00 | 5132.80 | 5117.59 | 5125.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-09 15:00:00 | 5132.80 | 5117.59 | 5125.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 5110.00 | 5116.07 | 5123.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 5084.05 | 5116.07 | 5123.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-12 12:15:00 | 5149.80 | 5104.30 | 5113.40 | SL hit (close>static) qty=1.00 sl=5140.00 alert=retest2 |

### Cycle 26 — BUY (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 11:15:00 | 5154.00 | 5119.54 | 5116.63 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2024-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 15:15:00 | 5095.90 | 5115.09 | 5116.26 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2024-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-14 10:15:00 | 5138.00 | 5119.77 | 5118.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-14 11:15:00 | 5173.40 | 5130.50 | 5123.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 13:15:00 | 5506.50 | 5563.47 | 5493.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 14:00:00 | 5506.50 | 5563.47 | 5493.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 14:15:00 | 5529.15 | 5556.61 | 5496.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 09:15:00 | 5577.00 | 5551.49 | 5500.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-23 13:00:00 | 5543.55 | 5554.64 | 5518.21 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-26 09:15:00 | 5442.00 | 5512.36 | 5508.55 | SL hit (close<static) qty=1.00 sl=5485.00 alert=retest2 |

### Cycle 29 — SELL (started 2024-08-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-26 10:15:00 | 5438.85 | 5497.66 | 5502.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-26 11:15:00 | 5413.55 | 5480.84 | 5494.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 15:15:00 | 5448.00 | 5446.67 | 5470.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-27 09:15:00 | 5396.60 | 5446.67 | 5470.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 5355.95 | 5378.27 | 5415.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 09:15:00 | 5301.50 | 5372.25 | 5388.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-30 15:00:00 | 5267.40 | 5301.01 | 5338.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-02 10:15:00 | 5301.00 | 5300.65 | 5331.20 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 13:00:00 | 5300.00 | 5294.19 | 5303.89 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-03 13:15:00 | 5283.15 | 5291.98 | 5302.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-03 14:30:00 | 5267.75 | 5276.37 | 5294.00 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 10:15:00 | 5410.00 | 5278.83 | 5262.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2024-09-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 10:15:00 | 5410.00 | 5278.83 | 5262.36 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2024-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-11 15:15:00 | 5294.65 | 5346.64 | 5348.08 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2024-09-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-13 13:15:00 | 5339.55 | 5337.54 | 5337.54 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2024-09-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-13 14:15:00 | 5256.90 | 5321.41 | 5330.21 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2024-09-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-16 10:15:00 | 5371.25 | 5333.37 | 5333.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-16 12:15:00 | 5400.00 | 5351.75 | 5341.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-18 09:15:00 | 5387.70 | 5391.07 | 5376.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-18 09:15:00 | 5387.70 | 5391.07 | 5376.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 09:15:00 | 5387.70 | 5391.07 | 5376.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 09:30:00 | 5388.00 | 5391.07 | 5376.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 11:15:00 | 5360.00 | 5385.41 | 5376.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 12:00:00 | 5360.00 | 5385.41 | 5376.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 12:15:00 | 5369.10 | 5382.15 | 5375.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 12:30:00 | 5365.85 | 5382.15 | 5375.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 13:15:00 | 5369.65 | 5379.65 | 5375.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-18 13:45:00 | 5363.10 | 5379.65 | 5375.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2024-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-18 15:15:00 | 5346.00 | 5370.86 | 5371.95 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2024-09-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 09:15:00 | 5381.35 | 5372.96 | 5372.81 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-19 10:15:00 | 5338.75 | 5366.11 | 5369.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-19 11:15:00 | 5304.15 | 5353.72 | 5363.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-19 14:15:00 | 5399.00 | 5348.54 | 5357.60 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-19 14:15:00 | 5399.00 | 5348.54 | 5357.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-19 14:15:00 | 5399.00 | 5348.54 | 5357.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-19 15:00:00 | 5399.00 | 5348.54 | 5357.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2024-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-19 15:15:00 | 5471.85 | 5373.20 | 5367.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-20 14:15:00 | 5575.00 | 5430.93 | 5399.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-23 09:15:00 | 5448.60 | 5457.13 | 5418.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-23 10:00:00 | 5448.60 | 5457.13 | 5418.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-23 12:15:00 | 5443.10 | 5450.64 | 5424.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-23 12:45:00 | 5437.20 | 5450.64 | 5424.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 09:15:00 | 5400.55 | 5438.53 | 5427.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 09:30:00 | 5405.00 | 5438.53 | 5427.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-24 10:15:00 | 5399.80 | 5430.79 | 5424.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-24 10:45:00 | 5395.05 | 5430.79 | 5424.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2024-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 12:15:00 | 5397.75 | 5417.34 | 5419.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 09:15:00 | 5287.00 | 5382.02 | 5401.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 15:15:00 | 5286.95 | 5276.33 | 5305.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-27 09:15:00 | 5272.65 | 5276.33 | 5305.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 5268.55 | 5274.78 | 5301.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-30 09:30:00 | 5215.00 | 5268.63 | 5285.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-30 11:15:00 | 5369.60 | 5285.93 | 5290.25 | SL hit (close>static) qty=1.00 sl=5306.65 alert=retest2 |

### Cycle 40 — BUY (started 2024-09-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 12:15:00 | 5350.50 | 5298.84 | 5295.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-01 09:15:00 | 5410.75 | 5337.66 | 5316.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-03 11:15:00 | 5365.20 | 5382.90 | 5359.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-03 11:15:00 | 5365.20 | 5382.90 | 5359.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 11:15:00 | 5365.20 | 5382.90 | 5359.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 12:00:00 | 5365.20 | 5382.90 | 5359.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 12:15:00 | 5234.85 | 5353.29 | 5348.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:00:00 | 5234.85 | 5353.29 | 5348.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2024-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 13:15:00 | 5233.40 | 5329.31 | 5337.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 14:15:00 | 5141.20 | 5291.69 | 5320.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-04 09:15:00 | 5290.30 | 5279.94 | 5309.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-04 10:00:00 | 5290.30 | 5279.94 | 5309.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 10:15:00 | 5292.75 | 5282.50 | 5307.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 10:45:00 | 5307.70 | 5282.50 | 5307.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 14:15:00 | 5291.45 | 5279.90 | 5298.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-04 15:00:00 | 5291.45 | 5279.90 | 5298.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-04 15:15:00 | 5266.25 | 5277.17 | 5295.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-07 09:15:00 | 5290.00 | 5277.17 | 5295.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-07 09:15:00 | 5283.50 | 5278.44 | 5294.14 | EMA400 retest candle locked (from downside) |

### Cycle 42 — BUY (started 2024-10-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-07 14:15:00 | 5422.15 | 5314.79 | 5305.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-07 15:15:00 | 5490.00 | 5349.83 | 5322.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-08 10:15:00 | 5351.05 | 5352.17 | 5328.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-08 11:00:00 | 5351.05 | 5352.17 | 5328.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 13:15:00 | 5329.95 | 5344.77 | 5330.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-08 13:45:00 | 5325.00 | 5344.77 | 5330.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 5455.30 | 5366.88 | 5341.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-08 14:45:00 | 5332.10 | 5366.88 | 5341.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 5431.30 | 5436.68 | 5404.69 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2024-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 14:15:00 | 5299.40 | 5385.81 | 5390.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-11 09:15:00 | 5276.00 | 5350.12 | 5372.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-14 12:15:00 | 5290.00 | 5238.84 | 5280.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-14 12:15:00 | 5290.00 | 5238.84 | 5280.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 12:15:00 | 5290.00 | 5238.84 | 5280.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 12:45:00 | 5294.70 | 5238.84 | 5280.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 13:15:00 | 5291.15 | 5249.30 | 5281.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 14:15:00 | 5324.05 | 5249.30 | 5281.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 14:15:00 | 5282.60 | 5255.96 | 5281.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-14 14:30:00 | 5329.80 | 5255.96 | 5281.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 15:15:00 | 5265.00 | 5257.77 | 5280.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 09:15:00 | 5250.60 | 5257.77 | 5280.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 09:15:00 | 5245.05 | 5255.23 | 5277.20 | EMA400 retest candle locked (from downside) |

### Cycle 44 — BUY (started 2024-10-15 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 12:15:00 | 5344.05 | 5296.31 | 5292.26 | EMA200 above EMA400 |

### Cycle 45 — SELL (started 2024-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 11:15:00 | 5249.70 | 5288.73 | 5293.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 09:15:00 | 5188.10 | 5244.98 | 5268.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-17 13:15:00 | 5227.35 | 5215.96 | 5245.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-17 13:15:00 | 5227.35 | 5215.96 | 5245.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 13:15:00 | 5227.35 | 5215.96 | 5245.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-17 13:30:00 | 5249.90 | 5215.96 | 5245.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 14:15:00 | 5255.00 | 5223.77 | 5246.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 09:15:00 | 5177.00 | 5228.82 | 5246.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-18 11:15:00 | 5200.20 | 5222.90 | 5240.29 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-18 15:15:00 | 4940.19 | 5168.13 | 5206.27 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 10:15:00 | 4918.15 | 5002.89 | 5083.24 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-22 13:15:00 | 4659.30 | 4874.23 | 5001.74 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 46 — BUY (started 2024-11-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 12:15:00 | 4352.75 | 4306.67 | 4305.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-05 13:15:00 | 4380.10 | 4321.35 | 4311.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 15:15:00 | 4630.00 | 4661.26 | 4581.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-08 09:15:00 | 4621.00 | 4661.26 | 4581.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 4622.25 | 4653.46 | 4585.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 10:30:00 | 4641.65 | 4638.11 | 4605.97 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 13:45:00 | 4641.20 | 4643.67 | 4617.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-11 14:30:00 | 4653.45 | 4643.74 | 4619.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-12 10:30:00 | 4646.00 | 4640.87 | 4624.25 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 11:15:00 | 4625.35 | 4637.77 | 4624.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 12:00:00 | 4625.35 | 4637.77 | 4624.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 12:15:00 | 4610.50 | 4632.31 | 4623.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 12:30:00 | 4612.95 | 4632.31 | 4623.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 13:15:00 | 4614.20 | 4628.69 | 4622.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-12 13:30:00 | 4608.70 | 4628.69 | 4622.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-12 15:15:00 | 4605.00 | 4621.71 | 4620.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-13 09:15:00 | 4602.30 | 4621.71 | 4620.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-11-13 09:15:00 | 4519.35 | 4601.24 | 4610.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2024-11-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 09:15:00 | 4519.35 | 4601.24 | 4610.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 10:15:00 | 4500.90 | 4581.17 | 4600.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 4631.15 | 4560.05 | 4577.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 4631.15 | 4560.05 | 4577.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 4631.15 | 4560.05 | 4577.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:00:00 | 4631.15 | 4560.05 | 4577.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 10:15:00 | 4611.60 | 4570.36 | 4580.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-14 10:30:00 | 4641.75 | 4570.36 | 4580.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2024-11-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-14 11:15:00 | 4662.65 | 4588.82 | 4587.76 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2024-11-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-14 14:15:00 | 4535.20 | 4587.79 | 4588.71 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2024-11-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 13:15:00 | 4625.05 | 4585.29 | 4584.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 10:15:00 | 4653.85 | 4611.36 | 4598.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-21 09:15:00 | 4566.55 | 4624.83 | 4614.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-21 09:15:00 | 4566.55 | 4624.83 | 4614.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 4566.55 | 4624.83 | 4614.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:30:00 | 4552.55 | 4624.83 | 4614.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 10:15:00 | 4550.05 | 4609.87 | 4608.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 10:45:00 | 4553.95 | 4609.87 | 4608.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — SELL (started 2024-11-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 11:15:00 | 4563.20 | 4600.54 | 4604.58 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-11-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 11:15:00 | 4669.55 | 4581.37 | 4579.12 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-11-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-26 15:15:00 | 4560.00 | 4597.71 | 4599.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-27 09:15:00 | 4554.35 | 4589.03 | 4595.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-27 10:15:00 | 4599.85 | 4591.20 | 4595.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 10:15:00 | 4599.85 | 4591.20 | 4595.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 10:15:00 | 4599.85 | 4591.20 | 4595.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 11:00:00 | 4599.85 | 4591.20 | 4595.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 11:15:00 | 4613.25 | 4595.61 | 4597.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 12:00:00 | 4613.25 | 4595.61 | 4597.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 12:15:00 | 4595.60 | 4595.61 | 4597.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-27 12:45:00 | 4606.40 | 4595.61 | 4597.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 4557.35 | 4587.96 | 4593.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 10:30:00 | 4539.95 | 4576.83 | 4586.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 11:15:00 | 4550.10 | 4576.83 | 4586.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 12:00:00 | 4551.75 | 4571.81 | 4583.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-28 12:30:00 | 4551.25 | 4567.97 | 4580.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 4560.00 | 4554.21 | 4568.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:00:00 | 4560.00 | 4554.21 | 4568.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 4565.40 | 4556.45 | 4568.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-29 11:00:00 | 4565.40 | 4556.45 | 4568.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 11:15:00 | 4652.15 | 4575.59 | 4576.23 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-29 11:15:00 | 4652.15 | 4575.59 | 4576.23 | SL hit (close>static) qty=1.00 sl=4603.45 alert=retest2 |

### Cycle 54 — BUY (started 2024-11-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-29 12:15:00 | 4694.70 | 4599.41 | 4587.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-29 13:15:00 | 4704.05 | 4620.34 | 4597.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-29 15:15:00 | 4605.00 | 4622.74 | 4603.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-29 15:15:00 | 4605.00 | 4622.74 | 4603.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 15:15:00 | 4605.00 | 4622.74 | 4603.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-02 09:15:00 | 4798.00 | 4622.74 | 4603.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-02 09:15:00 | 4782.60 | 4654.72 | 4619.34 | EMA400 retest candle locked (from upside) |

### Cycle 55 — SELL (started 2024-12-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-05 10:15:00 | 4671.25 | 4724.49 | 4728.60 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2024-12-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-05 14:15:00 | 4759.15 | 4731.50 | 4730.22 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2024-12-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 09:15:00 | 4698.65 | 4729.40 | 4729.76 | EMA200 below EMA400 |

### Cycle 58 — BUY (started 2024-12-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 10:15:00 | 4744.45 | 4732.41 | 4731.10 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2024-12-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 11:15:00 | 4718.65 | 4729.66 | 4729.97 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2024-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-06 12:15:00 | 4777.45 | 4739.21 | 4734.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-09 10:15:00 | 4919.65 | 4780.69 | 4755.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-10 11:15:00 | 4909.45 | 4934.80 | 4870.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-10 12:00:00 | 4909.45 | 4934.80 | 4870.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-11 14:15:00 | 4959.40 | 4935.45 | 4909.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-11 14:30:00 | 4938.50 | 4935.45 | 4909.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 09:15:00 | 4871.70 | 4924.44 | 4909.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 09:45:00 | 4885.55 | 4924.44 | 4909.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 10:15:00 | 4854.65 | 4910.48 | 4904.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 10:30:00 | 4859.95 | 4910.48 | 4904.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — SELL (started 2024-12-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 11:15:00 | 4853.60 | 4899.10 | 4899.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 15:15:00 | 4831.00 | 4867.02 | 4882.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-16 09:15:00 | 4849.30 | 4816.85 | 4838.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-16 09:15:00 | 4849.30 | 4816.85 | 4838.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 09:15:00 | 4849.30 | 4816.85 | 4838.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 10:00:00 | 4849.30 | 4816.85 | 4838.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 10:15:00 | 4878.25 | 4829.13 | 4842.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:00:00 | 4878.25 | 4829.13 | 4842.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 11:15:00 | 4912.55 | 4845.81 | 4848.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-16 11:45:00 | 4913.75 | 4845.81 | 4848.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2024-12-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 12:15:00 | 4871.25 | 4850.90 | 4850.79 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2024-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-16 13:15:00 | 4826.95 | 4846.11 | 4848.62 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2024-12-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 14:15:00 | 4905.05 | 4857.90 | 4853.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-17 09:15:00 | 4964.75 | 4886.00 | 4867.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-17 13:15:00 | 4901.45 | 4910.19 | 4887.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-17 14:00:00 | 4901.45 | 4910.19 | 4887.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-19 09:15:00 | 5053.30 | 5059.46 | 4999.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 10:30:00 | 5082.95 | 5064.55 | 5006.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-19 11:00:00 | 5084.90 | 5064.55 | 5006.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-20 09:15:00 | 4927.30 | 5025.50 | 5012.94 | SL hit (close<static) qty=1.00 sl=4966.75 alert=retest2 |

### Cycle 65 — SELL (started 2024-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 12:15:00 | 4918.30 | 4992.59 | 5000.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 14:15:00 | 4831.50 | 4946.45 | 4976.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-24 13:15:00 | 4774.55 | 4753.99 | 4816.49 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 12:00:00 | 4727.00 | 4749.53 | 4790.93 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-12-26 13:30:00 | 4735.50 | 4742.72 | 4780.56 | SELL ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 15:15:00 | 4756.00 | 4748.46 | 4776.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 09:15:00 | 4798.20 | 4748.46 | 4776.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 09:15:00 | 4782.70 | 4755.31 | 4777.33 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-12-27 09:15:00 | 4782.70 | 4755.31 | 4777.33 | SL hit (close>ema400) qty=1.00 sl=4777.33 alert=retest1 |

### Cycle 66 — BUY (started 2025-01-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-03 09:15:00 | 4727.70 | 4699.32 | 4699.08 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2025-01-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-03 15:15:00 | 4625.00 | 4690.88 | 4699.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 11:15:00 | 4567.80 | 4652.90 | 4678.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-08 10:15:00 | 4542.35 | 4531.79 | 4573.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-08 11:00:00 | 4542.35 | 4531.79 | 4573.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 13:15:00 | 4562.80 | 4537.34 | 4565.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 13:45:00 | 4567.10 | 4537.34 | 4565.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 14:15:00 | 4607.00 | 4551.27 | 4569.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-08 14:30:00 | 4651.35 | 4551.27 | 4569.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-08 15:15:00 | 4600.00 | 4561.02 | 4571.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 09:15:00 | 4605.20 | 4561.02 | 4571.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2025-01-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 11:15:00 | 4637.05 | 4583.22 | 4579.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-09 13:15:00 | 4667.85 | 4610.83 | 4593.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 09:15:00 | 4571.65 | 4620.31 | 4603.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 09:15:00 | 4571.65 | 4620.31 | 4603.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 4571.65 | 4620.31 | 4603.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:00:00 | 4571.65 | 4620.31 | 4603.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 10:15:00 | 4594.00 | 4615.05 | 4603.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 10:30:00 | 4582.90 | 4615.05 | 4603.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 12:15:00 | 4620.00 | 4613.30 | 4604.15 | EMA400 retest candle locked (from upside) |

### Cycle 69 — SELL (started 2025-01-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-10 14:15:00 | 4546.10 | 4594.95 | 4597.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-13 09:15:00 | 4454.20 | 4558.49 | 4579.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-14 10:15:00 | 4373.25 | 4363.20 | 4443.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-14 11:00:00 | 4373.25 | 4363.20 | 4443.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 10:15:00 | 4415.00 | 4377.04 | 4411.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 10:45:00 | 4389.05 | 4377.04 | 4411.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 11:15:00 | 4409.50 | 4383.53 | 4411.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 11:45:00 | 4416.85 | 4383.53 | 4411.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 12:15:00 | 4404.15 | 4387.65 | 4410.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 12:30:00 | 4402.75 | 4387.65 | 4410.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 4373.00 | 4378.56 | 4398.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 10:45:00 | 4360.00 | 4374.39 | 4394.90 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-20 09:15:00 | 4142.00 | 4226.15 | 4292.12 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-22 09:15:00 | 3924.00 | 3995.04 | 4080.28 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 70 — BUY (started 2025-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 13:15:00 | 3777.60 | 3711.99 | 3707.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 14:15:00 | 3821.00 | 3733.80 | 3718.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 13:15:00 | 3752.00 | 3796.39 | 3764.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-30 13:15:00 | 3752.00 | 3796.39 | 3764.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 3752.00 | 3796.39 | 3764.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:00:00 | 3752.00 | 3796.39 | 3764.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 3778.65 | 3792.84 | 3765.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 14:30:00 | 3770.75 | 3792.84 | 3765.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 3645.00 | 3886.99 | 3864.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 3645.00 | 3886.99 | 3864.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 3734.75 | 3856.54 | 3852.54 | EMA400 retest candle locked (from upside) |

### Cycle 71 — SELL (started 2025-02-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-01 14:15:00 | 3741.55 | 3833.55 | 3842.45 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-03 11:15:00 | 3889.95 | 3843.59 | 3842.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 09:15:00 | 3991.00 | 3884.59 | 3863.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-05 14:15:00 | 4003.90 | 4012.09 | 3969.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-05 15:00:00 | 4003.90 | 4012.09 | 3969.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 3949.55 | 3996.82 | 3975.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-06 12:00:00 | 3949.55 | 3996.82 | 3975.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 12:15:00 | 3963.30 | 3990.12 | 3974.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 13:15:00 | 3973.30 | 3990.12 | 3974.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-06 15:15:00 | 3978.95 | 3980.59 | 3972.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 10:45:00 | 3972.45 | 3979.26 | 3973.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-10 09:15:00 | 3930.10 | 3987.03 | 3982.35 | SL hit (close<static) qty=1.00 sl=3942.00 alert=retest2 |

### Cycle 73 — SELL (started 2025-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 10:15:00 | 3920.25 | 3973.68 | 3976.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-11 11:15:00 | 3896.80 | 3926.51 | 3945.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-11 14:15:00 | 3902.60 | 3901.97 | 3927.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-11 15:00:00 | 3902.60 | 3901.97 | 3927.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-12 09:15:00 | 3789.85 | 3881.63 | 3914.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 10:45:00 | 3778.30 | 3858.90 | 3882.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 11:15:00 | 3774.20 | 3858.90 | 3882.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 11:15:00 | 3589.39 | 3695.28 | 3774.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-17 11:15:00 | 3585.49 | 3695.28 | 3774.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-18 11:15:00 | 3641.60 | 3633.85 | 3695.65 | SL hit (close>ema200) qty=0.50 sl=3633.85 alert=retest2 |

### Cycle 74 — BUY (started 2025-02-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 15:15:00 | 3702.45 | 3691.50 | 3690.29 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2025-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-20 09:15:00 | 3661.90 | 3685.58 | 3687.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-20 10:15:00 | 3627.75 | 3674.01 | 3682.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 13:15:00 | 3698.65 | 3671.28 | 3678.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-20 13:15:00 | 3698.65 | 3671.28 | 3678.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-20 13:15:00 | 3698.65 | 3671.28 | 3678.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-20 14:00:00 | 3698.65 | 3671.28 | 3678.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 76 — BUY (started 2025-02-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 14:15:00 | 3747.80 | 3686.59 | 3684.75 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2025-02-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 14:15:00 | 3677.30 | 3690.00 | 3690.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-24 09:15:00 | 3628.10 | 3674.26 | 3683.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 12:15:00 | 3684.60 | 3671.97 | 3679.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 12:15:00 | 3684.60 | 3671.97 | 3679.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 12:15:00 | 3684.60 | 3671.97 | 3679.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:00:00 | 3684.60 | 3671.97 | 3679.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 13:15:00 | 3679.05 | 3673.38 | 3679.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 13:45:00 | 3691.20 | 3673.38 | 3679.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 14:15:00 | 3682.15 | 3675.14 | 3679.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 15:00:00 | 3682.15 | 3675.14 | 3679.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 15:15:00 | 3685.05 | 3677.12 | 3680.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 09:15:00 | 3661.85 | 3677.12 | 3680.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-27 15:15:00 | 3478.76 | 3534.21 | 3583.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-03-03 11:15:00 | 3295.66 | 3362.63 | 3440.32 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 78 — BUY (started 2025-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 12:15:00 | 3369.70 | 3355.35 | 3353.72 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2025-03-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 15:15:00 | 3330.00 | 3348.35 | 3350.80 | EMA200 below EMA400 |

### Cycle 80 — BUY (started 2025-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 09:15:00 | 3360.15 | 3349.24 | 3348.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 10:15:00 | 3391.65 | 3357.72 | 3352.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 14:15:00 | 3422.10 | 3429.57 | 3406.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-19 15:00:00 | 3422.10 | 3429.57 | 3406.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 11:15:00 | 3453.40 | 3473.65 | 3454.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 11:45:00 | 3456.25 | 3473.65 | 3454.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 12:15:00 | 3456.10 | 3470.14 | 3454.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 12:30:00 | 3431.00 | 3470.14 | 3454.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 13:15:00 | 3408.55 | 3457.82 | 3450.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-21 14:00:00 | 3408.55 | 3457.82 | 3450.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 14:15:00 | 3429.35 | 3452.13 | 3448.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 15:15:00 | 3436.40 | 3452.13 | 3448.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 11:30:00 | 3440.65 | 3488.85 | 3483.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 14:15:00 | 3461.50 | 3477.53 | 3479.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 14:15:00 | 3461.50 | 3477.53 | 3479.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 15:15:00 | 3446.15 | 3471.25 | 3476.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-26 09:15:00 | 3471.90 | 3471.38 | 3475.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-26 09:15:00 | 3471.90 | 3471.38 | 3475.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 09:15:00 | 3471.90 | 3471.38 | 3475.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:00:00 | 3471.90 | 3471.38 | 3475.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 10:15:00 | 3477.80 | 3472.67 | 3476.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-26 10:45:00 | 3480.15 | 3472.67 | 3476.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 3451.50 | 3468.43 | 3473.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 13:00:00 | 3440.00 | 3462.75 | 3470.81 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-27 10:15:00 | 3482.95 | 3446.84 | 3457.27 | SL hit (close>static) qty=1.00 sl=3479.00 alert=retest2 |

### Cycle 82 — BUY (started 2025-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-27 13:15:00 | 3485.00 | 3467.81 | 3465.54 | EMA200 above EMA400 |

### Cycle 83 — SELL (started 2025-03-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 09:15:00 | 3454.20 | 3462.45 | 3463.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 12:15:00 | 3426.00 | 3454.96 | 3459.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-28 15:15:00 | 3459.95 | 3445.85 | 3453.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-28 15:15:00 | 3459.95 | 3445.85 | 3453.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-28 15:15:00 | 3459.95 | 3445.85 | 3453.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 09:30:00 | 3446.50 | 3442.10 | 3451.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-03 09:15:00 | 3379.35 | 3361.18 | 3380.04 | EMA400 retest candle locked (from downside) |

### Cycle 84 — BUY (started 2025-04-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-03 13:15:00 | 3414.30 | 3390.05 | 3389.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 14:15:00 | 3422.50 | 3396.54 | 3392.48 | Break + close above crossover candle high |

### Cycle 85 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 3304.65 | 3382.40 | 3387.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 11:15:00 | 3284.00 | 3350.66 | 3371.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 10:15:00 | 3161.75 | 3154.34 | 3218.24 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 11:00:00 | 3161.75 | 3154.34 | 3218.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 13:15:00 | 3215.00 | 3174.71 | 3212.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-08 14:00:00 | 3215.00 | 3174.71 | 3212.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 14:15:00 | 3202.95 | 3180.36 | 3211.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 3153.80 | 3191.28 | 3213.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-11 10:15:00 | 3166.40 | 3159.34 | 3178.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 09:15:00 | 3290.90 | 3190.40 | 3183.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 3290.90 | 3190.40 | 3183.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 11:15:00 | 3314.40 | 3230.15 | 3203.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 15:15:00 | 3382.50 | 3392.23 | 3357.45 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-04-21 09:15:00 | 3411.20 | 3392.23 | 3357.45 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 3486.90 | 3504.70 | 3473.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 3468.60 | 3504.70 | 3473.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 09:15:00 | 3500.00 | 3508.91 | 3490.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 09:45:00 | 3464.00 | 3508.91 | 3490.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 11:15:00 | 3459.20 | 3497.62 | 3488.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-24 11:15:00 | 3459.20 | 3497.62 | 3488.08 | SL hit (close<ema400) qty=1.00 sl=3488.08 alert=retest1 |

### Cycle 87 — SELL (started 2025-04-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 11:15:00 | 3432.50 | 3476.90 | 3482.74 | EMA200 below EMA400 |

### Cycle 88 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 3561.60 | 3493.23 | 3487.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 12:15:00 | 3603.80 | 3537.74 | 3511.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-29 09:15:00 | 3570.80 | 3581.25 | 3544.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-29 10:00:00 | 3570.80 | 3581.25 | 3544.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 13:15:00 | 3574.70 | 3579.82 | 3555.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 13:30:00 | 3577.30 | 3579.82 | 3555.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 09:15:00 | 3541.70 | 3570.50 | 3557.48 | EMA400 retest candle locked (from upside) |

### Cycle 89 — SELL (started 2025-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 13:15:00 | 3518.70 | 3550.35 | 3551.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 10:15:00 | 3512.10 | 3537.96 | 3544.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 12:15:00 | 3481.20 | 3472.53 | 3497.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-05 12:30:00 | 3480.60 | 3472.53 | 3497.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 14:15:00 | 3496.50 | 3479.64 | 3496.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 15:00:00 | 3496.50 | 3479.64 | 3496.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 15:15:00 | 3502.00 | 3484.11 | 3496.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:15:00 | 3489.60 | 3484.11 | 3496.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 3449.60 | 3477.21 | 3492.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 10:15:00 | 3435.00 | 3477.21 | 3492.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 11:00:00 | 3432.30 | 3468.23 | 3487.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-06 12:30:00 | 3438.60 | 3461.33 | 3480.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 09:30:00 | 3412.10 | 3444.35 | 3465.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 3457.40 | 3448.79 | 3464.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:30:00 | 3470.30 | 3448.79 | 3464.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 3433.40 | 3445.71 | 3461.51 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-05-08 10:15:00 | 3493.10 | 3467.60 | 3466.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 3493.10 | 3467.60 | 3466.63 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-05-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-09 09:15:00 | 3452.60 | 3468.57 | 3469.25 | EMA200 below EMA400 |

### Cycle 92 — BUY (started 2025-05-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-09 14:15:00 | 3485.20 | 3470.58 | 3469.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 09:15:00 | 3592.10 | 3497.51 | 3481.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 3576.10 | 3579.67 | 3549.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 14:00:00 | 3576.10 | 3579.67 | 3549.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 3573.30 | 3579.34 | 3557.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 09:15:00 | 3642.00 | 3569.46 | 3560.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2025-05-26 13:15:00 | 4006.20 | 3942.64 | 3888.20 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 93 — SELL (started 2025-06-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 10:15:00 | 4311.40 | 4354.38 | 4359.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 11:15:00 | 4291.00 | 4341.70 | 4353.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 10:15:00 | 4310.60 | 4296.40 | 4320.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 10:15:00 | 4310.60 | 4296.40 | 4320.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 10:15:00 | 4310.60 | 4296.40 | 4320.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 10:45:00 | 4343.80 | 4296.40 | 4320.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 11:15:00 | 4272.40 | 4291.60 | 4315.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 12:45:00 | 4265.50 | 4287.08 | 4311.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-13 14:15:00 | 4340.00 | 4302.93 | 4314.93 | SL hit (close>static) qty=1.00 sl=4320.70 alert=retest2 |

### Cycle 94 — BUY (started 2025-06-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 11:15:00 | 4376.40 | 4326.30 | 4322.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 12:15:00 | 4400.00 | 4341.04 | 4329.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 13:15:00 | 4622.90 | 4643.88 | 4574.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 13:45:00 | 4624.00 | 4643.88 | 4574.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 4490.50 | 4600.37 | 4570.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 10:00:00 | 4490.50 | 4600.37 | 4570.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 10:15:00 | 4489.00 | 4578.10 | 4563.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 11:15:00 | 4473.00 | 4578.10 | 4563.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 95 — SELL (started 2025-06-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 12:15:00 | 4455.00 | 4537.24 | 4546.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 15:15:00 | 4422.00 | 4487.08 | 4518.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 14:15:00 | 4454.90 | 4439.53 | 4475.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 14:15:00 | 4454.90 | 4439.53 | 4475.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 4454.90 | 4439.53 | 4475.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:45:00 | 4497.70 | 4439.53 | 4475.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 4452.10 | 4442.04 | 4473.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 4332.50 | 4442.04 | 4473.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 11:15:00 | 4536.00 | 4482.49 | 4477.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 96 — BUY (started 2025-06-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 11:15:00 | 4536.00 | 4482.49 | 4477.01 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-25 09:15:00 | 4449.10 | 4470.43 | 4473.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-25 10:15:00 | 4425.50 | 4461.44 | 4468.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-27 09:15:00 | 4385.50 | 4379.04 | 4403.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-27 09:15:00 | 4385.50 | 4379.04 | 4403.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 09:15:00 | 4385.50 | 4379.04 | 4403.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-27 09:30:00 | 4391.80 | 4379.04 | 4403.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 09:15:00 | 4336.80 | 4353.14 | 4377.27 | EMA400 retest candle locked (from downside) |

### Cycle 98 — BUY (started 2025-07-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 12:15:00 | 4382.80 | 4376.81 | 4376.06 | EMA200 above EMA400 |

### Cycle 99 — SELL (started 2025-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 09:15:00 | 4344.00 | 4373.09 | 4375.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 10:15:00 | 4314.60 | 4361.39 | 4369.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 09:15:00 | 4293.30 | 4287.20 | 4310.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 09:15:00 | 4293.30 | 4287.20 | 4310.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 4293.30 | 4287.20 | 4310.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 09:30:00 | 4295.50 | 4287.20 | 4310.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 4274.60 | 4271.56 | 4288.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 10:00:00 | 4274.60 | 4271.56 | 4288.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 10:15:00 | 4256.30 | 4268.51 | 4285.95 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:45:00 | 4246.60 | 4264.99 | 4282.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 14:30:00 | 4245.70 | 4263.38 | 4277.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:45:00 | 4253.00 | 4260.29 | 4273.45 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-15 11:15:00 | 4204.70 | 4144.70 | 4141.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 4204.70 | 4144.70 | 4141.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 4212.90 | 4158.34 | 4147.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 09:15:00 | 4186.00 | 4194.76 | 4171.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-16 10:00:00 | 4186.00 | 4194.76 | 4171.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 4216.10 | 4242.64 | 4223.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 10:00:00 | 4216.10 | 4242.64 | 4223.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 4231.00 | 4240.31 | 4224.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-18 13:45:00 | 4241.10 | 4234.37 | 4225.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 09:15:00 | 4268.50 | 4229.61 | 4224.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 09:15:00 | 4123.40 | 4228.40 | 4236.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — SELL (started 2025-07-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 09:15:00 | 4123.40 | 4228.40 | 4236.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 12:15:00 | 4072.30 | 4162.72 | 4201.42 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 4135.00 | 4112.40 | 4143.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 14:15:00 | 4135.00 | 4112.40 | 4143.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 4135.00 | 4112.40 | 4143.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:45:00 | 4146.20 | 4112.40 | 4143.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 4134.70 | 4116.86 | 4142.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 4083.00 | 4116.86 | 4142.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:45:00 | 4111.10 | 4115.95 | 4139.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 13:15:00 | 4236.10 | 4124.00 | 4134.07 | SL hit (close>static) qty=1.00 sl=4144.50 alert=retest2 |

### Cycle 102 — BUY (started 2025-07-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-24 14:15:00 | 4250.20 | 4149.24 | 4144.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-25 09:15:00 | 4292.80 | 4192.15 | 4165.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 14:15:00 | 4295.70 | 4307.28 | 4266.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-28 15:00:00 | 4295.70 | 4307.28 | 4266.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 4264.00 | 4312.67 | 4297.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:00:00 | 4264.00 | 4312.67 | 4297.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 4278.60 | 4305.85 | 4295.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:30:00 | 4249.60 | 4305.85 | 4295.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 4274.00 | 4297.17 | 4293.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-30 12:30:00 | 4276.00 | 4297.17 | 4293.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 13:15:00 | 4276.70 | 4293.08 | 4291.76 | EMA400 retest candle locked (from upside) |

### Cycle 103 — SELL (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 14:15:00 | 4276.10 | 4289.68 | 4290.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 15:15:00 | 4270.00 | 4285.74 | 4288.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 11:15:00 | 4283.50 | 4277.26 | 4283.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 11:15:00 | 4283.50 | 4277.26 | 4283.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 11:15:00 | 4283.50 | 4277.26 | 4283.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 12:00:00 | 4283.50 | 4277.26 | 4283.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 12:15:00 | 4294.30 | 4280.67 | 4284.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 13:00:00 | 4294.30 | 4280.67 | 4284.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — BUY (started 2025-07-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 13:15:00 | 4312.80 | 4287.09 | 4286.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 09:15:00 | 4319.00 | 4299.37 | 4293.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 11:15:00 | 4293.70 | 4300.99 | 4295.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 11:15:00 | 4293.70 | 4300.99 | 4295.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 4293.70 | 4300.99 | 4295.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:45:00 | 4298.20 | 4300.99 | 4295.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 4293.10 | 4299.42 | 4294.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 13:00:00 | 4293.10 | 4299.42 | 4294.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 13:15:00 | 4275.50 | 4294.63 | 4293.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 14:00:00 | 4275.50 | 4294.63 | 4293.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 4247.00 | 4285.11 | 4288.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 09:15:00 | 4239.00 | 4270.24 | 4281.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 11:15:00 | 4284.10 | 4266.52 | 4277.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 11:15:00 | 4284.10 | 4266.52 | 4277.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 4284.10 | 4266.52 | 4277.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:00:00 | 4284.10 | 4266.52 | 4277.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 4288.70 | 4270.96 | 4278.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 12:30:00 | 4287.30 | 4270.96 | 4278.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 4276.50 | 4272.07 | 4278.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-05 14:00:00 | 4248.30 | 4269.58 | 4275.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 09:15:00 | 4194.40 | 4270.05 | 4274.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 10:15:00 | 4300.80 | 4191.34 | 4184.25 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — BUY (started 2025-08-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 10:15:00 | 4300.80 | 4191.34 | 4184.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 13:15:00 | 4327.50 | 4251.90 | 4216.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 10:15:00 | 4273.00 | 4285.35 | 4247.21 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-12 11:00:00 | 4273.00 | 4285.35 | 4247.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 13:15:00 | 4242.40 | 4275.83 | 4252.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 14:00:00 | 4242.40 | 4275.83 | 4252.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 4215.00 | 4263.66 | 4248.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 4215.00 | 4263.66 | 4248.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 4289.20 | 4265.49 | 4252.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 10:45:00 | 4294.70 | 4265.49 | 4252.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 4301.80 | 4304.08 | 4288.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:30:00 | 4299.40 | 4304.08 | 4288.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 4307.00 | 4304.66 | 4290.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 4389.60 | 4304.66 | 4290.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-28 09:15:00 | 4570.10 | 4600.44 | 4601.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — SELL (started 2025-08-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-28 09:15:00 | 4570.10 | 4600.44 | 4601.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-28 14:15:00 | 4501.90 | 4557.08 | 4578.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 09:15:00 | 4523.90 | 4500.28 | 4527.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-01 09:15:00 | 4523.90 | 4500.28 | 4527.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 4523.90 | 4500.28 | 4527.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 09:45:00 | 4517.20 | 4500.28 | 4527.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 10:15:00 | 4534.40 | 4507.11 | 4527.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:00:00 | 4534.40 | 4507.11 | 4527.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 4560.00 | 4517.68 | 4530.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:45:00 | 4556.40 | 4517.68 | 4530.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 108 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 4573.80 | 4545.65 | 4541.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 4599.00 | 4559.48 | 4548.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 4586.20 | 4610.20 | 4596.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 4586.20 | 4610.20 | 4596.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 4586.20 | 4610.20 | 4596.05 | EMA400 retest candle locked (from upside) |

### Cycle 109 — SELL (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 13:15:00 | 4581.40 | 4589.65 | 4589.70 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 09:15:00 | 4599.40 | 4590.55 | 4589.99 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 4555.10 | 4583.36 | 4586.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 09:15:00 | 4530.20 | 4566.28 | 4576.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-11 09:15:00 | 4328.00 | 4315.02 | 4373.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-11 09:45:00 | 4331.20 | 4315.02 | 4373.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 11:15:00 | 4373.00 | 4334.37 | 4372.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-11 11:30:00 | 4362.00 | 4334.37 | 4372.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 4352.60 | 4338.02 | 4371.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-11 15:15:00 | 4348.90 | 4346.07 | 4369.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 09:15:00 | 4447.90 | 4366.89 | 4374.79 | SL hit (close>static) qty=1.00 sl=4379.90 alert=retest2 |

### Cycle 112 — BUY (started 2025-09-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 11:15:00 | 4449.40 | 4390.67 | 4384.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 12:15:00 | 4485.30 | 4409.60 | 4393.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 10:15:00 | 4470.40 | 4471.99 | 4436.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 11:00:00 | 4470.40 | 4471.99 | 4436.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 13:15:00 | 4510.40 | 4561.51 | 4535.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-17 14:00:00 | 4510.40 | 4561.51 | 4535.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 4507.80 | 4550.77 | 4533.12 | EMA400 retest candle locked (from upside) |

### Cycle 113 — SELL (started 2025-09-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 11:15:00 | 4491.80 | 4521.52 | 4523.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-18 12:15:00 | 4472.20 | 4511.66 | 4518.57 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 12:15:00 | 4375.00 | 4346.99 | 4372.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 12:15:00 | 4375.00 | 4346.99 | 4372.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 4375.00 | 4346.99 | 4372.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:00:00 | 4375.00 | 4346.99 | 4372.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 4378.40 | 4353.27 | 4372.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:45:00 | 4347.70 | 4353.12 | 4370.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 10:30:00 | 4340.00 | 4344.12 | 4361.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 10:15:00 | 4220.50 | 4195.12 | 4192.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2025-10-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-07 10:15:00 | 4220.50 | 4195.12 | 4192.07 | EMA200 above EMA400 |

### Cycle 115 — SELL (started 2025-10-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 10:15:00 | 4162.40 | 4193.57 | 4194.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 14:15:00 | 4149.30 | 4176.41 | 4185.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 4173.60 | 4172.37 | 4180.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 11:00:00 | 4173.60 | 4172.37 | 4180.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 4182.80 | 4174.46 | 4181.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:00:00 | 4182.80 | 4174.46 | 4181.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 4180.30 | 4175.63 | 4181.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 13:30:00 | 4170.00 | 4176.04 | 4180.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 14:30:00 | 4173.10 | 4176.17 | 4180.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-09 15:15:00 | 4173.00 | 4176.17 | 4180.40 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-10 10:15:00 | 4187.30 | 4178.56 | 4180.41 | SL hit (close>static) qty=1.00 sl=4186.00 alert=retest2 |

### Cycle 116 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 4235.50 | 4191.65 | 4186.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-10 13:15:00 | 4256.50 | 4204.62 | 4192.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 09:15:00 | 4245.40 | 4247.63 | 4227.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-14 10:00:00 | 4245.40 | 4247.63 | 4227.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 4220.10 | 4242.12 | 4227.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 4220.10 | 4242.12 | 4227.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 11:15:00 | 4198.10 | 4233.32 | 4224.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 12:00:00 | 4198.10 | 4233.32 | 4224.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 12:15:00 | 4173.40 | 4221.33 | 4219.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 13:00:00 | 4173.40 | 4221.33 | 4219.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — SELL (started 2025-10-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 13:15:00 | 4184.00 | 4213.87 | 4216.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 14:15:00 | 4163.70 | 4203.83 | 4211.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 4210.80 | 4198.68 | 4206.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 10:15:00 | 4210.80 | 4198.68 | 4206.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 4210.80 | 4198.68 | 4206.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 4210.80 | 4198.68 | 4206.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 4193.60 | 4197.67 | 4205.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 12:30:00 | 4189.90 | 4197.35 | 4204.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 14:00:00 | 4189.50 | 4195.78 | 4203.27 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 4223.70 | 4204.65 | 4205.67 | SL hit (close>static) qty=1.00 sl=4219.90 alert=retest2 |

### Cycle 118 — BUY (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 10:15:00 | 4225.70 | 4208.86 | 4207.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 15:15:00 | 4247.90 | 4229.46 | 4219.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 10:15:00 | 4213.00 | 4228.75 | 4220.96 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-17 10:15:00 | 4213.00 | 4228.75 | 4220.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 10:15:00 | 4213.00 | 4228.75 | 4220.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:00:00 | 4213.00 | 4228.75 | 4220.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 11:15:00 | 4213.10 | 4225.62 | 4220.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 11:45:00 | 4215.70 | 4225.62 | 4220.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 4195.60 | 4219.62 | 4218.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 12:45:00 | 4209.90 | 4219.62 | 4218.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2025-10-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-17 13:15:00 | 4198.40 | 4215.37 | 4216.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-23 09:15:00 | 4164.30 | 4192.74 | 4203.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 14:15:00 | 3847.60 | 3842.20 | 3899.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 15:00:00 | 3847.60 | 3842.20 | 3899.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 3796.20 | 3812.31 | 3836.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 14:45:00 | 3831.00 | 3812.31 | 3836.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 10:15:00 | 3876.90 | 3822.95 | 3834.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:00:00 | 3876.90 | 3822.95 | 3834.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 120 — BUY (started 2025-11-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-03 11:15:00 | 3930.30 | 3844.42 | 3843.46 | EMA200 above EMA400 |

### Cycle 121 — SELL (started 2025-11-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-06 11:15:00 | 3844.40 | 3871.60 | 3872.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 12:15:00 | 3841.00 | 3865.48 | 3869.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 14:15:00 | 3825.30 | 3817.26 | 3836.13 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 14:15:00 | 3825.30 | 3817.26 | 3836.13 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 14:15:00 | 3825.30 | 3817.26 | 3836.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 15:00:00 | 3825.30 | 3817.26 | 3836.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 15:15:00 | 3831.20 | 3820.05 | 3835.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:15:00 | 3832.00 | 3820.05 | 3835.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 3842.00 | 3824.44 | 3836.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 09:45:00 | 3853.70 | 3824.44 | 3836.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 3860.00 | 3831.55 | 3838.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 3860.00 | 3831.55 | 3838.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 3875.80 | 3844.15 | 3843.19 | EMA200 above EMA400 |

### Cycle 123 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 3829.40 | 3844.69 | 3844.84 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 11:15:00 | 3855.80 | 3846.91 | 3845.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 12:15:00 | 3859.00 | 3849.33 | 3847.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 13:15:00 | 3857.80 | 3873.93 | 3864.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 13:15:00 | 3857.80 | 3873.93 | 3864.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 13:15:00 | 3857.80 | 3873.93 | 3864.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-12 14:00:00 | 3857.80 | 3873.93 | 3864.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 3922.00 | 3883.54 | 3869.28 | EMA400 retest candle locked (from upside) |

### Cycle 125 — SELL (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 13:15:00 | 3833.00 | 3860.28 | 3863.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 14:15:00 | 3828.00 | 3853.83 | 3860.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 3442.80 | 3437.31 | 3465.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 3442.80 | 3437.31 | 3465.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 3442.80 | 3437.31 | 3465.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 09:15:00 | 3390.70 | 3425.36 | 3440.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 13:15:00 | 3370.60 | 3334.70 | 3331.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 126 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 3370.60 | 3334.70 | 3331.40 | EMA200 above EMA400 |

### Cycle 127 — SELL (started 2025-12-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 12:15:00 | 3332.30 | 3335.22 | 3335.47 | EMA200 below EMA400 |

### Cycle 128 — BUY (started 2025-12-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 13:15:00 | 3342.00 | 3336.58 | 3336.06 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 14:15:00 | 3328.30 | 3334.92 | 3335.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 10:15:00 | 3312.70 | 3330.48 | 3333.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 09:15:00 | 3250.60 | 3244.41 | 3270.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-12 09:30:00 | 3260.70 | 3244.41 | 3270.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 11:15:00 | 3257.20 | 3230.88 | 3245.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 12:00:00 | 3257.20 | 3230.88 | 3245.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 12:15:00 | 3256.20 | 3235.95 | 3246.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 13:15:00 | 3266.90 | 3235.95 | 3246.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2025-12-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 15:15:00 | 3284.00 | 3256.35 | 3254.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-16 09:15:00 | 3369.60 | 3279.00 | 3264.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 09:15:00 | 3362.90 | 3370.00 | 3329.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-17 10:00:00 | 3362.90 | 3370.00 | 3329.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 11:15:00 | 3329.10 | 3357.50 | 3330.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 11:30:00 | 3334.30 | 3357.50 | 3330.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 12:15:00 | 3341.70 | 3354.34 | 3331.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 13:15:00 | 3358.70 | 3354.34 | 3331.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 14:45:00 | 3349.70 | 3349.48 | 3332.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 3348.80 | 3346.58 | 3333.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 12:00:00 | 3349.80 | 3342.83 | 3334.30 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 15:15:00 | 3346.00 | 3345.19 | 3338.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:15:00 | 3378.00 | 3345.19 | 3338.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 3343.50 | 3344.85 | 3338.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 09:15:00 | 3393.30 | 3361.86 | 3350.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 15:15:00 | 3336.00 | 3354.46 | 3353.62 | SL hit (close<static) qty=1.00 sl=3336.10 alert=retest2 |

### Cycle 131 — SELL (started 2025-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 09:15:00 | 3346.30 | 3352.83 | 3352.96 | EMA200 below EMA400 |

### Cycle 132 — BUY (started 2025-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-23 13:15:00 | 3370.70 | 3353.57 | 3352.79 | EMA200 above EMA400 |

### Cycle 133 — SELL (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 09:15:00 | 3350.40 | 3352.08 | 3352.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 13:15:00 | 3329.80 | 3344.97 | 3348.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 3300.00 | 3282.78 | 3291.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 15:15:00 | 3300.00 | 3282.78 | 3291.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 3300.00 | 3282.78 | 3291.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 3364.00 | 3282.78 | 3291.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 3341.50 | 3294.52 | 3295.65 | EMA400 retest candle locked (from downside) |

### Cycle 134 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 3339.40 | 3303.50 | 3299.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 13:15:00 | 3358.00 | 3326.26 | 3311.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 11:15:00 | 3614.70 | 3622.35 | 3588.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 11:45:00 | 3613.20 | 3622.35 | 3588.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 13:15:00 | 3604.00 | 3615.32 | 3591.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-07 14:30:00 | 3613.10 | 3617.76 | 3594.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-08 09:30:00 | 3614.60 | 3617.39 | 3598.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 11:15:00 | 3585.80 | 3607.90 | 3597.23 | SL hit (close<static) qty=1.00 sl=3586.00 alert=retest2 |

### Cycle 135 — SELL (started 2026-01-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 14:15:00 | 3558.40 | 3586.32 | 3589.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 11:15:00 | 3528.60 | 3559.52 | 3574.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 14:15:00 | 3473.80 | 3472.52 | 3497.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 15:00:00 | 3473.80 | 3472.52 | 3497.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 3504.70 | 3479.35 | 3496.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:00:00 | 3504.70 | 3479.35 | 3496.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 3492.20 | 3481.92 | 3495.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 12:00:00 | 3481.70 | 3481.88 | 3494.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:00:00 | 3486.10 | 3482.72 | 3493.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 14:15:00 | 3484.70 | 3484.60 | 3493.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:00:00 | 3480.90 | 3483.86 | 3492.43 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 15:15:00 | 3499.90 | 3487.07 | 3493.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-16 09:15:00 | 3514.20 | 3487.07 | 3493.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 3543.00 | 3498.25 | 3497.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — BUY (started 2026-01-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 09:15:00 | 3543.00 | 3498.25 | 3497.65 | EMA200 above EMA400 |

### Cycle 137 — SELL (started 2026-01-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 09:15:00 | 3486.00 | 3499.34 | 3500.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 12:15:00 | 3443.80 | 3483.80 | 3493.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 3374.00 | 3363.89 | 3408.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 13:00:00 | 3374.00 | 3363.89 | 3408.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 3447.80 | 3370.79 | 3396.01 | EMA400 retest candle locked (from downside) |

### Cycle 138 — BUY (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 12:15:00 | 3461.20 | 3413.75 | 3411.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 13:15:00 | 3492.20 | 3429.44 | 3418.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-27 09:15:00 | 3441.90 | 3475.27 | 3459.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 3441.90 | 3475.27 | 3459.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 3441.90 | 3475.27 | 3459.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:00:00 | 3441.90 | 3475.27 | 3459.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 3463.60 | 3472.94 | 3459.42 | EMA400 retest candle locked (from upside) |

### Cycle 139 — SELL (started 2026-01-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 14:15:00 | 3441.40 | 3451.06 | 3451.97 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 15:15:00 | 3461.00 | 3453.05 | 3452.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 09:15:00 | 3526.00 | 3467.64 | 3459.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 3489.40 | 3496.90 | 3482.01 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 3489.40 | 3496.90 | 3482.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 3489.40 | 3496.90 | 3482.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:30:00 | 3488.20 | 3496.90 | 3482.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 3484.20 | 3494.36 | 3482.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 3476.90 | 3494.36 | 3482.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 3495.50 | 3494.59 | 3483.42 | EMA400 retest candle locked (from upside) |

### Cycle 141 — SELL (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 13:15:00 | 3474.50 | 3483.77 | 3484.51 | EMA200 below EMA400 |

### Cycle 142 — BUY (started 2026-01-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 14:15:00 | 3509.00 | 3488.82 | 3486.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 09:15:00 | 3530.20 | 3498.21 | 3491.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 13:15:00 | 3503.30 | 3508.35 | 3499.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 13:15:00 | 3503.30 | 3508.35 | 3499.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 3503.30 | 3508.35 | 3499.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 3500.30 | 3508.35 | 3499.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 3492.40 | 3505.16 | 3498.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 3492.40 | 3505.16 | 3498.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 3491.50 | 3502.43 | 3498.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 3478.80 | 3502.43 | 3498.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 10:15:00 | 3505.40 | 3502.54 | 3499.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 11:15:00 | 3493.90 | 3502.54 | 3499.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 11:15:00 | 3523.50 | 3506.73 | 3501.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 12:15:00 | 3528.30 | 3506.73 | 3501.26 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-16 14:15:00 | 3881.13 | 3833.73 | 3811.55 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2026-02-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 14:15:00 | 3975.50 | 3996.75 | 3997.32 | EMA200 below EMA400 |

### Cycle 144 — BUY (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-02 10:15:00 | 4037.00 | 4002.79 | 3999.65 | EMA200 above EMA400 |

### Cycle 145 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 3961.50 | 3995.29 | 3996.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 13:15:00 | 3941.00 | 3984.43 | 3991.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 12:15:00 | 3935.10 | 3905.66 | 3929.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-05 12:15:00 | 3935.10 | 3905.66 | 3929.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 3935.10 | 3905.66 | 3929.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:00:00 | 3935.10 | 3905.66 | 3929.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 3922.10 | 3908.95 | 3928.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:30:00 | 3926.70 | 3908.95 | 3928.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 3945.50 | 3916.26 | 3930.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 3945.50 | 3916.26 | 3930.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 3940.00 | 3921.01 | 3931.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 4025.00 | 3921.01 | 3931.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — BUY (started 2026-03-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 09:15:00 | 4029.50 | 3942.71 | 3940.26 | EMA200 above EMA400 |

### Cycle 147 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 3835.20 | 3935.87 | 3942.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-09 11:15:00 | 3787.50 | 3888.15 | 3918.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-10 09:15:00 | 3907.70 | 3855.31 | 3885.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-10 09:15:00 | 3907.70 | 3855.31 | 3885.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 3907.70 | 3855.31 | 3885.89 | EMA400 retest candle locked (from downside) |

### Cycle 148 — BUY (started 2026-03-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 15:15:00 | 3947.90 | 3907.90 | 3903.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 3962.10 | 3918.74 | 3908.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 3945.10 | 3983.81 | 3954.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 3945.10 | 3983.81 | 3954.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 3945.10 | 3983.81 | 3954.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 3935.10 | 3983.81 | 3954.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 4001.20 | 3987.29 | 3959.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 11:30:00 | 4028.10 | 3995.89 | 3965.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 10:15:00 | 3920.40 | 3986.18 | 3977.42 | SL hit (close<static) qty=1.00 sl=3934.50 alert=retest2 |

### Cycle 149 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 3894.70 | 3957.89 | 3965.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 15:15:00 | 3877.30 | 3924.34 | 3946.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 3929.60 | 3911.54 | 3928.74 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-16 14:15:00 | 3929.60 | 3911.54 | 3928.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 3929.60 | 3911.54 | 3928.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 15:00:00 | 3929.60 | 3911.54 | 3928.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 3890.20 | 3907.27 | 3925.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 3914.00 | 3907.27 | 3925.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 3921.40 | 3910.10 | 3924.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:45:00 | 3922.00 | 3910.10 | 3924.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 3907.70 | 3909.62 | 3923.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 3877.00 | 3909.62 | 3923.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 15:15:00 | 3960.10 | 3917.34 | 3920.60 | SL hit (close>static) qty=1.00 sl=3941.80 alert=retest2 |

### Cycle 150 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 4003.20 | 3934.51 | 3928.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 4006.30 | 3971.12 | 3949.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 3914.60 | 3967.43 | 3953.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 3914.60 | 3967.43 | 3953.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 3914.60 | 3967.43 | 3953.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:45:00 | 3925.40 | 3967.43 | 3953.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 10:15:00 | 3896.80 | 3953.30 | 3948.75 | EMA400 retest candle locked (from upside) |

### Cycle 151 — SELL (started 2026-03-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 11:15:00 | 3903.20 | 3943.28 | 3944.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 3867.30 | 3917.62 | 3932.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 3720.00 | 3718.72 | 3770.46 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:45:00 | 3724.90 | 3718.72 | 3770.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 3758.80 | 3726.74 | 3769.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:30:00 | 3760.10 | 3726.74 | 3769.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 15:15:00 | 3768.50 | 3743.45 | 3767.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:15:00 | 3901.40 | 3743.45 | 3767.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 3897.40 | 3774.24 | 3779.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 3897.40 | 3774.24 | 3779.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 152 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 3900.00 | 3799.39 | 3790.04 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2026-03-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 13:15:00 | 3797.90 | 3805.28 | 3805.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 3745.90 | 3793.40 | 3800.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-30 15:15:00 | 3740.00 | 3735.70 | 3759.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 15:15:00 | 3740.00 | 3735.70 | 3759.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 3740.00 | 3735.70 | 3759.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:15:00 | 3837.50 | 3735.70 | 3759.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 3792.70 | 3747.10 | 3762.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 3756.10 | 3747.10 | 3762.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-04-02 09:15:00 | 3568.29 | 3650.12 | 3704.66 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-02 13:15:00 | 3612.30 | 3599.82 | 3657.96 | SL hit (close>ema200) qty=0.50 sl=3599.82 alert=retest2 |

### Cycle 154 — BUY (started 2026-04-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 09:15:00 | 3694.10 | 3665.57 | 3662.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 10:15:00 | 3711.10 | 3674.68 | 3666.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 13:15:00 | 3764.80 | 3778.66 | 3742.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-08 14:00:00 | 3764.80 | 3778.66 | 3742.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 14:15:00 | 3794.60 | 3781.84 | 3746.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 12:15:00 | 3800.20 | 3783.02 | 3758.57 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:45:00 | 3802.20 | 3779.86 | 3765.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 15:15:00 | 3801.00 | 3782.88 | 3772.39 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 09:45:00 | 3824.10 | 3793.04 | 3779.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 12:15:00 | 3796.30 | 3797.32 | 3784.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-13 12:30:00 | 3802.00 | 3797.32 | 3784.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 15:15:00 | 3787.30 | 3796.22 | 3787.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 3833.40 | 3796.22 | 3787.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 3723.30 | 3801.85 | 3800.76 | SL hit (close<static) qty=1.00 sl=3745.60 alert=retest2 |

### Cycle 155 — SELL (started 2026-04-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 10:15:00 | 3743.00 | 3790.08 | 3795.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 11:15:00 | 3686.10 | 3769.28 | 3785.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 09:15:00 | 3718.10 | 3703.92 | 3740.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-17 09:45:00 | 3705.50 | 3703.92 | 3740.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 15:15:00 | 3731.00 | 3707.03 | 3725.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:15:00 | 3729.10 | 3707.03 | 3725.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 3706.00 | 3706.82 | 3723.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 13:15:00 | 3672.70 | 3705.27 | 3715.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:30:00 | 3671.00 | 3691.99 | 3698.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 11:30:00 | 3655.00 | 3685.25 | 3695.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 15:00:00 | 3673.60 | 3685.87 | 3693.22 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 09:15:00 | 3696.90 | 3684.87 | 3691.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 10:00:00 | 3696.90 | 3684.87 | 3691.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 3631.40 | 3674.17 | 3685.84 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-27 10:15:00 | 3720.00 | 3683.98 | 3682.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 156 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 3720.00 | 3683.98 | 3682.65 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2026-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 09:15:00 | 3657.50 | 3684.42 | 3685.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 12:15:00 | 3648.80 | 3672.49 | 3679.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-29 09:15:00 | 3674.60 | 3658.64 | 3669.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 09:15:00 | 3674.60 | 3658.64 | 3669.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 3674.60 | 3658.64 | 3669.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 09:15:00 | 3607.10 | 3667.41 | 3670.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-30 15:00:00 | 3618.40 | 3618.50 | 3638.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 12:00:00 | 3635.10 | 3638.29 | 3642.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 11:45:00 | 3631.60 | 3637.75 | 3638.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 12:15:00 | 3642.30 | 3638.66 | 3638.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:00:00 | 3642.30 | 3638.66 | 3638.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 3633.60 | 3637.65 | 3638.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:30:00 | 3648.80 | 3637.65 | 3638.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 3637.10 | 3637.54 | 3638.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-05 15:15:00 | 3625.20 | 3637.54 | 3638.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 3650.00 | 3638.06 | 3638.15 | SL hit (close>static) qty=1.00 sl=3644.30 alert=retest2 |

### Cycle 158 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 3661.30 | 3642.71 | 3640.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 11:15:00 | 3689.60 | 3652.08 | 3644.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 10:15:00 | 3687.10 | 3688.47 | 3669.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 11:00:00 | 3687.10 | 3688.47 | 3669.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 3676.50 | 3686.08 | 3669.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:45:00 | 3673.00 | 3686.08 | 3669.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 3665.10 | 3681.88 | 3669.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 13:00:00 | 3665.10 | 3681.88 | 3669.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 13:15:00 | 3665.00 | 3678.51 | 3669.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 13:45:00 | 3658.20 | 3678.51 | 3669.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 14:15:00 | 3663.10 | 3675.43 | 3668.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 14:30:00 | 3659.90 | 3675.43 | 3668.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 3669.60 | 3674.26 | 3668.63 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-08 09:15:00 | 3688.20 | 3674.26 | 3668.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 12:15:00 | 3650.20 | 3666.88 | 3666.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 159 — SELL (started 2026-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 12:15:00 | 3650.20 | 3666.88 | 3666.89 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-07 15:00:00 | 5763.00 | 2024-06-12 09:15:00 | 6339.30 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-06-25 12:45:00 | 5945.80 | 2024-06-28 14:15:00 | 5950.10 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2024-06-27 10:30:00 | 5887.15 | 2024-06-28 14:15:00 | 5950.10 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2024-07-02 14:15:00 | 5970.00 | 2024-07-08 10:15:00 | 5939.40 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2024-07-12 12:30:00 | 5795.60 | 2024-07-23 12:15:00 | 5505.82 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-15 12:00:00 | 5820.05 | 2024-07-23 12:15:00 | 5529.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-15 13:15:00 | 5819.45 | 2024-07-23 12:15:00 | 5528.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-15 14:00:00 | 5824.05 | 2024-07-23 12:15:00 | 5532.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-16 12:45:00 | 5814.00 | 2024-07-23 12:15:00 | 5523.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-16 13:15:00 | 5815.95 | 2024-07-23 12:15:00 | 5525.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-16 13:45:00 | 5793.40 | 2024-07-23 12:15:00 | 5503.73 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-19 09:30:00 | 5775.70 | 2024-07-23 12:15:00 | 5486.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-22 09:15:00 | 5720.00 | 2024-07-24 09:15:00 | 5434.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-22 10:45:00 | 5719.95 | 2024-07-24 09:15:00 | 5433.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-12 12:30:00 | 5795.60 | 2024-07-25 09:15:00 | 5238.05 | TARGET_HIT | 0.50 | 9.62% |
| SELL | retest2 | 2024-07-15 12:00:00 | 5820.05 | 2024-07-25 09:15:00 | 5241.65 | TARGET_HIT | 0.50 | 9.94% |
| SELL | retest2 | 2024-07-15 13:15:00 | 5819.45 | 2024-07-26 09:15:00 | 5483.00 | STOP_HIT | 0.50 | 5.78% |
| SELL | retest2 | 2024-07-15 14:00:00 | 5824.05 | 2024-07-26 09:15:00 | 5483.00 | STOP_HIT | 0.50 | 5.86% |
| SELL | retest2 | 2024-07-16 12:45:00 | 5814.00 | 2024-07-26 09:15:00 | 5483.00 | STOP_HIT | 0.50 | 5.69% |
| SELL | retest2 | 2024-07-16 13:15:00 | 5815.95 | 2024-07-26 09:15:00 | 5483.00 | STOP_HIT | 0.50 | 5.72% |
| SELL | retest2 | 2024-07-16 13:45:00 | 5793.40 | 2024-07-26 09:15:00 | 5483.00 | STOP_HIT | 0.50 | 5.36% |
| SELL | retest2 | 2024-07-19 09:30:00 | 5775.70 | 2024-07-26 09:15:00 | 5483.00 | STOP_HIT | 0.50 | 5.07% |
| SELL | retest2 | 2024-07-22 09:15:00 | 5720.00 | 2024-07-26 09:15:00 | 5483.00 | STOP_HIT | 0.50 | 4.14% |
| SELL | retest2 | 2024-07-22 10:45:00 | 5719.95 | 2024-07-26 09:15:00 | 5483.00 | STOP_HIT | 0.50 | 4.14% |
| SELL | retest2 | 2024-07-31 12:30:00 | 5345.00 | 2024-08-05 09:15:00 | 5077.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-31 15:15:00 | 5350.00 | 2024-08-05 09:15:00 | 5082.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 12:15:00 | 5348.30 | 2024-08-05 09:15:00 | 5080.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-01 13:45:00 | 5337.00 | 2024-08-05 09:15:00 | 5070.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-07-31 12:30:00 | 5345.00 | 2024-08-06 09:15:00 | 5036.80 | STOP_HIT | 0.50 | 5.77% |
| SELL | retest2 | 2024-07-31 15:15:00 | 5350.00 | 2024-08-06 09:15:00 | 5036.80 | STOP_HIT | 0.50 | 5.85% |
| SELL | retest2 | 2024-08-01 12:15:00 | 5348.30 | 2024-08-06 09:15:00 | 5036.80 | STOP_HIT | 0.50 | 5.82% |
| SELL | retest2 | 2024-08-01 13:45:00 | 5337.00 | 2024-08-06 09:15:00 | 5036.80 | STOP_HIT | 0.50 | 5.62% |
| SELL | retest2 | 2024-08-06 14:30:00 | 5058.95 | 2024-08-07 11:15:00 | 5283.80 | STOP_HIT | 1.00 | -4.44% |
| SELL | retest2 | 2024-08-06 15:00:00 | 5011.40 | 2024-08-07 11:15:00 | 5283.80 | STOP_HIT | 1.00 | -5.44% |
| SELL | retest2 | 2024-08-12 09:15:00 | 5084.05 | 2024-08-12 12:15:00 | 5149.80 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2024-08-12 13:45:00 | 5075.65 | 2024-08-13 11:15:00 | 5154.00 | STOP_HIT | 1.00 | -1.54% |
| SELL | retest2 | 2024-08-13 09:30:00 | 5101.00 | 2024-08-13 11:15:00 | 5154.00 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-08-23 09:15:00 | 5577.00 | 2024-08-26 09:15:00 | 5442.00 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2024-08-23 13:00:00 | 5543.55 | 2024-08-26 09:15:00 | 5442.00 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-08-30 09:15:00 | 5301.50 | 2024-09-06 10:15:00 | 5410.00 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2024-08-30 15:00:00 | 5267.40 | 2024-09-06 10:15:00 | 5410.00 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2024-09-02 10:15:00 | 5301.00 | 2024-09-06 10:15:00 | 5410.00 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2024-09-03 13:00:00 | 5300.00 | 2024-09-06 10:15:00 | 5410.00 | STOP_HIT | 1.00 | -2.08% |
| SELL | retest2 | 2024-09-03 14:30:00 | 5267.75 | 2024-09-06 10:15:00 | 5410.00 | STOP_HIT | 1.00 | -2.70% |
| SELL | retest2 | 2024-09-30 09:30:00 | 5215.00 | 2024-09-30 11:15:00 | 5369.60 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2024-10-18 09:15:00 | 5177.00 | 2024-10-18 15:15:00 | 4940.19 | PARTIAL | 0.50 | 4.57% |
| SELL | retest2 | 2024-10-18 11:15:00 | 5200.20 | 2024-10-22 10:15:00 | 4918.15 | PARTIAL | 0.50 | 5.42% |
| SELL | retest2 | 2024-10-18 09:15:00 | 5177.00 | 2024-10-22 13:15:00 | 4659.30 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-18 11:15:00 | 5200.20 | 2024-10-22 13:15:00 | 4680.18 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2024-11-11 10:30:00 | 4641.65 | 2024-11-13 09:15:00 | 4519.35 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2024-11-11 13:45:00 | 4641.20 | 2024-11-13 09:15:00 | 4519.35 | STOP_HIT | 1.00 | -2.63% |
| BUY | retest2 | 2024-11-11 14:30:00 | 4653.45 | 2024-11-13 09:15:00 | 4519.35 | STOP_HIT | 1.00 | -2.88% |
| BUY | retest2 | 2024-11-12 10:30:00 | 4646.00 | 2024-11-13 09:15:00 | 4519.35 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2024-11-28 10:30:00 | 4539.95 | 2024-11-29 11:15:00 | 4652.15 | STOP_HIT | 1.00 | -2.47% |
| SELL | retest2 | 2024-11-28 11:15:00 | 4550.10 | 2024-11-29 11:15:00 | 4652.15 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2024-11-28 12:00:00 | 4551.75 | 2024-11-29 11:15:00 | 4652.15 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2024-11-28 12:30:00 | 4551.25 | 2024-11-29 11:15:00 | 4652.15 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2024-12-19 10:30:00 | 5082.95 | 2024-12-20 09:15:00 | 4927.30 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2024-12-19 11:00:00 | 5084.90 | 2024-12-20 09:15:00 | 4927.30 | STOP_HIT | 1.00 | -3.10% |
| SELL | retest1 | 2024-12-26 12:00:00 | 4727.00 | 2024-12-27 09:15:00 | 4782.70 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest1 | 2024-12-26 13:30:00 | 4735.50 | 2024-12-27 09:15:00 | 4782.70 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2024-12-27 15:15:00 | 4750.00 | 2025-01-03 09:15:00 | 4727.70 | STOP_HIT | 1.00 | 0.47% |
| SELL | retest2 | 2025-01-16 10:45:00 | 4360.00 | 2025-01-20 09:15:00 | 4142.00 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-16 10:45:00 | 4360.00 | 2025-01-22 09:15:00 | 3924.00 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-02-06 13:15:00 | 3973.30 | 2025-02-10 09:15:00 | 3930.10 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-02-06 15:15:00 | 3978.95 | 2025-02-10 09:15:00 | 3930.10 | STOP_HIT | 1.00 | -1.23% |
| BUY | retest2 | 2025-02-07 10:45:00 | 3972.45 | 2025-02-10 09:15:00 | 3930.10 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-02-14 10:45:00 | 3778.30 | 2025-02-17 11:15:00 | 3589.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 11:15:00 | 3774.20 | 2025-02-17 11:15:00 | 3585.49 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-14 10:45:00 | 3778.30 | 2025-02-18 11:15:00 | 3641.60 | STOP_HIT | 0.50 | 3.62% |
| SELL | retest2 | 2025-02-14 11:15:00 | 3774.20 | 2025-02-18 11:15:00 | 3641.60 | STOP_HIT | 0.50 | 3.51% |
| SELL | retest2 | 2025-02-25 09:15:00 | 3661.85 | 2025-02-27 15:15:00 | 3478.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 09:15:00 | 3661.85 | 2025-03-03 11:15:00 | 3295.66 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-03-21 15:15:00 | 3436.40 | 2025-03-25 14:15:00 | 3461.50 | STOP_HIT | 1.00 | 0.73% |
| BUY | retest2 | 2025-03-25 11:30:00 | 3440.65 | 2025-03-25 14:15:00 | 3461.50 | STOP_HIT | 1.00 | 0.61% |
| SELL | retest2 | 2025-03-26 13:00:00 | 3440.00 | 2025-03-27 10:15:00 | 3482.95 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2025-04-09 09:15:00 | 3153.80 | 2025-04-15 09:15:00 | 3290.90 | STOP_HIT | 1.00 | -4.35% |
| SELL | retest2 | 2025-04-11 10:15:00 | 3166.40 | 2025-04-15 09:15:00 | 3290.90 | STOP_HIT | 1.00 | -3.93% |
| BUY | retest1 | 2025-04-21 09:15:00 | 3411.20 | 2025-04-24 11:15:00 | 3459.20 | STOP_HIT | 1.00 | 1.41% |
| SELL | retest2 | 2025-05-06 10:15:00 | 3435.00 | 2025-05-08 10:15:00 | 3493.10 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-05-06 11:00:00 | 3432.30 | 2025-05-08 10:15:00 | 3493.10 | STOP_HIT | 1.00 | -1.77% |
| SELL | retest2 | 2025-05-06 12:30:00 | 3438.60 | 2025-05-08 10:15:00 | 3493.10 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-05-07 09:30:00 | 3412.10 | 2025-05-08 10:15:00 | 3493.10 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-05-15 09:15:00 | 3642.00 | 2025-05-26 13:15:00 | 4006.20 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-06-13 12:45:00 | 4265.50 | 2025-06-13 14:15:00 | 4340.00 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2025-06-23 09:15:00 | 4332.50 | 2025-06-24 11:15:00 | 4536.00 | STOP_HIT | 1.00 | -4.70% |
| SELL | retest2 | 2025-07-07 11:45:00 | 4246.60 | 2025-07-15 11:15:00 | 4204.70 | STOP_HIT | 1.00 | 0.99% |
| SELL | retest2 | 2025-07-07 14:30:00 | 4245.70 | 2025-07-15 11:15:00 | 4204.70 | STOP_HIT | 1.00 | 0.97% |
| SELL | retest2 | 2025-07-08 09:45:00 | 4253.00 | 2025-07-15 11:15:00 | 4204.70 | STOP_HIT | 1.00 | 1.14% |
| BUY | retest2 | 2025-07-18 13:45:00 | 4241.10 | 2025-07-22 09:15:00 | 4123.40 | STOP_HIT | 1.00 | -2.78% |
| BUY | retest2 | 2025-07-21 09:15:00 | 4268.50 | 2025-07-22 09:15:00 | 4123.40 | STOP_HIT | 1.00 | -3.40% |
| SELL | retest2 | 2025-07-24 09:15:00 | 4083.00 | 2025-07-24 13:15:00 | 4236.10 | STOP_HIT | 1.00 | -3.75% |
| SELL | retest2 | 2025-07-24 09:45:00 | 4111.10 | 2025-07-24 13:15:00 | 4236.10 | STOP_HIT | 1.00 | -3.04% |
| SELL | retest2 | 2025-08-05 14:00:00 | 4248.30 | 2025-08-11 10:15:00 | 4300.80 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-08-06 09:15:00 | 4194.40 | 2025-08-11 10:15:00 | 4300.80 | STOP_HIT | 1.00 | -2.54% |
| BUY | retest2 | 2025-08-18 09:15:00 | 4389.60 | 2025-08-28 09:15:00 | 4570.10 | STOP_HIT | 1.00 | 4.11% |
| SELL | retest2 | 2025-09-11 15:15:00 | 4348.90 | 2025-09-12 09:15:00 | 4447.90 | STOP_HIT | 1.00 | -2.28% |
| SELL | retest2 | 2025-09-24 14:45:00 | 4347.70 | 2025-10-07 10:15:00 | 4220.50 | STOP_HIT | 1.00 | 2.93% |
| SELL | retest2 | 2025-09-25 10:30:00 | 4340.00 | 2025-10-07 10:15:00 | 4220.50 | STOP_HIT | 1.00 | 2.75% |
| SELL | retest2 | 2025-10-09 13:30:00 | 4170.00 | 2025-10-10 10:15:00 | 4187.30 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-10-09 14:30:00 | 4173.10 | 2025-10-10 10:15:00 | 4187.30 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-10-09 15:15:00 | 4173.00 | 2025-10-10 10:15:00 | 4187.30 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-10-15 12:30:00 | 4189.90 | 2025-10-16 09:15:00 | 4223.70 | STOP_HIT | 1.00 | -0.81% |
| SELL | retest2 | 2025-10-15 14:00:00 | 4189.50 | 2025-10-16 09:15:00 | 4223.70 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-11-28 09:15:00 | 3390.70 | 2025-12-05 13:15:00 | 3370.60 | STOP_HIT | 1.00 | 0.59% |
| BUY | retest2 | 2025-12-17 13:15:00 | 3358.70 | 2025-12-22 15:15:00 | 3336.00 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-12-17 14:45:00 | 3349.70 | 2025-12-23 09:15:00 | 3346.30 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2025-12-18 09:15:00 | 3348.80 | 2025-12-23 09:15:00 | 3346.30 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-12-18 12:00:00 | 3349.80 | 2025-12-23 09:15:00 | 3346.30 | STOP_HIT | 1.00 | -0.10% |
| BUY | retest2 | 2025-12-22 09:15:00 | 3393.30 | 2025-12-23 09:15:00 | 3346.30 | STOP_HIT | 1.00 | -1.39% |
| BUY | retest2 | 2026-01-07 14:30:00 | 3613.10 | 2026-01-08 11:15:00 | 3585.80 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2026-01-08 09:30:00 | 3614.60 | 2026-01-08 11:15:00 | 3585.80 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2026-01-14 12:00:00 | 3481.70 | 2026-01-16 09:15:00 | 3543.00 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2026-01-14 13:00:00 | 3486.10 | 2026-01-16 09:15:00 | 3543.00 | STOP_HIT | 1.00 | -1.63% |
| SELL | retest2 | 2026-01-14 14:15:00 | 3484.70 | 2026-01-16 09:15:00 | 3543.00 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2026-01-14 15:00:00 | 3480.90 | 2026-01-16 09:15:00 | 3543.00 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2026-02-02 12:15:00 | 3528.30 | 2026-02-16 14:15:00 | 3881.13 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-03-12 11:30:00 | 4028.10 | 2026-03-13 10:15:00 | 3920.40 | STOP_HIT | 1.00 | -2.67% |
| SELL | retest2 | 2026-03-17 11:15:00 | 3877.00 | 2026-03-17 15:15:00 | 3960.10 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2026-04-01 10:15:00 | 3756.10 | 2026-04-02 09:15:00 | 3568.29 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-01 10:15:00 | 3756.10 | 2026-04-02 13:15:00 | 3612.30 | STOP_HIT | 0.50 | 3.83% |
| BUY | retest2 | 2026-04-09 12:15:00 | 3800.20 | 2026-04-16 09:15:00 | 3723.30 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2026-04-10 09:45:00 | 3802.20 | 2026-04-16 09:15:00 | 3723.30 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2026-04-10 15:15:00 | 3801.00 | 2026-04-16 09:15:00 | 3723.30 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2026-04-13 09:45:00 | 3824.10 | 2026-04-16 09:15:00 | 3723.30 | STOP_HIT | 1.00 | -2.64% |
| BUY | retest2 | 2026-04-15 09:15:00 | 3833.40 | 2026-04-16 09:15:00 | 3723.30 | STOP_HIT | 1.00 | -2.87% |
| SELL | retest2 | 2026-04-21 13:15:00 | 3672.70 | 2026-04-27 10:15:00 | 3720.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2026-04-23 10:30:00 | 3671.00 | 2026-04-27 10:15:00 | 3720.00 | STOP_HIT | 1.00 | -1.33% |
| SELL | retest2 | 2026-04-23 11:30:00 | 3655.00 | 2026-04-27 10:15:00 | 3720.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2026-04-23 15:00:00 | 3673.60 | 2026-04-27 10:15:00 | 3720.00 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-04-30 09:15:00 | 3607.10 | 2026-05-06 09:15:00 | 3650.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2026-04-30 15:00:00 | 3618.40 | 2026-05-06 10:15:00 | 3661.30 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2026-05-04 12:00:00 | 3635.10 | 2026-05-06 10:15:00 | 3661.30 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2026-05-05 11:45:00 | 3631.60 | 2026-05-06 10:15:00 | 3661.30 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2026-05-05 15:15:00 | 3625.20 | 2026-05-06 10:15:00 | 3661.30 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2026-05-08 09:15:00 | 3688.20 | 2026-05-08 12:15:00 | 3650.20 | STOP_HIT | 1.00 | -1.03% |
