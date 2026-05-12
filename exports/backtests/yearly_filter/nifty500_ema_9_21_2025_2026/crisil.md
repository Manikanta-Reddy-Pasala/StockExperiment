# CRISIL Ltd. (CRISIL)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 4160.70
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 67 |
| ALERT1 | 48 |
| ALERT2 | 47 |
| ALERT2_SKIP | 23 |
| ALERT3 | 129 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 63 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 62 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 66 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 17 / 49
- **Target hits / Stop hits / Partials:** 1 / 62 / 3
- **Avg / median % per leg:** -0.26% / -0.78%
- **Sum % (uncompounded):** -17.12%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 22 | 5 | 22.7% | 0 | 22 | 0 | -0.06% | -1.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 22 | 5 | 22.7% | 0 | 22 | 0 | -0.06% | -1.4% |
| SELL (all) | 44 | 12 | 27.3% | 1 | 40 | 3 | -0.36% | -15.7% |
| SELL @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 0 | 2 | 1 | 3.49% | 10.5% |
| SELL @ 3rd Alert (retest2) | 41 | 10 | 24.4% | 1 | 38 | 2 | -0.64% | -26.1% |
| retest1 (combined) | 3 | 2 | 66.7% | 0 | 2 | 1 | 3.49% | 10.5% |
| retest2 (combined) | 63 | 15 | 23.8% | 1 | 60 | 2 | -0.44% | -27.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 11:15:00 | 5019.10 | 5045.61 | 5048.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-16 14:15:00 | 5010.40 | 5033.80 | 5041.95 | Break + close below crossover candle low |

### Cycle 2 — BUY (started 2025-05-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-19 09:15:00 | 5159.70 | 5054.05 | 5049.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 10:15:00 | 5170.00 | 5077.24 | 5060.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-20 09:15:00 | 5095.30 | 5124.92 | 5098.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-20 09:15:00 | 5095.30 | 5124.92 | 5098.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 5095.30 | 5124.92 | 5098.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:45:00 | 5095.80 | 5124.92 | 5098.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 10:15:00 | 5066.50 | 5113.24 | 5095.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 11:00:00 | 5066.50 | 5113.24 | 5095.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 11:15:00 | 5045.30 | 5099.65 | 5091.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 12:00:00 | 5045.30 | 5099.65 | 5091.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 5012.90 | 5073.04 | 5079.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 11:15:00 | 5003.90 | 5045.14 | 5062.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 5099.80 | 5053.52 | 5061.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 14:15:00 | 5099.80 | 5053.52 | 5061.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 5099.80 | 5053.52 | 5061.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 15:00:00 | 5099.80 | 5053.52 | 5061.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 15:15:00 | 5086.20 | 5060.06 | 5063.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 09:15:00 | 5072.40 | 5060.06 | 5063.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 5080.20 | 5064.09 | 5065.23 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2025-05-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 10:15:00 | 5092.80 | 5069.83 | 5067.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 5129.80 | 5107.64 | 5094.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-26 14:15:00 | 5124.30 | 5125.29 | 5110.09 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-26 15:00:00 | 5124.30 | 5125.29 | 5110.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 5165.00 | 5132.39 | 5115.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 09:30:00 | 5154.90 | 5132.39 | 5115.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 5150.50 | 5155.56 | 5134.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:00:00 | 5150.50 | 5155.56 | 5134.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 14:15:00 | 5249.20 | 5265.90 | 5246.16 | EMA400 retest candle locked (from upside) |

### Cycle 5 — SELL (started 2025-06-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 11:15:00 | 5182.00 | 5227.63 | 5232.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 12:15:00 | 5162.50 | 5192.37 | 5208.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 09:15:00 | 5296.50 | 5201.75 | 5206.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 5296.50 | 5201.75 | 5206.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 5296.50 | 5201.75 | 5206.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:45:00 | 5274.00 | 5201.75 | 5206.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 5312.00 | 5223.80 | 5215.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-09 09:15:00 | 5415.00 | 5352.83 | 5314.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 5654.50 | 5683.18 | 5580.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 10:00:00 | 5654.50 | 5683.18 | 5580.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 5588.00 | 5643.45 | 5586.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 13:00:00 | 5588.00 | 5643.45 | 5586.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 5538.50 | 5622.46 | 5581.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 5538.50 | 5622.46 | 5581.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 5536.50 | 5605.27 | 5577.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 15:00:00 | 5536.50 | 5605.27 | 5577.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-06-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-13 09:15:00 | 5356.50 | 5546.67 | 5555.41 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-06-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 10:15:00 | 5673.50 | 5495.85 | 5475.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 14:15:00 | 5687.00 | 5604.72 | 5540.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 09:15:00 | 5600.50 | 5615.92 | 5557.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 10:00:00 | 5600.50 | 5615.92 | 5557.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 5556.50 | 5604.04 | 5557.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:00:00 | 5556.50 | 5604.04 | 5557.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 11:15:00 | 5560.00 | 5595.23 | 5557.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:30:00 | 5549.00 | 5595.23 | 5557.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 12:15:00 | 5486.50 | 5573.48 | 5550.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 13:00:00 | 5486.50 | 5573.48 | 5550.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 13:15:00 | 5465.50 | 5551.89 | 5543.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-18 14:00:00 | 5465.50 | 5551.89 | 5543.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 5617.00 | 5616.80 | 5585.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 12:45:00 | 5595.00 | 5616.80 | 5585.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 5585.00 | 5613.86 | 5591.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 09:15:00 | 5592.50 | 5613.86 | 5591.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 5702.00 | 5631.49 | 5601.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 10:15:00 | 5735.00 | 5631.49 | 5601.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 12:45:00 | 5725.00 | 5688.36 | 5638.25 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-20 15:15:00 | 5500.00 | 5674.65 | 5646.44 | SL hit (close<static) qty=1.00 sl=5533.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 11:15:00 | 5948.00 | 5998.81 | 6000.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-02 14:15:00 | 5925.00 | 5982.77 | 5992.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-03 09:15:00 | 6008.00 | 5973.22 | 5985.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-03 09:15:00 | 6008.00 | 5973.22 | 5985.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 6008.00 | 5973.22 | 5985.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:00:00 | 6008.00 | 5973.22 | 5985.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 6057.00 | 5989.97 | 5992.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 10:45:00 | 6050.00 | 5989.97 | 5992.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2025-07-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 11:15:00 | 6056.50 | 6003.28 | 5998.11 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 11:15:00 | 5874.50 | 6001.06 | 6004.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 12:15:00 | 5847.00 | 5970.25 | 5990.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 09:15:00 | 5830.00 | 5813.49 | 5870.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 09:15:00 | 5830.00 | 5813.49 | 5870.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 5830.00 | 5813.49 | 5870.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 11:00:00 | 5769.50 | 5804.69 | 5861.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 14:15:00 | 5880.00 | 5832.12 | 5837.07 | SL hit (close>static) qty=1.00 sl=5877.50 alert=retest2 |

### Cycle 12 — BUY (started 2025-07-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 15:15:00 | 5879.50 | 5841.59 | 5840.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 10:15:00 | 5892.00 | 5851.58 | 5845.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 11:15:00 | 5917.00 | 5921.77 | 5891.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 11:45:00 | 5918.50 | 5921.77 | 5891.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 12:15:00 | 5904.50 | 5918.32 | 5892.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:00:00 | 5904.50 | 5918.32 | 5892.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 5900.00 | 5914.65 | 5893.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:00:00 | 5900.00 | 5914.65 | 5893.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 5959.00 | 6010.41 | 5988.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:00:00 | 5959.00 | 6010.41 | 5988.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 5983.50 | 6005.03 | 5988.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:30:00 | 5985.00 | 6005.03 | 5988.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 5983.00 | 6000.62 | 5987.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 09:15:00 | 6095.00 | 6000.62 | 5987.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 11:30:00 | 6008.00 | 6004.52 | 5993.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-16 12:15:00 | 5948.00 | 5993.22 | 5989.77 | SL hit (close<static) qty=1.00 sl=5961.50 alert=retest2 |

### Cycle 13 — SELL (started 2025-07-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-16 13:15:00 | 5940.00 | 5982.57 | 5985.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-16 14:15:00 | 5934.00 | 5972.86 | 5980.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-17 09:15:00 | 5980.00 | 5965.83 | 5975.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 5980.00 | 5965.83 | 5975.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 5980.00 | 5965.83 | 5975.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:00:00 | 5980.00 | 5965.83 | 5975.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 5934.50 | 5959.56 | 5971.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:00:00 | 5901.50 | 5935.58 | 5953.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 14:15:00 | 5606.43 | 5664.33 | 5732.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-07-29 09:15:00 | 5311.35 | 5428.14 | 5544.46 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 14 — BUY (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 09:15:00 | 5313.00 | 5283.41 | 5283.00 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 5197.00 | 5292.53 | 5297.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 10:15:00 | 5180.00 | 5224.98 | 5254.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-08 11:15:00 | 5212.00 | 5188.66 | 5214.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-08 12:00:00 | 5212.00 | 5188.66 | 5214.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 12:15:00 | 5210.00 | 5192.93 | 5214.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 12:30:00 | 5236.00 | 5192.93 | 5214.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 14:15:00 | 5202.50 | 5196.37 | 5212.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 15:00:00 | 5202.50 | 5196.37 | 5212.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 15:15:00 | 5229.50 | 5203.00 | 5213.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 09:15:00 | 5221.50 | 5203.00 | 5213.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 09:15:00 | 5205.00 | 5203.40 | 5212.90 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 5231.00 | 5216.79 | 5216.47 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-12 09:15:00 | 5185.00 | 5210.94 | 5213.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-12 10:15:00 | 5158.00 | 5200.35 | 5208.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-13 14:15:00 | 5206.00 | 5150.20 | 5165.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 14:15:00 | 5206.00 | 5150.20 | 5165.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 5206.00 | 5150.20 | 5165.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 5206.00 | 5150.20 | 5165.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 5207.50 | 5161.66 | 5169.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 5221.50 | 5161.66 | 5169.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2025-08-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-14 10:15:00 | 5217.00 | 5182.78 | 5178.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 11:15:00 | 5267.00 | 5199.63 | 5186.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 13:15:00 | 5302.50 | 5304.45 | 5262.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 14:00:00 | 5302.50 | 5304.45 | 5262.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 14:15:00 | 5363.00 | 5383.31 | 5355.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 15:00:00 | 5363.00 | 5383.31 | 5355.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 15:15:00 | 5345.00 | 5375.65 | 5354.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 09:15:00 | 5376.00 | 5375.65 | 5354.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 13:15:00 | 5329.50 | 5375.72 | 5375.68 | SL hit (close<static) qty=1.00 sl=5341.50 alert=retest2 |

### Cycle 19 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 5309.50 | 5362.47 | 5369.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-25 09:15:00 | 5273.50 | 5335.48 | 5355.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-29 10:15:00 | 5018.50 | 4992.00 | 5074.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-29 11:00:00 | 5018.50 | 4992.00 | 5074.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 11:15:00 | 5038.20 | 5005.78 | 5037.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 11:30:00 | 5039.80 | 5005.78 | 5037.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 12:15:00 | 5031.40 | 5010.90 | 5036.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:00:00 | 5031.40 | 5010.90 | 5036.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 13:15:00 | 5030.90 | 5014.90 | 5035.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 13:30:00 | 5034.70 | 5014.90 | 5035.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 5088.00 | 5029.52 | 5040.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 15:00:00 | 5088.00 | 5029.52 | 5040.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 5058.90 | 5035.40 | 5042.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 09:15:00 | 5026.30 | 5035.40 | 5042.38 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-02 10:15:00 | 5076.30 | 5051.39 | 5048.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 20 — BUY (started 2025-09-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 10:15:00 | 5076.30 | 5051.39 | 5048.92 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 11:15:00 | 5012.40 | 5044.20 | 5047.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-03 13:15:00 | 5001.00 | 5029.02 | 5039.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-03 14:15:00 | 5045.50 | 5032.31 | 5040.46 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-03 14:15:00 | 5045.50 | 5032.31 | 5040.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 5045.50 | 5032.31 | 5040.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 15:00:00 | 5045.50 | 5032.31 | 5040.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 5044.10 | 5034.67 | 5040.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-04 09:15:00 | 5028.00 | 5034.67 | 5040.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 5031.50 | 5034.04 | 5039.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 13:00:00 | 5011.00 | 5028.65 | 5035.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-04 14:45:00 | 5019.00 | 5019.68 | 5030.36 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 10:00:00 | 5011.40 | 5014.96 | 5026.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-05 11:15:00 | 5019.10 | 5018.82 | 5026.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 11:15:00 | 5016.90 | 5018.44 | 5026.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-05 13:15:00 | 5062.90 | 5031.89 | 5031.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2025-09-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 13:15:00 | 5062.90 | 5031.89 | 5031.08 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2025-09-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 14:15:00 | 5010.10 | 5027.53 | 5029.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 11:15:00 | 4983.90 | 5012.03 | 5020.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 10:15:00 | 4982.10 | 4981.21 | 4998.95 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 10:15:00 | 4982.10 | 4981.21 | 4998.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 10:15:00 | 4982.10 | 4981.21 | 4998.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 11:00:00 | 4982.10 | 4981.21 | 4998.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 4989.40 | 4982.67 | 4993.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 4989.40 | 4982.67 | 4993.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 4981.80 | 4982.50 | 4992.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:15:00 | 5010.30 | 4982.50 | 4992.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 5029.90 | 4991.98 | 4996.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-10 09:30:00 | 5046.40 | 4991.98 | 4996.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 24 — BUY (started 2025-09-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 10:15:00 | 5078.10 | 5009.20 | 5003.65 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2025-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 13:15:00 | 5009.80 | 5023.68 | 5024.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 15:15:00 | 5005.90 | 5017.70 | 5021.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-15 12:15:00 | 5012.50 | 5012.30 | 5017.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-15 13:00:00 | 5012.50 | 5012.30 | 5017.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 13:15:00 | 5020.40 | 5013.92 | 5017.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 14:00:00 | 5020.40 | 5013.92 | 5017.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 5013.80 | 5013.90 | 5017.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-15 15:00:00 | 5013.80 | 5013.90 | 5017.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 5000.00 | 5011.12 | 5015.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:15:00 | 5043.90 | 5011.12 | 5015.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 5028.00 | 5014.49 | 5016.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 09:30:00 | 5041.10 | 5014.49 | 5016.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 5016.00 | 5014.80 | 5016.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:00:00 | 4993.90 | 5011.13 | 5014.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 13:30:00 | 5007.70 | 5010.84 | 5014.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 14:00:00 | 5009.70 | 5010.84 | 5014.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 14:15:00 | 5046.20 | 5017.91 | 5017.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 14:15:00 | 5046.20 | 5017.91 | 5017.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 15:15:00 | 5051.90 | 5024.71 | 5020.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 13:15:00 | 5034.80 | 5034.85 | 5028.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 13:45:00 | 5039.90 | 5034.85 | 5028.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 14:15:00 | 5080.00 | 5043.88 | 5032.83 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2025-09-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 15:15:00 | 5018.60 | 5031.92 | 5032.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 5015.30 | 5028.60 | 5031.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 10:15:00 | 4818.00 | 4800.23 | 4846.48 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-25 12:00:00 | 4765.40 | 4793.26 | 4839.11 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-30 09:15:00 | 4527.13 | 4601.41 | 4663.70 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 4479.90 | 4464.31 | 4516.89 | SL hit (close>ema200) qty=0.50 sl=4464.31 alert=retest1 |

### Cycle 28 — BUY (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 14:15:00 | 4632.90 | 4549.71 | 4540.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-07 13:15:00 | 4680.10 | 4613.10 | 4586.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-08 11:15:00 | 4654.20 | 4667.03 | 4628.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-08 12:00:00 | 4654.20 | 4667.03 | 4628.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 12:15:00 | 4629.70 | 4659.56 | 4628.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 13:00:00 | 4629.70 | 4659.56 | 4628.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 13:15:00 | 4624.30 | 4652.51 | 4628.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 13:30:00 | 4626.50 | 4652.51 | 4628.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 14:15:00 | 4633.40 | 4648.69 | 4628.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 14:45:00 | 4605.40 | 4648.69 | 4628.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 15:15:00 | 4636.10 | 4646.17 | 4629.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:15:00 | 4608.00 | 4646.17 | 4629.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 4598.70 | 4636.68 | 4626.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 09:45:00 | 4616.00 | 4636.68 | 4626.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 10:15:00 | 4623.30 | 4634.00 | 4626.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 10:30:00 | 4598.00 | 4634.00 | 4626.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 11:15:00 | 4630.00 | 4633.20 | 4626.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 13:15:00 | 4639.00 | 4631.42 | 4626.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-09 15:15:00 | 4635.80 | 4631.01 | 4627.24 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 10:00:00 | 4638.20 | 4633.21 | 4628.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-10 10:45:00 | 4655.30 | 4640.67 | 4632.72 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 4612.60 | 4659.15 | 4648.90 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-13 11:15:00 | 4606.10 | 4642.52 | 4642.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 4606.10 | 4642.52 | 4642.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 13:15:00 | 4582.00 | 4624.25 | 4634.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 13:15:00 | 4591.60 | 4588.97 | 4607.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 13:15:00 | 4591.60 | 4588.97 | 4607.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 13:15:00 | 4591.60 | 4588.97 | 4607.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:00:00 | 4591.60 | 4588.97 | 4607.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 4585.00 | 4587.87 | 4603.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:15:00 | 4662.30 | 4587.87 | 4603.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2025-10-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 09:15:00 | 4736.20 | 4617.54 | 4615.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 09:15:00 | 4786.40 | 4729.46 | 4709.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 09:15:00 | 4948.10 | 5001.29 | 4971.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 09:15:00 | 4948.10 | 5001.29 | 4971.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 09:15:00 | 4948.10 | 5001.29 | 4971.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:00:00 | 4948.10 | 5001.29 | 4971.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 4958.70 | 4992.77 | 4970.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 10:30:00 | 4934.40 | 4992.77 | 4970.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 14:15:00 | 4985.00 | 4991.18 | 4977.00 | EMA400 retest candle locked (from upside) |

### Cycle 31 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 4952.30 | 4969.04 | 4970.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-31 14:15:00 | 4930.10 | 4948.16 | 4954.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 12:15:00 | 4808.70 | 4800.90 | 4837.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 13:00:00 | 4808.70 | 4800.90 | 4837.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 4806.00 | 4804.59 | 4833.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 4735.70 | 4805.67 | 4831.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 10:00:00 | 4769.10 | 4798.36 | 4825.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 11:00:00 | 4757.00 | 4790.09 | 4819.30 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 10:15:00 | 4736.20 | 4680.50 | 4674.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — BUY (started 2025-11-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 10:15:00 | 4736.20 | 4680.50 | 4674.93 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 4636.10 | 4680.90 | 4681.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 15:15:00 | 4628.50 | 4649.28 | 4659.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 12:15:00 | 4548.20 | 4545.97 | 4571.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-24 13:15:00 | 4544.00 | 4545.97 | 4571.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 4485.00 | 4530.32 | 4556.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 13:00:00 | 4463.70 | 4510.42 | 4539.85 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 11:00:00 | 4471.30 | 4509.48 | 4519.01 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 09:45:00 | 4446.40 | 4434.96 | 4454.99 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 13:15:00 | 4485.30 | 4465.18 | 4464.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2025-12-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 13:15:00 | 4485.30 | 4465.18 | 4464.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 14:15:00 | 4493.80 | 4470.91 | 4467.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 11:15:00 | 4442.10 | 4473.11 | 4470.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-02 11:15:00 | 4442.10 | 4473.11 | 4470.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 11:15:00 | 4442.10 | 4473.11 | 4470.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 12:00:00 | 4442.10 | 4473.11 | 4470.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2025-12-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 12:15:00 | 4431.10 | 4464.71 | 4466.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 13:15:00 | 4396.20 | 4451.01 | 4460.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 10:15:00 | 4432.60 | 4429.01 | 4444.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 10:15:00 | 4432.60 | 4429.01 | 4444.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 4432.60 | 4429.01 | 4444.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:45:00 | 4435.30 | 4429.01 | 4444.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 4435.80 | 4430.37 | 4444.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:45:00 | 4438.80 | 4430.37 | 4444.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 4464.90 | 4437.28 | 4446.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:45:00 | 4481.40 | 4437.28 | 4446.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 13:15:00 | 4439.90 | 4437.80 | 4445.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 10:30:00 | 4415.00 | 4435.62 | 4442.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 14:30:00 | 4427.00 | 4429.21 | 4430.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 15:15:00 | 4462.60 | 4435.89 | 4433.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2025-12-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 15:15:00 | 4462.60 | 4435.89 | 4433.11 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 4384.80 | 4424.21 | 4428.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 13:15:00 | 4358.30 | 4411.03 | 4422.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 4369.20 | 4361.84 | 4387.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 13:00:00 | 4369.20 | 4361.84 | 4387.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 4376.50 | 4364.77 | 4386.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 4374.00 | 4364.77 | 4386.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 4396.00 | 4371.02 | 4387.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 4396.00 | 4371.02 | 4387.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 4398.50 | 4376.51 | 4388.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 4439.50 | 4376.51 | 4388.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 4365.10 | 4357.57 | 4370.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 09:30:00 | 4324.50 | 4350.32 | 4365.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 10:45:00 | 4343.40 | 4350.65 | 4364.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-11 13:15:00 | 4376.10 | 4359.15 | 4365.43 | SL hit (close>static) qty=1.00 sl=4376.00 alert=retest2 |

### Cycle 38 — BUY (started 2025-12-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 15:15:00 | 4389.70 | 4369.14 | 4369.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 09:15:00 | 4468.80 | 4389.07 | 4378.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-15 09:15:00 | 4453.70 | 4462.35 | 4428.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-15 09:30:00 | 4438.90 | 4462.35 | 4428.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 09:15:00 | 4450.60 | 4465.38 | 4449.12 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 4417.40 | 4440.90 | 4441.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 4391.00 | 4430.32 | 4436.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 15:15:00 | 4424.00 | 4420.75 | 4427.94 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-12-18 09:15:00 | 4366.40 | 4420.75 | 4427.94 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 4354.70 | 4351.40 | 4377.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:00:00 | 4354.70 | 4351.40 | 4377.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 4389.60 | 4359.04 | 4378.62 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-19 10:15:00 | 4389.60 | 4359.04 | 4378.62 | SL hit (close>ema400) qty=1.00 sl=4378.62 alert=retest1 |

### Cycle 40 — BUY (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 15:15:00 | 4307.00 | 4296.68 | 4295.84 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 09:15:00 | 4261.30 | 4289.61 | 4292.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 10:15:00 | 4213.80 | 4274.45 | 4285.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 4290.90 | 4241.34 | 4258.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 4290.90 | 4241.34 | 4258.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 4290.90 | 4241.34 | 4258.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:00:00 | 4290.90 | 4241.34 | 4258.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 4307.00 | 4254.47 | 4263.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 4307.00 | 4254.47 | 4263.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 42 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 4298.60 | 4271.70 | 4270.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 4323.70 | 4287.99 | 4278.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 4662.20 | 4687.52 | 4613.65 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-08 10:00:00 | 4662.20 | 4687.52 | 4613.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 4715.80 | 4708.47 | 4661.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 10:45:00 | 4756.00 | 4715.42 | 4669.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-09 14:45:00 | 4751.00 | 4721.32 | 4686.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 12:15:00 | 4751.10 | 4722.40 | 4697.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-12 13:30:00 | 4761.40 | 4732.84 | 4707.20 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 4818.90 | 4804.14 | 4770.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-14 11:30:00 | 4855.90 | 4812.43 | 4780.14 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 15:15:00 | 4742.30 | 4793.66 | 4781.74 | SL hit (close<static) qty=1.00 sl=4769.50 alert=retest2 |

### Cycle 43 — SELL (started 2026-01-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-16 10:15:00 | 4696.00 | 4765.03 | 4770.23 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2026-01-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 11:15:00 | 4800.00 | 4770.38 | 4767.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-19 13:15:00 | 4800.80 | 4778.80 | 4772.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 10:15:00 | 4787.30 | 4791.92 | 4781.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-20 10:45:00 | 4785.50 | 4791.92 | 4781.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 11:15:00 | 4765.90 | 4786.72 | 4780.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 12:00:00 | 4765.90 | 4786.72 | 4780.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 4743.90 | 4778.15 | 4776.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 13:00:00 | 4743.90 | 4778.15 | 4776.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2026-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 13:15:00 | 4757.80 | 4774.08 | 4775.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 4589.80 | 4730.25 | 4754.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 4602.90 | 4495.56 | 4517.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-28 09:15:00 | 4602.90 | 4495.56 | 4517.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 4602.90 | 4495.56 | 4517.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:00:00 | 4602.90 | 4495.56 | 4517.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 10:15:00 | 4620.10 | 4520.47 | 4527.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:30:00 | 4610.00 | 4520.47 | 4527.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2026-01-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 11:15:00 | 4607.40 | 4537.85 | 4534.46 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 4500.00 | 4544.30 | 4545.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-29 12:15:00 | 4494.10 | 4534.26 | 4540.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 4548.50 | 4530.37 | 4536.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 4548.50 | 4530.37 | 4536.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 4548.50 | 4530.37 | 4536.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:00:00 | 4548.50 | 4530.37 | 4536.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — BUY (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 10:15:00 | 4588.10 | 4541.91 | 4540.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 11:15:00 | 4653.60 | 4564.25 | 4551.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 15:15:00 | 4598.00 | 4667.38 | 4635.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 15:15:00 | 4598.00 | 4667.38 | 4635.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 4598.00 | 4667.38 | 4635.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-02 09:15:00 | 4561.90 | 4667.38 | 4635.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 4635.50 | 4661.00 | 4635.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 12:15:00 | 4684.70 | 4654.85 | 4636.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 13:00:00 | 4690.30 | 4661.94 | 4641.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 12:00:00 | 4689.70 | 4754.27 | 4736.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 14:15:00 | 4689.60 | 4723.33 | 4725.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — SELL (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 14:15:00 | 4689.60 | 4723.33 | 4725.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 09:15:00 | 4671.80 | 4709.11 | 4718.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 09:15:00 | 4689.30 | 4665.89 | 4687.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-06 09:15:00 | 4689.30 | 4665.89 | 4687.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 4689.30 | 4665.89 | 4687.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:45:00 | 4690.00 | 4665.89 | 4687.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 4724.20 | 4677.55 | 4690.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:00:00 | 4724.20 | 4677.55 | 4690.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 4689.20 | 4679.88 | 4690.32 | EMA400 retest candle locked (from downside) |

### Cycle 50 — BUY (started 2026-02-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 14:15:00 | 4757.20 | 4707.78 | 4701.45 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2026-02-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 12:15:00 | 4694.00 | 4699.12 | 4699.72 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2026-02-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 10:15:00 | 4714.60 | 4699.69 | 4699.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 11:15:00 | 4725.60 | 4704.87 | 4702.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 14:15:00 | 4678.00 | 4707.48 | 4704.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-10 14:15:00 | 4678.00 | 4707.48 | 4704.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 4678.00 | 4707.48 | 4704.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:45:00 | 4680.00 | 4707.48 | 4704.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — SELL (started 2026-02-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 15:15:00 | 4673.30 | 4700.65 | 4701.83 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2026-02-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 11:15:00 | 4706.90 | 4703.26 | 4702.88 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2026-02-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 12:15:00 | 4690.00 | 4700.61 | 4701.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 09:15:00 | 4638.30 | 4686.36 | 4694.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-12 12:15:00 | 4666.80 | 4663.69 | 4680.69 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-12 12:30:00 | 4670.00 | 4663.69 | 4680.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-12 13:15:00 | 4675.60 | 4666.07 | 4680.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-13 09:15:00 | 4429.80 | 4665.68 | 4677.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-16 10:30:00 | 4637.60 | 4536.11 | 4572.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 13:15:00 | 4610.00 | 4584.54 | 4583.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 4610.00 | 4584.54 | 4583.26 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2026-02-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-18 09:15:00 | 4555.00 | 4581.48 | 4582.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-19 09:15:00 | 4512.90 | 4553.89 | 4566.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-19 15:15:00 | 4549.00 | 4530.79 | 4546.32 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 15:15:00 | 4549.00 | 4530.79 | 4546.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 4549.00 | 4530.79 | 4546.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 4594.70 | 4530.79 | 4546.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 4567.10 | 4538.05 | 4548.20 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2026-02-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 13:15:00 | 4582.60 | 4555.42 | 4554.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-23 09:15:00 | 4605.30 | 4566.40 | 4559.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-23 15:15:00 | 4594.80 | 4595.30 | 4580.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-24 09:15:00 | 4524.90 | 4595.30 | 4580.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 09:15:00 | 4514.00 | 4579.04 | 4574.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-24 09:30:00 | 4496.20 | 4579.04 | 4574.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2026-02-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 10:15:00 | 4522.30 | 4567.69 | 4569.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 12:15:00 | 4465.80 | 4536.16 | 4554.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 4523.60 | 4520.82 | 4539.53 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 10:00:00 | 4523.60 | 4520.82 | 4539.53 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 4495.90 | 4512.46 | 4530.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 12:30:00 | 4511.00 | 4512.46 | 4530.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 13:15:00 | 4536.50 | 4517.27 | 4531.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:00:00 | 4536.50 | 4517.27 | 4531.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 4539.60 | 4521.74 | 4532.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 15:00:00 | 4539.60 | 4521.74 | 4532.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 4536.50 | 4524.69 | 4532.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 09:15:00 | 4497.20 | 4524.69 | 4532.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 4272.34 | 4416.79 | 4452.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 4437.20 | 4416.79 | 4452.04 | SL hit (close>static) qty=0.50 sl=4416.79 alert=retest2 |

### Cycle 60 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 4304.60 | 4239.27 | 4237.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 11:15:00 | 4318.70 | 4255.16 | 4245.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 09:15:00 | 4217.80 | 4276.38 | 4262.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 09:15:00 | 4217.80 | 4276.38 | 4262.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 09:15:00 | 4217.80 | 4276.38 | 4262.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 09:30:00 | 4212.60 | 4276.38 | 4262.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 10:15:00 | 4262.40 | 4273.58 | 4262.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-12 12:00:00 | 4296.00 | 4278.06 | 4265.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-12 14:15:00 | 4245.00 | 4256.16 | 4257.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2026-03-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-12 14:15:00 | 4245.00 | 4256.16 | 4257.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 15:15:00 | 4217.00 | 4248.33 | 4253.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-18 10:15:00 | 4023.80 | 4018.86 | 4055.51 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-18 11:15:00 | 4026.70 | 4018.86 | 4055.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 11:15:00 | 4060.10 | 4027.11 | 4055.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 12:00:00 | 4060.10 | 4027.11 | 4055.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 12:15:00 | 4098.50 | 4041.39 | 4059.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:00:00 | 4098.50 | 4041.39 | 4059.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 13:15:00 | 4047.20 | 4042.55 | 4058.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 15:00:00 | 4033.20 | 4040.68 | 4056.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-19 09:15:00 | 4019.30 | 4039.94 | 4054.58 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-20 10:30:00 | 4031.20 | 4013.40 | 4026.19 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 10:15:00 | 4050.50 | 3995.25 | 3989.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 4050.50 | 3995.25 | 3989.61 | EMA200 above EMA400 |

### Cycle 63 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 3894.80 | 3970.64 | 3980.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 3849.00 | 3924.37 | 3953.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 3847.90 | 3804.30 | 3855.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 3847.90 | 3804.30 | 3855.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 3847.90 | 3804.30 | 3855.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 3751.50 | 3816.78 | 3840.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 14:30:00 | 3775.20 | 3764.61 | 3773.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 09:30:00 | 3772.80 | 3774.59 | 3777.01 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 10:45:00 | 3770.30 | 3775.11 | 3777.03 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 11:15:00 | 3779.00 | 3775.89 | 3777.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-07 12:15:00 | 3770.40 | 3775.89 | 3777.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-08 09:15:00 | 3937.20 | 3805.10 | 3789.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 3937.20 | 3805.10 | 3789.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 11:15:00 | 4010.70 | 3915.85 | 3867.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 14:15:00 | 3924.70 | 3929.36 | 3887.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 15:00:00 | 3924.70 | 3929.36 | 3887.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 4006.10 | 4039.44 | 3982.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 4058.80 | 4043.47 | 3989.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:45:00 | 4063.50 | 4046.73 | 4004.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 4085.00 | 4042.98 | 4009.88 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 11:15:00 | 4319.80 | 4344.21 | 4346.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 65 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 4319.80 | 4344.21 | 4346.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 13:15:00 | 4274.70 | 4324.83 | 4336.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 14:15:00 | 4295.00 | 4286.11 | 4304.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 15:00:00 | 4295.00 | 4286.11 | 4304.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 15:15:00 | 4290.00 | 4286.89 | 4303.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 09:15:00 | 4298.10 | 4286.89 | 4303.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 4294.00 | 4288.31 | 4302.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:15:00 | 4242.00 | 4282.85 | 4298.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 12:00:00 | 4247.50 | 4275.78 | 4293.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 13:30:00 | 4258.70 | 4270.03 | 4287.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 12:15:00 | 4354.80 | 4300.32 | 4295.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — BUY (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 12:15:00 | 4354.80 | 4300.32 | 4295.49 | EMA200 above EMA400 |

### Cycle 67 — SELL (started 2026-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 15:15:00 | 4266.60 | 4291.86 | 4295.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-04 10:15:00 | 4246.50 | 4278.70 | 4288.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-07 09:15:00 | 4181.40 | 4165.01 | 4186.01 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 09:15:00 | 4181.40 | 4165.01 | 4186.01 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 4181.40 | 4165.01 | 4186.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:30:00 | 4195.50 | 4165.01 | 4186.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 4190.00 | 4170.01 | 4186.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 15:00:00 | 4147.10 | 4171.02 | 4182.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 11:15:00 | 4167.20 | 4170.81 | 4179.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-08 12:15:00 | 4201.90 | 4179.30 | 4181.65 | SL hit (close>static) qty=1.00 sl=4195.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-06-20 10:15:00 | 5735.00 | 2025-06-20 15:15:00 | 5500.00 | STOP_HIT | 1.00 | -4.10% |
| BUY | retest2 | 2025-06-20 12:45:00 | 5725.00 | 2025-06-20 15:15:00 | 5500.00 | STOP_HIT | 1.00 | -3.93% |
| BUY | retest2 | 2025-06-23 09:45:00 | 5727.00 | 2025-07-02 11:15:00 | 5948.00 | STOP_HIT | 1.00 | 3.86% |
| SELL | retest2 | 2025-07-08 11:00:00 | 5769.50 | 2025-07-09 14:15:00 | 5880.00 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-07-16 09:15:00 | 6095.00 | 2025-07-16 12:15:00 | 5948.00 | STOP_HIT | 1.00 | -2.41% |
| BUY | retest2 | 2025-07-16 11:30:00 | 6008.00 | 2025-07-16 12:15:00 | 5948.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-07-18 10:00:00 | 5901.50 | 2025-07-25 14:15:00 | 5606.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-18 10:00:00 | 5901.50 | 2025-07-29 09:15:00 | 5311.35 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-08-21 09:15:00 | 5376.00 | 2025-08-22 13:15:00 | 5329.50 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2025-09-02 09:15:00 | 5026.30 | 2025-09-02 10:15:00 | 5076.30 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-09-04 13:00:00 | 5011.00 | 2025-09-05 13:15:00 | 5062.90 | STOP_HIT | 1.00 | -1.04% |
| SELL | retest2 | 2025-09-04 14:45:00 | 5019.00 | 2025-09-05 13:15:00 | 5062.90 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-09-05 10:00:00 | 5011.40 | 2025-09-05 13:15:00 | 5062.90 | STOP_HIT | 1.00 | -1.03% |
| SELL | retest2 | 2025-09-05 11:15:00 | 5019.10 | 2025-09-05 13:15:00 | 5062.90 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-09-16 13:00:00 | 4993.90 | 2025-09-16 14:15:00 | 5046.20 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-09-16 13:30:00 | 5007.70 | 2025-09-16 14:15:00 | 5046.20 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-09-16 14:00:00 | 5009.70 | 2025-09-16 14:15:00 | 5046.20 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest1 | 2025-09-25 12:00:00 | 4765.40 | 2025-09-30 09:15:00 | 4527.13 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-09-25 12:00:00 | 4765.40 | 2025-10-01 15:15:00 | 4479.90 | STOP_HIT | 0.50 | 5.99% |
| BUY | retest2 | 2025-10-09 13:15:00 | 4639.00 | 2025-10-13 11:15:00 | 4606.10 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-10-09 15:15:00 | 4635.80 | 2025-10-13 11:15:00 | 4606.10 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-10-10 10:00:00 | 4638.20 | 2025-10-13 11:15:00 | 4606.10 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-10-10 10:45:00 | 4655.30 | 2025-10-13 11:15:00 | 4606.10 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-11-07 09:15:00 | 4735.70 | 2025-11-17 10:15:00 | 4736.20 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-11-07 10:00:00 | 4769.10 | 2025-11-17 10:15:00 | 4736.20 | STOP_HIT | 1.00 | 0.69% |
| SELL | retest2 | 2025-11-07 11:00:00 | 4757.00 | 2025-11-17 10:15:00 | 4736.20 | STOP_HIT | 1.00 | 0.44% |
| SELL | retest2 | 2025-11-25 13:00:00 | 4463.70 | 2025-12-01 13:15:00 | 4485.30 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2025-11-27 11:00:00 | 4471.30 | 2025-12-01 13:15:00 | 4485.30 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-12-01 09:45:00 | 4446.40 | 2025-12-01 13:15:00 | 4485.30 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-12-04 10:30:00 | 4415.00 | 2025-12-05 15:15:00 | 4462.60 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-12-05 14:30:00 | 4427.00 | 2025-12-05 15:15:00 | 4462.60 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-12-11 09:30:00 | 4324.50 | 2025-12-11 13:15:00 | 4376.10 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-12-11 10:45:00 | 4343.40 | 2025-12-11 13:15:00 | 4376.10 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest1 | 2025-12-18 09:15:00 | 4366.40 | 2025-12-19 10:15:00 | 4389.60 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-12-19 12:15:00 | 4358.60 | 2025-12-29 15:15:00 | 4307.00 | STOP_HIT | 1.00 | 1.18% |
| SELL | retest2 | 2025-12-22 09:30:00 | 4353.00 | 2025-12-29 15:15:00 | 4307.00 | STOP_HIT | 1.00 | 1.06% |
| SELL | retest2 | 2025-12-22 10:30:00 | 4356.20 | 2025-12-29 15:15:00 | 4307.00 | STOP_HIT | 1.00 | 1.13% |
| BUY | retest2 | 2026-01-09 10:45:00 | 4756.00 | 2026-01-14 15:15:00 | 4742.30 | STOP_HIT | 1.00 | -0.29% |
| BUY | retest2 | 2026-01-09 14:45:00 | 4751.00 | 2026-01-16 10:15:00 | 4696.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2026-01-12 12:15:00 | 4751.10 | 2026-01-16 10:15:00 | 4696.00 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2026-01-12 13:30:00 | 4761.40 | 2026-01-16 10:15:00 | 4696.00 | STOP_HIT | 1.00 | -1.37% |
| BUY | retest2 | 2026-01-14 11:30:00 | 4855.90 | 2026-01-16 10:15:00 | 4696.00 | STOP_HIT | 1.00 | -3.29% |
| BUY | retest2 | 2026-02-02 12:15:00 | 4684.70 | 2026-02-04 14:15:00 | 4689.60 | STOP_HIT | 1.00 | 0.10% |
| BUY | retest2 | 2026-02-02 13:00:00 | 4690.30 | 2026-02-04 14:15:00 | 4689.60 | STOP_HIT | 1.00 | -0.01% |
| BUY | retest2 | 2026-02-04 12:00:00 | 4689.70 | 2026-02-04 14:15:00 | 4689.60 | STOP_HIT | 1.00 | -0.00% |
| SELL | retest2 | 2026-02-13 09:15:00 | 4429.80 | 2026-02-17 13:15:00 | 4610.00 | STOP_HIT | 1.00 | -4.07% |
| SELL | retest2 | 2026-02-16 10:30:00 | 4637.60 | 2026-02-17 13:15:00 | 4610.00 | STOP_HIT | 1.00 | 0.60% |
| SELL | retest2 | 2026-02-26 09:15:00 | 4497.20 | 2026-03-02 09:15:00 | 4272.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 09:15:00 | 4497.20 | 2026-03-02 09:15:00 | 4437.20 | STOP_HIT | 0.50 | 1.33% |
| BUY | retest2 | 2026-03-12 12:00:00 | 4296.00 | 2026-03-12 14:15:00 | 4245.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2026-03-18 15:00:00 | 4033.20 | 2026-03-25 10:15:00 | 4050.50 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2026-03-19 09:15:00 | 4019.30 | 2026-03-25 10:15:00 | 4050.50 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2026-03-20 10:30:00 | 4031.20 | 2026-03-25 10:15:00 | 4050.50 | STOP_HIT | 1.00 | -0.48% |
| SELL | retest2 | 2026-04-02 09:15:00 | 3751.50 | 2026-04-08 09:15:00 | 3937.20 | STOP_HIT | 1.00 | -4.95% |
| SELL | retest2 | 2026-04-06 14:30:00 | 3775.20 | 2026-04-08 09:15:00 | 3937.20 | STOP_HIT | 1.00 | -4.29% |
| SELL | retest2 | 2026-04-07 09:30:00 | 3772.80 | 2026-04-08 09:15:00 | 3937.20 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest2 | 2026-04-07 10:45:00 | 3770.30 | 2026-04-08 09:15:00 | 3937.20 | STOP_HIT | 1.00 | -4.43% |
| SELL | retest2 | 2026-04-07 12:15:00 | 3770.40 | 2026-04-08 09:15:00 | 3937.20 | STOP_HIT | 1.00 | -4.42% |
| BUY | retest2 | 2026-04-13 10:45:00 | 4058.80 | 2026-04-24 11:15:00 | 4319.80 | STOP_HIT | 1.00 | 6.43% |
| BUY | retest2 | 2026-04-13 13:45:00 | 4063.50 | 2026-04-24 11:15:00 | 4319.80 | STOP_HIT | 1.00 | 6.31% |
| BUY | retest2 | 2026-04-15 09:15:00 | 4085.00 | 2026-04-24 11:15:00 | 4319.80 | STOP_HIT | 1.00 | 5.75% |
| SELL | retest2 | 2026-04-28 11:15:00 | 4242.00 | 2026-04-29 12:15:00 | 4354.80 | STOP_HIT | 1.00 | -2.66% |
| SELL | retest2 | 2026-04-28 12:00:00 | 4247.50 | 2026-04-29 12:15:00 | 4354.80 | STOP_HIT | 1.00 | -2.53% |
| SELL | retest2 | 2026-04-28 13:30:00 | 4258.70 | 2026-04-29 12:15:00 | 4354.80 | STOP_HIT | 1.00 | -2.26% |
| SELL | retest2 | 2026-05-07 15:00:00 | 4147.10 | 2026-05-08 12:15:00 | 4201.90 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2026-05-08 11:15:00 | 4167.20 | 2026-05-08 12:15:00 | 4201.90 | STOP_HIT | 1.00 | -0.83% |
