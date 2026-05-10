# Trent Ltd. (TRENT)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 4249.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 57 |
| ALERT1 | 40 |
| ALERT2 | 41 |
| ALERT2_SKIP | 18 |
| ALERT3 | 100 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 37 |
| PARTIAL | 10 |
| TARGET_HIT | 1 |
| STOP_HIT | 39 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 50 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 27 / 23
- **Target hits / Stop hits / Partials:** 1 / 39 / 10
- **Avg / median % per leg:** 1.96% / 0.77%
- **Sum % (uncompounded):** 97.77%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 19 | 5 | 26.3% | 1 | 18 | 0 | -0.18% | -3.3% |
| BUY @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | -3.17% | -6.3% |
| BUY @ 3rd Alert (retest2) | 17 | 4 | 23.5% | 1 | 16 | 0 | 0.18% | 3.0% |
| SELL (all) | 31 | 22 | 71.0% | 0 | 21 | 10 | 3.26% | 101.1% |
| SELL @ 2nd Alert (retest1) | 2 | 2 | 100.0% | 0 | 1 | 1 | 4.85% | 9.7% |
| SELL @ 3rd Alert (retest2) | 29 | 20 | 69.0% | 0 | 20 | 9 | 3.15% | 91.4% |
| retest1 (combined) | 4 | 3 | 75.0% | 0 | 3 | 1 | 0.84% | 3.4% |
| retest2 (combined) | 46 | 24 | 52.2% | 1 | 36 | 9 | 2.05% | 94.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 5362.00 | 5218.48 | 5214.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 11:15:00 | 5381.50 | 5251.09 | 5229.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 10:15:00 | 5333.00 | 5347.89 | 5297.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 10:30:00 | 5342.00 | 5347.89 | 5297.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 5381.00 | 5351.51 | 5321.33 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 10:30:00 | 5413.00 | 5363.07 | 5343.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 12:00:00 | 5415.50 | 5373.56 | 5350.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 12:45:00 | 5420.50 | 5389.84 | 5359.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 5457.00 | 5505.82 | 5511.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 5457.00 | 5505.82 | 5511.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-20 13:15:00 | 5457.00 | 5505.82 | 5511.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 5457.00 | 5505.82 | 5511.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 5443.50 | 5493.35 | 5505.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 10:15:00 | 5485.00 | 5479.31 | 5494.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 10:15:00 | 5485.00 | 5479.31 | 5494.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 5485.00 | 5479.31 | 5494.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:30:00 | 5511.50 | 5479.31 | 5494.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 11:15:00 | 5420.00 | 5467.45 | 5487.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-21 13:45:00 | 5405.00 | 5451.39 | 5476.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 5350.00 | 5449.73 | 5471.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 09:30:00 | 5418.00 | 5382.23 | 5414.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 10:15:00 | 5458.50 | 5432.57 | 5429.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-26 10:15:00 | 5458.50 | 5432.57 | 5429.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-26 10:15:00 | 5458.50 | 5432.57 | 5429.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 5458.50 | 5432.57 | 5429.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 11:15:00 | 5495.00 | 5445.06 | 5435.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 10:15:00 | 5548.00 | 5557.48 | 5524.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 10:30:00 | 5537.50 | 5557.48 | 5524.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 09:15:00 | 5601.00 | 5632.53 | 5615.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-02 09:30:00 | 5578.50 | 5632.53 | 5615.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 10:15:00 | 5619.00 | 5629.83 | 5615.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 11:15:00 | 5626.00 | 5629.83 | 5615.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 15:15:00 | 5625.00 | 5622.28 | 5616.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-03 12:15:00 | 5597.00 | 5610.57 | 5612.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-03 12:15:00 | 5597.00 | 5610.57 | 5612.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-06-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 12:15:00 | 5597.00 | 5610.57 | 5612.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 15:15:00 | 5577.50 | 5601.45 | 5607.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 09:15:00 | 5649.00 | 5553.28 | 5569.88 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 09:15:00 | 5649.00 | 5553.28 | 5569.88 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 5649.00 | 5553.28 | 5569.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:00:00 | 5649.00 | 5553.28 | 5569.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 5631.50 | 5568.92 | 5575.48 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 5679.00 | 5590.94 | 5584.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 12:15:00 | 5703.50 | 5613.45 | 5595.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 5876.00 | 5876.88 | 5806.67 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 11:00:00 | 5876.00 | 5876.88 | 5806.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 14:15:00 | 5816.50 | 5848.93 | 5814.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-10 15:00:00 | 5816.50 | 5848.93 | 5814.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 15:15:00 | 5812.50 | 5841.64 | 5814.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 09:15:00 | 5825.50 | 5841.64 | 5814.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 5794.50 | 5832.21 | 5812.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:00:00 | 5794.50 | 5832.21 | 5812.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 5805.00 | 5826.77 | 5811.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:30:00 | 5818.50 | 5826.77 | 5811.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 5796.00 | 5820.62 | 5810.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:00:00 | 5796.00 | 5820.62 | 5810.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 5789.00 | 5814.29 | 5808.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:45:00 | 5791.00 | 5814.29 | 5808.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-06-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 14:15:00 | 5776.00 | 5799.63 | 5802.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 5725.50 | 5777.15 | 5790.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 15:15:00 | 5610.00 | 5597.21 | 5655.54 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 09:15:00 | 5640.50 | 5597.21 | 5655.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 09:15:00 | 5610.00 | 5599.77 | 5651.40 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2025-06-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 10:15:00 | 5731.50 | 5671.20 | 5663.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-20 09:15:00 | 5781.00 | 5722.61 | 5704.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 6063.50 | 6082.73 | 5989.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 14:00:00 | 6063.50 | 6082.73 | 5989.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 6085.50 | 6101.58 | 6058.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 09:45:00 | 6096.00 | 6101.58 | 6058.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 6091.00 | 6099.46 | 6061.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:45:00 | 6073.50 | 6099.46 | 6061.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 6070.50 | 6091.84 | 6064.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 12:45:00 | 6016.00 | 6091.84 | 6064.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 6074.50 | 6094.89 | 6076.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 11:00:00 | 6074.50 | 6094.89 | 6076.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 11:15:00 | 6063.00 | 6088.51 | 6075.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 12:15:00 | 6054.00 | 6088.51 | 6075.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 6051.00 | 6081.01 | 6073.44 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2025-06-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-27 14:15:00 | 6036.50 | 6066.75 | 6067.94 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2025-06-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-30 13:15:00 | 6221.50 | 6079.56 | 6068.91 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 09:15:00 | 5758.00 | 6099.62 | 6128.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 5573.00 | 5928.68 | 6041.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 5481.50 | 5448.63 | 5544.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-09 09:45:00 | 5492.00 | 5448.63 | 5544.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 5361.00 | 5339.31 | 5358.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 13:00:00 | 5361.00 | 5339.31 | 5358.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 5357.50 | 5342.95 | 5358.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 14:15:00 | 5368.00 | 5342.95 | 5358.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 14:15:00 | 5388.00 | 5351.96 | 5360.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 15:00:00 | 5388.00 | 5351.96 | 5360.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 15:15:00 | 5396.00 | 5360.77 | 5364.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-16 09:15:00 | 5450.00 | 5360.77 | 5364.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2025-07-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-16 09:15:00 | 5412.50 | 5371.11 | 5368.56 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 15:15:00 | 5369.50 | 5393.18 | 5395.60 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2025-07-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 10:15:00 | 5411.50 | 5390.89 | 5390.44 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 12:15:00 | 5380.00 | 5389.13 | 5389.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 14:15:00 | 5363.50 | 5382.54 | 5386.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 15:15:00 | 5369.50 | 5363.41 | 5371.07 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-07-24 09:15:00 | 5282.00 | 5363.41 | 5371.07 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-28 14:15:00 | 5017.90 | 5055.68 | 5119.04 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-07-30 09:15:00 | 5034.00 | 5016.68 | 5055.15 | SL hit (close>ema200) qty=0.50 sl=5016.68 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 5037.50 | 5020.85 | 5053.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:45:00 | 5056.00 | 5020.85 | 5053.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 5046.00 | 5025.88 | 5052.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:30:00 | 5060.50 | 5025.88 | 5052.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 5032.00 | 5027.10 | 5050.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 09:15:00 | 5017.50 | 5034.17 | 5048.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-31 12:15:00 | 5061.00 | 5034.78 | 5043.32 | SL hit (close>static) qty=1.00 sl=5055.00 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-31 14:45:00 | 5020.00 | 5036.78 | 5043.04 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-01 09:15:00 | 5012.00 | 5034.22 | 5041.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 5103.00 | 5047.98 | 5046.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 5103.00 | 5047.98 | 5046.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 09:15:00 | 5103.00 | 5047.98 | 5046.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 10:15:00 | 5108.00 | 5059.98 | 5052.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 09:15:00 | 5208.00 | 5224.15 | 5178.00 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-05 10:00:00 | 5208.00 | 5224.15 | 5178.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 09:15:00 | 5398.00 | 5359.50 | 5310.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-11 13:30:00 | 5469.50 | 5398.41 | 5359.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-14 14:15:00 | 5374.50 | 5394.17 | 5394.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 16 — SELL (started 2025-08-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 14:15:00 | 5374.50 | 5394.17 | 5394.88 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 5563.50 | 5422.89 | 5407.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-18 10:15:00 | 5590.00 | 5456.31 | 5424.08 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-18 14:15:00 | 5505.00 | 5521.59 | 5470.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 15:00:00 | 5505.00 | 5521.59 | 5470.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 5503.00 | 5512.34 | 5478.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:30:00 | 5473.50 | 5512.34 | 5478.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 13:15:00 | 5479.50 | 5500.19 | 5481.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 13:45:00 | 5480.50 | 5500.19 | 5481.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 14:15:00 | 5503.50 | 5500.85 | 5483.23 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-08-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 11:15:00 | 5440.00 | 5474.52 | 5475.43 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 5480.00 | 5472.12 | 5471.96 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 14:15:00 | 5456.00 | 5471.00 | 5471.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 15:15:00 | 5443.00 | 5465.40 | 5469.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 09:15:00 | 5471.00 | 5466.52 | 5469.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-22 09:15:00 | 5471.00 | 5466.52 | 5469.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 09:15:00 | 5471.00 | 5466.52 | 5469.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:00:00 | 5471.00 | 5466.52 | 5469.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 5454.00 | 5464.01 | 5468.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-22 12:45:00 | 5445.50 | 5457.57 | 5464.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 14:30:00 | 5444.50 | 5436.00 | 5444.95 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 5395.00 | 5331.92 | 5331.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 10:15:00 | 5395.00 | 5331.92 | 5331.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — BUY (started 2025-09-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 10:15:00 | 5395.00 | 5331.92 | 5331.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 12:15:00 | 5427.50 | 5362.57 | 5346.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 5397.50 | 5423.09 | 5395.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 13:15:00 | 5397.50 | 5423.09 | 5395.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 5397.50 | 5423.09 | 5395.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 5397.50 | 5423.09 | 5395.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 5413.50 | 5421.17 | 5397.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 14:45:00 | 5446.50 | 5412.41 | 5400.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-08 13:15:00 | 5395.00 | 5487.00 | 5494.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 5395.00 | 5487.00 | 5494.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 14:15:00 | 5320.50 | 5453.70 | 5478.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 09:15:00 | 5188.00 | 5180.65 | 5222.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 09:30:00 | 5190.00 | 5180.65 | 5222.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 09:15:00 | 5175.00 | 5145.47 | 5153.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-17 09:30:00 | 5199.50 | 5145.47 | 5153.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 23 — BUY (started 2025-09-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 11:15:00 | 5222.00 | 5169.82 | 5163.67 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-09-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-18 13:15:00 | 5144.00 | 5173.28 | 5173.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-19 09:15:00 | 5119.50 | 5154.51 | 5164.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-22 09:15:00 | 5120.00 | 5111.84 | 5133.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 09:15:00 | 5120.00 | 5111.84 | 5133.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 09:15:00 | 5120.00 | 5111.84 | 5133.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-22 10:00:00 | 5120.00 | 5111.84 | 5133.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 4762.70 | 4712.71 | 4732.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:30:00 | 4767.20 | 4712.71 | 4732.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 4829.80 | 4736.12 | 4741.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 4829.80 | 4736.12 | 4741.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 11:15:00 | 4830.00 | 4754.90 | 4749.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 12:15:00 | 4843.80 | 4772.68 | 4757.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 10:15:00 | 4795.80 | 4804.01 | 4782.35 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-03 11:00:00 | 4795.80 | 4804.01 | 4782.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 4799.30 | 4803.07 | 4783.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 11:30:00 | 4782.20 | 4803.07 | 4783.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 4775.40 | 4797.53 | 4783.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 13:00:00 | 4775.40 | 4797.53 | 4783.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 13:15:00 | 4775.00 | 4793.03 | 4782.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 14:15:00 | 4789.50 | 4793.03 | 4782.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-06 15:15:00 | 4705.00 | 4793.06 | 4793.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-10-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-06 15:15:00 | 4705.00 | 4793.06 | 4793.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 09:15:00 | 4675.80 | 4769.60 | 4782.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 09:15:00 | 4712.90 | 4707.99 | 4737.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-08 09:15:00 | 4712.90 | 4707.99 | 4737.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 4712.90 | 4707.99 | 4737.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-08 09:45:00 | 4725.90 | 4707.99 | 4737.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 4663.10 | 4648.50 | 4669.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 4697.00 | 4648.50 | 4669.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 4680.90 | 4654.98 | 4670.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:30:00 | 4689.00 | 4654.98 | 4670.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 4708.80 | 4665.75 | 4673.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 11:00:00 | 4708.80 | 4665.75 | 4673.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2025-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 12:15:00 | 4711.70 | 4681.38 | 4679.84 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 4646.50 | 4677.53 | 4679.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 4613.00 | 4649.01 | 4661.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 4664.30 | 4640.03 | 4652.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 09:15:00 | 4664.30 | 4640.03 | 4652.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 09:15:00 | 4664.30 | 4640.03 | 4652.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 09:45:00 | 4658.70 | 4640.03 | 4652.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 4658.40 | 4643.70 | 4652.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 4679.00 | 4643.70 | 4652.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 4618.50 | 4638.66 | 4649.68 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2025-10-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 15:15:00 | 4739.00 | 4667.21 | 4659.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 09:15:00 | 4764.00 | 4686.57 | 4669.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 15:15:00 | 4795.00 | 4800.67 | 4763.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 09:15:00 | 4805.00 | 4800.67 | 4763.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 4779.00 | 4797.48 | 4783.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 4779.00 | 4797.48 | 4783.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 4792.00 | 4804.26 | 4793.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 14:00:00 | 4792.00 | 4804.26 | 4793.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 4776.40 | 4798.69 | 4791.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 4776.40 | 4798.69 | 4791.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 4787.00 | 4796.35 | 4791.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 09:15:00 | 4795.10 | 4796.35 | 4791.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 12:00:00 | 4791.10 | 4790.35 | 4789.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 15:00:00 | 4789.70 | 4790.20 | 4789.68 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 15:15:00 | 4782.20 | 4788.60 | 4789.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-24 15:15:00 | 4782.20 | 4788.60 | 4789.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-24 15:15:00 | 4782.20 | 4788.60 | 4789.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2025-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 15:15:00 | 4782.20 | 4788.60 | 4789.00 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 4805.50 | 4791.98 | 4790.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 11:15:00 | 4825.00 | 4803.12 | 4796.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-27 12:15:00 | 4802.00 | 4802.89 | 4796.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-27 13:00:00 | 4802.00 | 4802.89 | 4796.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 13:15:00 | 4802.40 | 4802.79 | 4797.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:30:00 | 4813.30 | 4799.19 | 4796.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-28 10:15:00 | 4773.50 | 4794.06 | 4794.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 4773.50 | 4794.06 | 4794.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 11:15:00 | 4751.90 | 4785.62 | 4790.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 11:15:00 | 4756.90 | 4752.22 | 4766.64 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 12:00:00 | 4756.90 | 4752.22 | 4766.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 12:15:00 | 4763.80 | 4754.54 | 4766.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 13:00:00 | 4763.80 | 4754.54 | 4766.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 13:15:00 | 4776.00 | 4758.83 | 4767.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:00:00 | 4776.00 | 4758.83 | 4767.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 14:15:00 | 4778.20 | 4762.70 | 4768.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-29 14:45:00 | 4781.70 | 4762.70 | 4768.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 4710.60 | 4700.89 | 4715.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 15:00:00 | 4710.60 | 4700.89 | 4715.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 4714.00 | 4703.51 | 4715.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 4695.40 | 4703.51 | 4715.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 4671.30 | 4697.07 | 4711.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:15:00 | 4665.00 | 4691.34 | 4707.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 13:00:00 | 4665.10 | 4682.68 | 4700.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:00:00 | 4665.00 | 4669.31 | 4687.38 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 09:15:00 | 4656.30 | 4671.63 | 4680.30 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 4671.20 | 4671.55 | 4679.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 13:30:00 | 4637.00 | 4659.60 | 4671.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-07 14:30:00 | 4635.40 | 4652.92 | 4667.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:15:00 | 4431.75 | 4580.15 | 4631.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:15:00 | 4431.85 | 4580.15 | 4631.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:15:00 | 4431.75 | 4580.15 | 4631.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:15:00 | 4423.48 | 4580.15 | 4631.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:15:00 | 4405.15 | 4580.15 | 4631.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 09:15:00 | 4403.63 | 4580.15 | 4631.15 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 4332.00 | 4328.33 | 4395.94 | SL hit (close>ema200) qty=0.50 sl=4328.33 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 4332.00 | 4328.33 | 4395.94 | SL hit (close>ema200) qty=0.50 sl=4328.33 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 4332.00 | 4328.33 | 4395.94 | SL hit (close>ema200) qty=0.50 sl=4328.33 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 4332.00 | 4328.33 | 4395.94 | SL hit (close>ema200) qty=0.50 sl=4328.33 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 4332.00 | 4328.33 | 4395.94 | SL hit (close>ema200) qty=0.50 sl=4328.33 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 4332.00 | 4328.33 | 4395.94 | SL hit (close>ema200) qty=0.50 sl=4328.33 alert=retest2 |

### Cycle 33 — BUY (started 2025-11-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 09:15:00 | 4400.00 | 4377.74 | 4376.31 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-11-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 15:15:00 | 4369.00 | 4384.97 | 4385.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 4362.30 | 4380.44 | 4383.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 12:15:00 | 4370.10 | 4369.91 | 4377.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-19 13:00:00 | 4370.10 | 4369.91 | 4377.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 09:15:00 | 4354.50 | 4361.06 | 4370.09 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-11-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 13:15:00 | 4406.00 | 4376.35 | 4374.79 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 4359.30 | 4375.63 | 4376.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 09:15:00 | 4335.10 | 4365.66 | 4371.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 4347.30 | 4292.46 | 4312.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 4347.30 | 4292.46 | 4312.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 4347.30 | 4292.46 | 4312.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-26 10:00:00 | 4347.30 | 4292.46 | 4312.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 10:15:00 | 4321.90 | 4298.35 | 4313.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-26 12:45:00 | 4307.10 | 4306.75 | 4314.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 4091.75 | 4146.94 | 4172.69 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 4095.40 | 4091.37 | 4116.77 | SL hit (close>ema200) qty=0.50 sl=4091.37 alert=retest2 |

### Cycle 37 — BUY (started 2025-12-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 14:15:00 | 4082.90 | 4063.36 | 4062.89 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-15 09:15:00 | 4051.50 | 4061.57 | 4062.19 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-12-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 10:15:00 | 4074.30 | 4064.12 | 4063.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 12:15:00 | 4087.50 | 4068.50 | 4065.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-17 09:15:00 | 4083.70 | 4102.25 | 4092.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 09:15:00 | 4083.70 | 4102.25 | 4092.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 4083.70 | 4102.25 | 4092.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:45:00 | 4075.10 | 4102.25 | 4092.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 10:15:00 | 4074.00 | 4096.60 | 4090.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 10:45:00 | 4070.70 | 4096.60 | 4090.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-12-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 12:15:00 | 4053.40 | 4082.90 | 4085.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 13:15:00 | 4044.40 | 4075.20 | 4081.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 15:15:00 | 4034.00 | 4031.62 | 4048.95 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 09:15:00 | 4062.00 | 4031.62 | 4048.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 4037.00 | 4032.69 | 4047.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-19 11:15:00 | 4026.50 | 4032.75 | 4046.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-19 15:15:00 | 4063.90 | 4052.49 | 4052.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 15:15:00 | 4063.90 | 4052.49 | 4052.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 09:15:00 | 4128.10 | 4067.61 | 4059.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 15:15:00 | 4183.00 | 4186.49 | 4152.66 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-24 09:15:00 | 4222.00 | 4186.49 | 4152.66 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 4246.20 | 4271.13 | 4251.52 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-29 11:15:00 | 4246.20 | 4271.13 | 4251.52 | SL hit (close<ema400) qty=1.00 sl=4251.52 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-12-29 11:45:00 | 4244.40 | 4271.13 | 4251.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 4220.00 | 4260.90 | 4248.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 13:00:00 | 4220.00 | 4260.90 | 4248.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 4235.00 | 4255.72 | 4247.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 09:45:00 | 4250.70 | 4243.52 | 4243.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-30 10:15:00 | 4247.00 | 4243.52 | 4243.12 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 10:15:00 | 4231.50 | 4241.12 | 4242.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-30 10:15:00 | 4231.50 | 4241.12 | 4242.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 10:15:00 | 4231.50 | 4241.12 | 4242.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-30 11:15:00 | 4199.00 | 4232.70 | 4238.15 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-31 09:15:00 | 4246.30 | 4224.75 | 4230.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-31 09:15:00 | 4246.30 | 4224.75 | 4230.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 4246.30 | 4224.75 | 4230.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 10:15:00 | 4253.00 | 4224.75 | 4230.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 10:15:00 | 4270.00 | 4233.80 | 4234.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 11:00:00 | 4270.00 | 4233.80 | 4234.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — BUY (started 2025-12-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 11:15:00 | 4283.80 | 4243.80 | 4238.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 10:15:00 | 4304.30 | 4269.32 | 4255.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 09:15:00 | 4378.40 | 4378.53 | 4336.82 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-01-05 11:00:00 | 4423.50 | 4387.52 | 4344.70 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-06 09:15:00 | 4117.50 | 4355.98 | 4353.72 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-01-06 09:15:00 | 4117.50 | 4355.98 | 4353.72 | SL hit (close<ema400) qty=1.00 sl=4353.72 alert=retest1 |

### Cycle 44 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 4100.00 | 4304.78 | 4330.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 13:15:00 | 4057.70 | 4194.41 | 4268.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 09:15:00 | 4015.00 | 3994.31 | 4027.92 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 09:15:00 | 4015.00 | 3994.31 | 4027.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 09:15:00 | 4015.00 | 3994.31 | 4027.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 10:00:00 | 4015.00 | 3994.31 | 4027.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 10:15:00 | 4008.90 | 3997.23 | 4026.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 12:00:00 | 3977.40 | 4012.52 | 4023.92 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 09:15:00 | 3778.53 | 3844.19 | 3882.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 3821.10 | 3790.92 | 3828.68 | SL hit (close>ema200) qty=0.50 sl=3790.92 alert=retest2 |

### Cycle 45 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 3841.70 | 3796.69 | 3791.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 3851.80 | 3807.71 | 3797.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 3775.90 | 3823.35 | 3811.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 3775.90 | 3823.35 | 3811.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 3775.90 | 3823.35 | 3811.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:00:00 | 3775.90 | 3823.35 | 3811.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 3775.60 | 3813.80 | 3807.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 3766.10 | 3813.80 | 3807.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 12:15:00 | 3779.00 | 3801.35 | 3802.96 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 13:15:00 | 3819.40 | 3804.96 | 3804.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-29 14:15:00 | 3824.30 | 3808.83 | 3806.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 3801.60 | 3809.97 | 3807.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 3801.60 | 3809.97 | 3807.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 3801.60 | 3809.97 | 3807.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-30 10:15:00 | 3792.60 | 3809.97 | 3807.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2026-01-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 10:15:00 | 3780.00 | 3803.98 | 3804.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 12:15:00 | 3763.50 | 3791.89 | 3798.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-01 11:15:00 | 3783.50 | 3782.32 | 3790.08 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 3783.50 | 3782.32 | 3790.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 3783.50 | 3782.32 | 3790.08 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 3724.90 | 3782.32 | 3790.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 10:15:00 | 3805.60 | 3742.23 | 3744.94 | SL hit (close>static) qty=1.00 sl=3804.90 alert=retest2 |

### Cycle 49 — BUY (started 2026-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 11:15:00 | 3806.00 | 3754.98 | 3750.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 14:15:00 | 3822.70 | 3784.07 | 3766.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-06 09:15:00 | 4077.00 | 4078.94 | 3994.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 10:00:00 | 4077.00 | 4078.94 | 3994.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 4138.30 | 4166.32 | 4139.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 4138.30 | 4166.32 | 4139.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 10:15:00 | 4163.00 | 4165.66 | 4142.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:30:00 | 4136.00 | 4165.66 | 4142.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 4214.50 | 4249.17 | 4216.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 09:45:00 | 4198.00 | 4249.17 | 4216.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 4226.50 | 4244.63 | 4217.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-13 11:30:00 | 4262.30 | 4253.71 | 4224.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 14:30:00 | 4240.00 | 4243.11 | 4238.64 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 4218.50 | 4233.69 | 4234.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-17 09:15:00 | 4218.50 | 4233.69 | 4234.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-02-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-17 09:15:00 | 4218.50 | 4233.69 | 4234.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-17 10:15:00 | 4184.00 | 4223.75 | 4230.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 14:15:00 | 4187.80 | 4173.15 | 4188.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 14:15:00 | 4187.80 | 4173.15 | 4188.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 14:15:00 | 4187.80 | 4173.15 | 4188.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 15:00:00 | 4187.80 | 4173.15 | 4188.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 15:15:00 | 4181.90 | 4174.90 | 4188.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 09:15:00 | 4165.00 | 4174.90 | 4188.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-24 13:15:00 | 3956.75 | 4002.85 | 4044.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-27 11:15:00 | 3888.80 | 3880.89 | 3914.46 | SL hit (close>ema200) qty=0.50 sl=3880.89 alert=retest2 |

### Cycle 51 — BUY (started 2026-03-17 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 11:15:00 | 3593.30 | 3556.65 | 3554.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 12:15:00 | 3649.80 | 3575.28 | 3562.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 14:15:00 | 3645.00 | 3646.70 | 3618.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-18 14:30:00 | 3638.00 | 3646.70 | 3618.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 3538.30 | 3622.99 | 3612.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:00:00 | 3538.30 | 3622.99 | 3612.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 3520.00 | 3602.39 | 3603.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 3514.70 | 3574.87 | 3590.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 3559.50 | 3538.89 | 3565.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 10:00:00 | 3559.50 | 3538.89 | 3565.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 10:15:00 | 3573.00 | 3545.71 | 3565.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:00:00 | 3573.00 | 3545.71 | 3565.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 11:15:00 | 3587.60 | 3554.09 | 3567.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 11:30:00 | 3589.70 | 3554.09 | 3567.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 13:15:00 | 3566.70 | 3558.52 | 3567.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 14:00:00 | 3566.70 | 3558.52 | 3567.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 3563.00 | 3559.41 | 3567.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:15:00 | 3572.00 | 3559.41 | 3567.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 15:15:00 | 3572.00 | 3561.93 | 3567.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-23 09:15:00 | 3482.90 | 3561.93 | 3567.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-25 12:15:00 | 3524.70 | 3460.72 | 3454.32 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 3524.70 | 3460.72 | 3454.32 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 3407.60 | 3450.62 | 3455.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 09:15:00 | 3342.40 | 3409.54 | 3432.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 3511.20 | 3372.48 | 3393.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 3511.20 | 3372.48 | 3393.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 3511.20 | 3372.48 | 3393.53 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 3489.00 | 3411.23 | 3408.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 3497.10 | 3428.40 | 3416.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 3414.80 | 3456.12 | 3436.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 09:15:00 | 3414.80 | 3456.12 | 3436.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 09:15:00 | 3414.80 | 3456.12 | 3436.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:15:00 | 3414.90 | 3456.12 | 3436.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 3419.20 | 3448.74 | 3434.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 11:15:00 | 3435.80 | 3448.74 | 3434.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-06 09:15:00 | 3779.38 | 3554.24 | 3493.98 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-04-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 15:15:00 | 4244.00 | 4315.67 | 4317.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-29 15:15:00 | 4221.10 | 4246.86 | 4258.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 14:15:00 | 4157.80 | 4139.97 | 4171.29 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-04 15:00:00 | 4157.80 | 4139.97 | 4171.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 15:15:00 | 4144.90 | 4129.33 | 4146.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 09:15:00 | 4205.30 | 4129.33 | 4146.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 09:15:00 | 4266.40 | 4156.74 | 4157.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:00:00 | 4266.40 | 4156.74 | 4157.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-05-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 10:15:00 | 4250.00 | 4175.39 | 4166.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 13:15:00 | 4296.10 | 4220.96 | 4191.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 14:15:00 | 4291.10 | 4291.43 | 4252.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-07 15:00:00 | 4291.10 | 4291.43 | 4252.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 11:15:00 | 4253.50 | 4285.78 | 4262.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 11:30:00 | 4259.20 | 4285.78 | 4262.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 12:15:00 | 4230.60 | 4274.75 | 4259.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 13:00:00 | 4230.60 | 4274.75 | 4259.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 15:15:00 | 4249.10 | 4257.94 | 4254.56 | EMA400 retest candle locked (from upside) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-15 10:30:00 | 5413.00 | 2025-05-20 13:15:00 | 5457.00 | STOP_HIT | 1.00 | 0.81% |
| BUY | retest2 | 2025-05-15 12:00:00 | 5415.50 | 2025-05-20 13:15:00 | 5457.00 | STOP_HIT | 1.00 | 0.77% |
| BUY | retest2 | 2025-05-15 12:45:00 | 5420.50 | 2025-05-20 13:15:00 | 5457.00 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2025-05-21 13:45:00 | 5405.00 | 2025-05-26 10:15:00 | 5458.50 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-05-22 09:15:00 | 5350.00 | 2025-05-26 10:15:00 | 5458.50 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-05-23 09:30:00 | 5418.00 | 2025-05-26 10:15:00 | 5458.50 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-06-02 11:15:00 | 5626.00 | 2025-06-03 12:15:00 | 5597.00 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-06-02 15:15:00 | 5625.00 | 2025-06-03 12:15:00 | 5597.00 | STOP_HIT | 1.00 | -0.50% |
| SELL | retest1 | 2025-07-24 09:15:00 | 5282.00 | 2025-07-28 14:15:00 | 5017.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest1 | 2025-07-24 09:15:00 | 5282.00 | 2025-07-30 09:15:00 | 5034.00 | STOP_HIT | 0.50 | 4.70% |
| SELL | retest2 | 2025-07-31 09:15:00 | 5017.50 | 2025-07-31 12:15:00 | 5061.00 | STOP_HIT | 1.00 | -0.87% |
| SELL | retest2 | 2025-07-31 14:45:00 | 5020.00 | 2025-08-01 09:15:00 | 5103.00 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-08-01 09:15:00 | 5012.00 | 2025-08-01 09:15:00 | 5103.00 | STOP_HIT | 1.00 | -1.82% |
| BUY | retest2 | 2025-08-11 13:30:00 | 5469.50 | 2025-08-14 14:15:00 | 5374.50 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2025-08-22 12:45:00 | 5445.50 | 2025-09-01 10:15:00 | 5395.00 | STOP_HIT | 1.00 | 0.93% |
| SELL | retest2 | 2025-08-25 14:30:00 | 5444.50 | 2025-09-01 10:15:00 | 5395.00 | STOP_HIT | 1.00 | 0.91% |
| BUY | retest2 | 2025-09-03 14:45:00 | 5446.50 | 2025-09-08 13:15:00 | 5395.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-10-03 14:15:00 | 4789.50 | 2025-10-06 15:15:00 | 4705.00 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-10-24 09:15:00 | 4795.10 | 2025-10-24 15:15:00 | 4782.20 | STOP_HIT | 1.00 | -0.27% |
| BUY | retest2 | 2025-10-24 12:00:00 | 4791.10 | 2025-10-24 15:15:00 | 4782.20 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2025-10-24 15:00:00 | 4789.70 | 2025-10-24 15:15:00 | 4782.20 | STOP_HIT | 1.00 | -0.16% |
| BUY | retest2 | 2025-10-28 09:30:00 | 4813.30 | 2025-10-28 10:15:00 | 4773.50 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-11-04 11:15:00 | 4665.00 | 2025-11-10 09:15:00 | 4431.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 13:00:00 | 4665.10 | 2025-11-10 09:15:00 | 4431.85 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 10:00:00 | 4665.00 | 2025-11-10 09:15:00 | 4431.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-07 09:15:00 | 4656.30 | 2025-11-10 09:15:00 | 4423.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-07 13:30:00 | 4637.00 | 2025-11-10 09:15:00 | 4405.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-07 14:30:00 | 4635.40 | 2025-11-10 09:15:00 | 4403.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-04 11:15:00 | 4665.00 | 2025-11-12 09:15:00 | 4332.00 | STOP_HIT | 0.50 | 7.14% |
| SELL | retest2 | 2025-11-04 13:00:00 | 4665.10 | 2025-11-12 09:15:00 | 4332.00 | STOP_HIT | 0.50 | 7.14% |
| SELL | retest2 | 2025-11-06 10:00:00 | 4665.00 | 2025-11-12 09:15:00 | 4332.00 | STOP_HIT | 0.50 | 7.14% |
| SELL | retest2 | 2025-11-07 09:15:00 | 4656.30 | 2025-11-12 09:15:00 | 4332.00 | STOP_HIT | 0.50 | 6.96% |
| SELL | retest2 | 2025-11-07 13:30:00 | 4637.00 | 2025-11-12 09:15:00 | 4332.00 | STOP_HIT | 0.50 | 6.58% |
| SELL | retest2 | 2025-11-07 14:30:00 | 4635.40 | 2025-11-12 09:15:00 | 4332.00 | STOP_HIT | 0.50 | 6.55% |
| SELL | retest2 | 2025-11-26 12:45:00 | 4307.10 | 2025-12-08 13:15:00 | 4091.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-26 12:45:00 | 4307.10 | 2025-12-10 09:15:00 | 4095.40 | STOP_HIT | 0.50 | 4.92% |
| SELL | retest2 | 2025-12-19 11:15:00 | 4026.50 | 2025-12-19 15:15:00 | 4063.90 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest1 | 2025-12-24 09:15:00 | 4222.00 | 2025-12-29 11:15:00 | 4246.20 | STOP_HIT | 1.00 | 0.57% |
| BUY | retest2 | 2025-12-30 09:45:00 | 4250.70 | 2025-12-30 10:15:00 | 4231.50 | STOP_HIT | 1.00 | -0.45% |
| BUY | retest2 | 2025-12-30 10:15:00 | 4247.00 | 2025-12-30 10:15:00 | 4231.50 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest1 | 2026-01-05 11:00:00 | 4423.50 | 2026-01-06 09:15:00 | 4117.50 | STOP_HIT | 1.00 | -6.92% |
| SELL | retest2 | 2026-01-13 12:00:00 | 3977.40 | 2026-01-21 09:15:00 | 3778.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 12:00:00 | 3977.40 | 2026-01-22 09:15:00 | 3821.10 | STOP_HIT | 0.50 | 3.93% |
| SELL | retest2 | 2026-02-01 12:15:00 | 3724.90 | 2026-02-03 10:15:00 | 3805.60 | STOP_HIT | 1.00 | -2.17% |
| BUY | retest2 | 2026-02-13 11:30:00 | 4262.30 | 2026-02-17 09:15:00 | 4218.50 | STOP_HIT | 1.00 | -1.03% |
| BUY | retest2 | 2026-02-16 14:30:00 | 4240.00 | 2026-02-17 09:15:00 | 4218.50 | STOP_HIT | 1.00 | -0.51% |
| SELL | retest2 | 2026-02-19 09:15:00 | 4165.00 | 2026-02-24 13:15:00 | 3956.75 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 09:15:00 | 4165.00 | 2026-02-27 11:15:00 | 3888.80 | STOP_HIT | 0.50 | 6.63% |
| SELL | retest2 | 2026-03-23 09:15:00 | 3482.90 | 2026-03-25 12:15:00 | 3524.70 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-04-02 11:15:00 | 3435.80 | 2026-04-06 09:15:00 | 3779.38 | TARGET_HIT | 1.00 | 10.00% |
