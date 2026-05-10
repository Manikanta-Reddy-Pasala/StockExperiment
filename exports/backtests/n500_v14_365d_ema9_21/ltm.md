# LTM Ltd. (LTM)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 4360.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 78 |
| ALERT1 | 53 |
| ALERT2 | 52 |
| ALERT2_SKIP | 28 |
| ALERT3 | 127 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 56 |
| PARTIAL | 3 |
| TARGET_HIT | 3 |
| STOP_HIT | 59 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 63 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 11 / 52
- **Target hits / Stop hits / Partials:** 2 / 58 / 3
- **Avg / median % per leg:** -0.53% / -0.99%
- **Sum % (uncompounded):** -33.27%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 38 | 8 | 21.1% | 1 | 35 | 2 | -0.73% | -27.7% |
| BUY @ 2nd Alert (retest1) | 5 | 5 | 100.0% | 0 | 3 | 2 | 4.07% | 20.4% |
| BUY @ 3rd Alert (retest2) | 33 | 3 | 9.1% | 1 | 32 | 0 | -1.46% | -48.0% |
| SELL (all) | 25 | 3 | 12.0% | 1 | 23 | 1 | -0.22% | -5.6% |
| SELL @ 2nd Alert (retest1) | 1 | 1 | 100.0% | 0 | 1 | 0 | 1.38% | 1.4% |
| SELL @ 3rd Alert (retest2) | 24 | 2 | 8.3% | 1 | 22 | 1 | -0.29% | -7.0% |
| retest1 (combined) | 6 | 6 | 100.0% | 0 | 4 | 2 | 3.62% | 21.7% |
| retest2 (combined) | 57 | 5 | 8.8% | 2 | 54 | 1 | -0.96% | -55.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 15:15:00 | 5020.80 | 5033.01 | 5034.34 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 5070.40 | 5040.48 | 5037.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 5087.90 | 5059.61 | 5051.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 5082.20 | 5098.51 | 5080.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 5082.20 | 5098.51 | 5080.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 5082.20 | 5098.51 | 5080.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 5082.20 | 5098.51 | 5080.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 5077.80 | 5094.37 | 5080.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 5097.30 | 5094.37 | 5080.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 14:45:00 | 5109.40 | 5088.59 | 5081.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 11:30:00 | 5089.20 | 5095.72 | 5087.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 11:15:00 | 5070.60 | 5109.60 | 5113.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 11:15:00 | 5070.60 | 5109.60 | 5113.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-30 11:15:00 | 5070.60 | 5109.60 | 5113.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-05-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 11:15:00 | 5070.60 | 5109.60 | 5113.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 09:15:00 | 5044.50 | 5076.68 | 5094.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-02 12:15:00 | 5080.50 | 5075.47 | 5089.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-02 12:30:00 | 5087.00 | 5075.47 | 5089.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 13:15:00 | 5085.00 | 5077.38 | 5088.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-02 13:45:00 | 5080.00 | 5077.38 | 5088.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-02 14:15:00 | 5075.00 | 5076.90 | 5087.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-03 09:30:00 | 5032.00 | 5067.82 | 5081.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 09:15:00 | 5136.00 | 5076.35 | 5076.83 | SL hit (close>static) qty=1.00 sl=5122.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-06-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 10:15:00 | 5143.50 | 5089.78 | 5082.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 10:15:00 | 5177.00 | 5140.59 | 5116.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 5170.00 | 5170.69 | 5147.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 10:45:00 | 5176.50 | 5170.69 | 5147.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 5434.50 | 5419.88 | 5387.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 10:15:00 | 5449.00 | 5419.88 | 5387.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 10:15:00 | 5457.00 | 5431.42 | 5411.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-16 12:15:00 | 5440.00 | 5435.09 | 5416.55 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-18 13:30:00 | 5449.50 | 5479.62 | 5471.85 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-19 09:15:00 | 5314.50 | 5439.25 | 5454.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-19 09:15:00 | 5314.50 | 5439.25 | 5454.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-19 09:15:00 | 5314.50 | 5439.25 | 5454.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-19 09:15:00 | 5314.50 | 5439.25 | 5454.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 5 — SELL (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 09:15:00 | 5314.50 | 5439.25 | 5454.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-19 10:15:00 | 5273.00 | 5406.00 | 5438.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 11:15:00 | 5366.00 | 5349.91 | 5386.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-20 12:00:00 | 5366.00 | 5349.91 | 5386.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 5377.00 | 5350.39 | 5377.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 5377.00 | 5350.39 | 5377.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 5362.00 | 5352.71 | 5375.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:45:00 | 5340.00 | 5350.17 | 5372.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 13:30:00 | 5356.00 | 5362.21 | 5364.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 14:00:00 | 5356.50 | 5362.21 | 5364.13 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-24 14:45:00 | 5344.50 | 5358.17 | 5362.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 15:15:00 | 5330.00 | 5352.53 | 5359.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 09:15:00 | 5398.50 | 5352.53 | 5359.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-25 09:15:00 | 5402.50 | 5362.53 | 5363.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-25 10:00:00 | 5402.50 | 5362.53 | 5363.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-25 10:15:00 | 5435.00 | 5377.02 | 5369.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 10:15:00 | 5435.00 | 5377.02 | 5369.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 10:15:00 | 5435.00 | 5377.02 | 5369.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-25 10:15:00 | 5435.00 | 5377.02 | 5369.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — BUY (started 2025-06-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 10:15:00 | 5435.00 | 5377.02 | 5369.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 12:15:00 | 5459.00 | 5401.41 | 5382.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-25 15:15:00 | 5413.50 | 5417.65 | 5395.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-26 09:15:00 | 5414.00 | 5417.65 | 5395.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 09:15:00 | 5365.00 | 5407.12 | 5393.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-26 10:00:00 | 5365.00 | 5407.12 | 5393.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 10:15:00 | 5387.50 | 5403.19 | 5392.59 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2025-06-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 15:15:00 | 5375.00 | 5386.87 | 5387.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-27 10:15:00 | 5363.50 | 5380.62 | 5384.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 13:15:00 | 5311.00 | 5304.74 | 5332.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 14:00:00 | 5311.00 | 5304.74 | 5332.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 09:15:00 | 5314.00 | 5309.31 | 5327.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:30:00 | 5345.50 | 5309.31 | 5327.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 5317.00 | 5310.85 | 5326.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 10:30:00 | 5333.00 | 5310.85 | 5326.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 15:15:00 | 5320.00 | 5306.25 | 5317.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 09:15:00 | 5339.50 | 5306.25 | 5317.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 09:15:00 | 5321.50 | 5309.30 | 5317.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 10:15:00 | 5309.00 | 5309.30 | 5317.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 12:15:00 | 5307.00 | 5311.51 | 5317.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 09:15:00 | 5305.00 | 5309.30 | 5313.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 09:45:00 | 5314.00 | 5312.94 | 5314.91 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 5337.00 | 5317.75 | 5316.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 5337.00 | 5317.75 | 5316.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 5337.00 | 5317.75 | 5316.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-03 10:15:00 | 5337.00 | 5317.75 | 5316.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 5337.00 | 5317.75 | 5316.92 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 11:15:00 | 5310.00 | 5319.39 | 5319.51 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-07-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 09:15:00 | 5360.00 | 5324.39 | 5321.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-08 09:15:00 | 5376.50 | 5347.91 | 5335.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 13:15:00 | 5344.50 | 5359.99 | 5346.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-08 13:15:00 | 5344.50 | 5359.99 | 5346.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 5344.50 | 5359.99 | 5346.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:00:00 | 5344.50 | 5359.99 | 5346.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 5358.00 | 5359.59 | 5347.45 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-08 15:15:00 | 5377.00 | 5359.59 | 5347.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 09:15:00 | 5324.50 | 5355.36 | 5347.81 | SL hit (close<static) qty=1.00 sl=5336.00 alert=retest2 |

### Cycle 11 — SELL (started 2025-07-09 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 12:15:00 | 5316.50 | 5338.96 | 5341.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 09:15:00 | 5233.00 | 5295.79 | 5313.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 5200.00 | 5172.19 | 5207.65 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 5200.00 | 5172.19 | 5207.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 5200.00 | 5172.19 | 5207.65 | EMA400 retest candle locked (from downside) |

### Cycle 12 — BUY (started 2025-07-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 14:15:00 | 5248.00 | 5223.92 | 5223.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-16 09:15:00 | 5293.00 | 5242.07 | 5232.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 09:15:00 | 5238.00 | 5282.06 | 5262.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-17 09:15:00 | 5238.00 | 5282.06 | 5262.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 5238.00 | 5282.06 | 5262.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:00:00 | 5238.00 | 5282.06 | 5262.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 10:15:00 | 5236.00 | 5272.84 | 5260.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 10:30:00 | 5210.00 | 5272.84 | 5260.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2025-07-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 13:15:00 | 5228.00 | 5249.30 | 5251.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-17 14:15:00 | 5203.00 | 5240.04 | 5247.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 10:15:00 | 5147.50 | 5140.95 | 5172.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 09:15:00 | 5170.00 | 5153.94 | 5166.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 5170.00 | 5153.94 | 5166.32 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2025-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 15:15:00 | 5178.00 | 5172.42 | 5171.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 10:15:00 | 5224.00 | 5183.07 | 5176.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-24 09:15:00 | 5224.00 | 5243.10 | 5215.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-24 10:00:00 | 5224.00 | 5243.10 | 5215.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 5231.00 | 5240.68 | 5217.00 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 12:30:00 | 5244.00 | 5232.61 | 5217.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 14:15:00 | 5201.00 | 5222.99 | 5215.19 | SL hit (close<static) qty=1.00 sl=5205.00 alert=retest2 |

### Cycle 15 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 5126.50 | 5200.02 | 5205.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 5080.00 | 5176.01 | 5194.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 13:15:00 | 5073.00 | 5060.80 | 5092.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 14:00:00 | 5073.00 | 5060.80 | 5092.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 5083.50 | 5063.19 | 5085.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 10:00:00 | 5083.50 | 5063.19 | 5085.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 10:15:00 | 5084.50 | 5067.45 | 5085.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:15:00 | 5101.00 | 5067.45 | 5085.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 11:15:00 | 5104.00 | 5074.76 | 5087.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 11:45:00 | 5113.50 | 5074.76 | 5087.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 12:15:00 | 5100.00 | 5079.81 | 5088.19 | EMA400 retest candle locked (from downside) |

### Cycle 16 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 5138.00 | 5096.76 | 5094.77 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 09:15:00 | 5062.50 | 5100.93 | 5101.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 11:15:00 | 5020.00 | 5078.92 | 5090.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 5050.00 | 5025.78 | 5051.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-04 12:15:00 | 5050.00 | 5025.78 | 5051.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 12:15:00 | 5050.00 | 5025.78 | 5051.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:00:00 | 5050.00 | 5025.78 | 5051.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 5075.50 | 5035.73 | 5053.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 5075.50 | 5035.73 | 5053.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 5087.00 | 5045.98 | 5056.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:30:00 | 5089.50 | 5045.98 | 5056.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 10:15:00 | 5077.00 | 5059.90 | 5061.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-05 10:30:00 | 5068.50 | 5059.90 | 5061.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2025-08-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 11:15:00 | 5107.00 | 5069.32 | 5065.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-05 14:15:00 | 5144.00 | 5086.77 | 5074.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-06 10:15:00 | 5065.50 | 5091.29 | 5080.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-06 10:15:00 | 5065.50 | 5091.29 | 5080.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 10:15:00 | 5065.50 | 5091.29 | 5080.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 10:45:00 | 5063.00 | 5091.29 | 5080.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 11:15:00 | 5076.00 | 5088.23 | 5080.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 11:30:00 | 5066.50 | 5088.23 | 5080.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 12:15:00 | 5079.50 | 5086.48 | 5080.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 12:45:00 | 5076.50 | 5086.48 | 5080.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-06 13:15:00 | 5044.50 | 5078.09 | 5076.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-06 14:00:00 | 5044.50 | 5078.09 | 5076.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2025-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 14:15:00 | 5034.00 | 5069.27 | 5073.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 11:15:00 | 5002.50 | 5052.24 | 5063.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 5037.50 | 5031.17 | 5049.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 15:00:00 | 5037.50 | 5031.17 | 5049.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 5044.50 | 5033.84 | 5048.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 09:15:00 | 5012.00 | 5033.84 | 5048.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 12:15:00 | 5030.50 | 5029.71 | 5042.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-11 13:15:00 | 5062.00 | 5031.18 | 5033.48 | SL hit (close>static) qty=1.00 sl=5051.50 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-11 13:15:00 | 5062.00 | 5031.18 | 5033.48 | SL hit (close>static) qty=1.00 sl=5051.50 alert=retest2 |

### Cycle 20 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 5064.50 | 5037.85 | 5036.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 5115.00 | 5055.78 | 5044.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 5092.50 | 5095.26 | 5074.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 11:15:00 | 5067.50 | 5087.26 | 5074.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 11:15:00 | 5067.50 | 5087.26 | 5074.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-13 11:45:00 | 5062.00 | 5087.26 | 5074.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 5106.00 | 5091.01 | 5077.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 14:45:00 | 5120.50 | 5099.09 | 5083.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 5178.50 | 5099.17 | 5084.84 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 11:15:00 | 5030.00 | 5099.28 | 5102.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-18 11:15:00 | 5030.00 | 5099.28 | 5102.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 21 — SELL (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 11:15:00 | 5030.00 | 5099.28 | 5102.46 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2025-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 15:15:00 | 5120.00 | 5098.38 | 5095.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 09:15:00 | 5133.50 | 5105.40 | 5099.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 09:15:00 | 5167.00 | 5171.30 | 5145.14 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-21 09:15:00 | 5167.00 | 5171.30 | 5145.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 09:15:00 | 5167.00 | 5171.30 | 5145.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 10:30:00 | 5234.00 | 5177.64 | 5150.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 12:00:00 | 5218.00 | 5185.71 | 5156.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-21 13:00:00 | 5225.00 | 5193.57 | 5162.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-22 10:00:00 | 5225.50 | 5216.28 | 5185.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 5277.00 | 5289.08 | 5259.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 10:30:00 | 5301.00 | 5289.86 | 5262.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-26 11:15:00 | 5300.00 | 5289.86 | 5262.28 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-26 12:15:00 | 5240.00 | 5278.39 | 5261.76 | SL hit (close<static) qty=1.00 sl=5250.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 12:15:00 | 5240.00 | 5278.39 | 5261.76 | SL hit (close<static) qty=1.00 sl=5250.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 14:15:00 | 5156.50 | 5240.51 | 5246.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 14:15:00 | 5156.50 | 5240.51 | 5246.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 14:15:00 | 5156.50 | 5240.51 | 5246.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-26 14:15:00 | 5156.50 | 5240.51 | 5246.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 14:15:00 | 5156.50 | 5240.51 | 5246.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 15:15:00 | 5153.00 | 5223.01 | 5238.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-28 13:15:00 | 5154.00 | 5153.04 | 5191.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-28 13:45:00 | 5184.00 | 5153.04 | 5191.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 5144.00 | 5135.16 | 5155.03 | EMA400 retest candle locked (from downside) |

### Cycle 24 — BUY (started 2025-09-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 14:15:00 | 5200.00 | 5170.64 | 5167.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 15:15:00 | 5212.50 | 5179.01 | 5171.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 13:15:00 | 5209.50 | 5213.02 | 5193.37 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-02 14:00:00 | 5209.50 | 5213.02 | 5193.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 5209.00 | 5220.34 | 5202.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 10:15:00 | 5199.00 | 5220.34 | 5202.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 5155.50 | 5207.37 | 5198.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 11:00:00 | 5155.50 | 5207.37 | 5198.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 11:15:00 | 5170.00 | 5199.90 | 5195.59 | EMA400 retest candle locked (from upside) |

### Cycle 25 — SELL (started 2025-09-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 12:15:00 | 5147.00 | 5189.32 | 5191.17 | EMA200 below EMA400 |

### Cycle 26 — BUY (started 2025-09-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 09:15:00 | 5196.00 | 5191.48 | 5191.16 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-09-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 10:15:00 | 5181.50 | 5189.48 | 5190.28 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2025-09-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-04 13:15:00 | 5218.50 | 5194.59 | 5192.31 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 5134.00 | 5185.36 | 5189.03 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2025-09-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 15:15:00 | 5215.00 | 5192.23 | 5190.92 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-09-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 09:15:00 | 5163.50 | 5186.48 | 5188.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 10:15:00 | 5140.00 | 5177.18 | 5184.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-09 09:15:00 | 5144.00 | 5137.42 | 5158.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 09:15:00 | 5144.00 | 5137.42 | 5158.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 09:15:00 | 5144.00 | 5137.42 | 5158.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 09:45:00 | 5166.50 | 5137.42 | 5158.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 5168.50 | 5141.73 | 5156.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:00:00 | 5168.50 | 5141.73 | 5156.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 5182.00 | 5149.78 | 5158.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:00:00 | 5182.00 | 5149.78 | 5158.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-09 15:15:00 | 5173.00 | 5165.77 | 5164.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 09:15:00 | 5350.00 | 5202.61 | 5181.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 09:15:00 | 5217.00 | 5278.48 | 5244.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 09:15:00 | 5217.00 | 5278.48 | 5244.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 5217.00 | 5278.48 | 5244.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 10:00:00 | 5217.00 | 5278.48 | 5244.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 10:15:00 | 5248.50 | 5272.48 | 5245.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 12:15:00 | 5272.50 | 5268.59 | 5245.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 10:15:00 | 5271.00 | 5434.87 | 5452.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — SELL (started 2025-09-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 10:15:00 | 5271.00 | 5434.87 | 5452.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 5246.00 | 5310.16 | 5372.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 15:15:00 | 5293.50 | 5281.14 | 5327.04 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-09-24 09:15:00 | 5237.00 | 5281.14 | 5327.04 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 14:15:00 | 5164.50 | 5123.12 | 5148.11 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 5164.50 | 5123.12 | 5148.11 | SL hit (close>ema400) qty=1.00 sl=5148.11 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-09-29 15:00:00 | 5164.50 | 5123.12 | 5148.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 15:15:00 | 5147.50 | 5128.00 | 5148.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 5122.50 | 5128.00 | 5148.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 09:15:00 | 5085.50 | 5135.53 | 5141.65 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 12:15:00 | 5145.50 | 5128.06 | 5126.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 12:15:00 | 5145.50 | 5128.06 | 5126.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2025-10-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 12:15:00 | 5145.50 | 5128.06 | 5126.24 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2025-10-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 14:15:00 | 5119.00 | 5124.64 | 5124.90 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 5225.00 | 5143.89 | 5133.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 10:15:00 | 5252.50 | 5165.61 | 5144.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 09:15:00 | 5330.00 | 5333.53 | 5294.65 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 12:00:00 | 5377.00 | 5346.62 | 5307.62 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-10-09 13:45:00 | 5385.50 | 5359.76 | 5320.71 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 5465.50 | 5491.07 | 5464.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 15:00:00 | 5465.50 | 5491.07 | 5464.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 15:15:00 | 5470.00 | 5486.86 | 5464.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-15 09:15:00 | 5615.00 | 5486.86 | 5464.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 09:15:00 | 5645.85 | 5583.45 | 5535.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-16 10:15:00 | 5654.78 | 5598.06 | 5546.09 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-10-16 15:15:00 | 5615.00 | 5617.49 | 5577.43 | SL hit (close<ema200) qty=0.50 sl=5617.49 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-10-16 15:15:00 | 5615.00 | 5617.49 | 5577.43 | SL hit (close<ema200) qty=0.50 sl=5617.49 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-10-21 14:15:00 | 5545.50 | 5596.57 | 5599.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 37 — SELL (started 2025-10-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-21 14:15:00 | 5545.50 | 5596.57 | 5599.96 | EMA200 below EMA400 |

### Cycle 38 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 5635.00 | 5604.25 | 5603.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 11:15:00 | 5668.00 | 5623.44 | 5612.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 5608.50 | 5629.30 | 5618.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-23 14:15:00 | 5608.50 | 5629.30 | 5618.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 5608.50 | 5629.30 | 5618.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 5608.50 | 5629.30 | 5618.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 5602.00 | 5623.84 | 5617.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 5614.00 | 5623.84 | 5617.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 5631.50 | 5625.37 | 5618.53 | EMA400 retest candle locked (from upside) |

### Cycle 39 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 5567.00 | 5610.76 | 5612.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 13:15:00 | 5525.00 | 5587.89 | 5601.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-27 09:15:00 | 5637.50 | 5584.98 | 5595.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-27 09:15:00 | 5637.50 | 5584.98 | 5595.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 5637.50 | 5584.98 | 5595.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-27 10:00:00 | 5637.50 | 5584.98 | 5595.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 10:15:00 | 5617.00 | 5591.38 | 5597.74 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2025-10-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 12:15:00 | 5623.00 | 5604.68 | 5603.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 14:15:00 | 5644.00 | 5615.32 | 5608.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-28 10:15:00 | 5604.00 | 5619.67 | 5612.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-28 10:15:00 | 5604.00 | 5619.67 | 5612.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 5604.00 | 5619.67 | 5612.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:00:00 | 5604.00 | 5619.67 | 5612.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 5600.00 | 5615.73 | 5611.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:30:00 | 5605.50 | 5615.73 | 5611.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 41 — SELL (started 2025-10-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 13:15:00 | 5582.00 | 5607.59 | 5608.45 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2025-10-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-29 10:15:00 | 5634.00 | 5608.59 | 5608.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-29 12:15:00 | 5664.00 | 5625.18 | 5616.14 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 5645.50 | 5645.86 | 5630.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 09:15:00 | 5645.50 | 5645.86 | 5630.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 5645.50 | 5645.86 | 5630.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:45:00 | 5637.00 | 5645.86 | 5630.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 10:15:00 | 5656.50 | 5647.99 | 5632.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 10:45:00 | 5638.00 | 5647.99 | 5632.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 5680.00 | 5694.06 | 5677.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 5708.00 | 5694.06 | 5677.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 11:15:00 | 5711.00 | 5692.48 | 5679.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 13:00:00 | 5705.50 | 5692.45 | 5681.53 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-04 10:15:00 | 5614.00 | 5675.05 | 5678.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-04 10:15:00 | 5614.00 | 5675.05 | 5678.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-04 10:15:00 | 5614.00 | 5675.05 | 5678.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — SELL (started 2025-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 10:15:00 | 5614.00 | 5675.05 | 5678.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 11:15:00 | 5594.00 | 5658.84 | 5670.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-06 10:15:00 | 5640.00 | 5633.30 | 5649.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-06 11:00:00 | 5640.00 | 5633.30 | 5649.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 11:15:00 | 5625.00 | 5631.64 | 5647.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 12:00:00 | 5625.00 | 5631.64 | 5647.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 12:15:00 | 5670.50 | 5639.41 | 5649.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 13:00:00 | 5670.50 | 5639.41 | 5649.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 13:15:00 | 5664.00 | 5644.33 | 5650.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 15:15:00 | 5641.00 | 5645.56 | 5650.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 5682.00 | 5607.27 | 5615.29 | SL hit (close>static) qty=1.00 sl=5673.00 alert=retest2 |

### Cycle 44 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 5662.50 | 5628.91 | 5624.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 14:15:00 | 5712.50 | 5659.64 | 5643.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 14:15:00 | 5845.50 | 5866.78 | 5812.81 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-13 15:00:00 | 5845.50 | 5866.78 | 5812.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 5807.00 | 5852.14 | 5815.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 10:15:00 | 5810.00 | 5852.14 | 5815.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 10:15:00 | 5817.00 | 5845.11 | 5815.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 11:45:00 | 5830.50 | 5841.19 | 5816.42 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-14 12:15:00 | 5772.50 | 5827.45 | 5812.42 | SL hit (close<static) qty=1.00 sl=5798.00 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 5836.00 | 5817.46 | 5810.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 11:45:00 | 5828.50 | 5821.13 | 5814.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-17 13:30:00 | 5835.00 | 5823.57 | 5816.63 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 5788.00 | 5824.07 | 5819.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 5788.00 | 5824.07 | 5819.29 | SL hit (close<static) qty=1.00 sl=5798.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 5788.00 | 5824.07 | 5819.29 | SL hit (close<static) qty=1.00 sl=5798.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-18 09:15:00 | 5788.00 | 5824.07 | 5819.29 | SL hit (close<static) qty=1.00 sl=5798.00 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 5788.00 | 5824.07 | 5819.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 5782.50 | 5815.75 | 5815.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-18 14:15:00 | 5756.00 | 5791.14 | 5803.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-19 09:15:00 | 5898.50 | 5808.03 | 5808.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-19 09:15:00 | 5898.50 | 5808.03 | 5808.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 5898.50 | 5808.03 | 5808.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-19 10:00:00 | 5898.50 | 5808.03 | 5808.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — BUY (started 2025-11-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 10:15:00 | 5896.00 | 5825.62 | 5816.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-19 11:15:00 | 5983.00 | 5857.10 | 5831.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-21 09:15:00 | 5952.00 | 6010.86 | 5964.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-21 09:15:00 | 5952.00 | 6010.86 | 5964.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 09:15:00 | 5952.00 | 6010.86 | 5964.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 09:45:00 | 5954.00 | 6010.86 | 5964.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 10:15:00 | 5936.50 | 5995.99 | 5961.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:00:00 | 5936.50 | 5995.99 | 5961.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-21 11:15:00 | 5926.50 | 5982.09 | 5958.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-21 11:30:00 | 5922.00 | 5982.09 | 5958.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 5940.50 | 5966.42 | 5959.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 13:00:00 | 5940.50 | 5966.42 | 5959.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 5939.50 | 5961.04 | 5957.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:00:00 | 5939.50 | 5961.04 | 5957.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 47 — SELL (started 2025-11-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-24 14:15:00 | 5918.00 | 5952.43 | 5953.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 15:15:00 | 5905.50 | 5943.04 | 5949.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-26 09:15:00 | 5860.00 | 5858.16 | 5891.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-26 09:15:00 | 5860.00 | 5858.16 | 5891.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-26 09:15:00 | 5860.00 | 5858.16 | 5891.35 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 10:15:00 | 5928.50 | 5896.04 | 5894.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 12:15:00 | 5947.00 | 5913.10 | 5902.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-02 09:15:00 | 6115.50 | 6120.04 | 6073.31 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-02 14:30:00 | 6160.50 | 6128.89 | 6094.67 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 12:15:00 | 6155.00 | 6157.51 | 6124.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 12:30:00 | 6137.00 | 6157.51 | 6124.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 6279.00 | 6297.36 | 6267.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 13:00:00 | 6279.00 | 6297.36 | 6267.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 13:15:00 | 6263.00 | 6290.49 | 6266.79 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-08 13:15:00 | 6263.00 | 6290.49 | 6266.79 | SL hit (close<ema400) qty=1.00 sl=6266.79 alert=retest1 |
| ALERT3_SIDEWAYS | 2025-12-08 13:30:00 | 6252.00 | 6290.49 | 6266.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 6258.50 | 6284.09 | 6266.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 14:30:00 | 6221.00 | 6284.09 | 6266.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 15:15:00 | 6253.50 | 6277.97 | 6264.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:15:00 | 6218.00 | 6277.97 | 6264.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 6191.00 | 6260.58 | 6258.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:45:00 | 6186.50 | 6260.58 | 6258.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 49 — SELL (started 2025-12-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 10:15:00 | 6210.00 | 6250.46 | 6253.80 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 10:15:00 | 6266.00 | 6242.85 | 6242.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 11:15:00 | 6301.00 | 6254.48 | 6248.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 09:15:00 | 6268.00 | 6272.14 | 6260.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 09:15:00 | 6268.00 | 6272.14 | 6260.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 09:15:00 | 6268.00 | 6272.14 | 6260.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:00:00 | 6268.00 | 6272.14 | 6260.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 10:15:00 | 6274.00 | 6272.52 | 6261.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 10:45:00 | 6260.00 | 6272.52 | 6261.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 6289.50 | 6282.67 | 6270.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:30:00 | 6272.50 | 6282.67 | 6270.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 6247.50 | 6279.37 | 6271.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:45:00 | 6250.50 | 6279.37 | 6271.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 10:15:00 | 6291.00 | 6281.70 | 6272.96 | EMA400 retest candle locked (from upside) |

### Cycle 51 — SELL (started 2025-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 09:15:00 | 6197.00 | 6264.42 | 6269.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-19 09:15:00 | 6108.00 | 6212.19 | 6228.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-19 13:15:00 | 6198.00 | 6189.30 | 6210.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 14:00:00 | 6198.00 | 6189.30 | 6210.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 14:15:00 | 6199.50 | 6191.34 | 6209.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 14:45:00 | 6185.50 | 6191.34 | 6209.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 15:15:00 | 6185.00 | 6190.07 | 6207.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-22 09:15:00 | 6235.00 | 6190.07 | 6207.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 09:15:00 | 6217.50 | 6195.56 | 6208.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 12:15:00 | 6188.50 | 6203.06 | 6209.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-22 14:45:00 | 6196.00 | 6202.59 | 6207.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-23 09:15:00 | 6142.00 | 6201.67 | 6207.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 6194.50 | 6192.77 | 6197.16 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 6230.00 | 6200.22 | 6200.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 6230.00 | 6200.22 | 6200.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 6230.00 | 6200.22 | 6200.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 09:15:00 | 6230.00 | 6200.22 | 6200.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 52 — BUY (started 2025-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-24 09:15:00 | 6230.00 | 6200.22 | 6200.15 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 13:15:00 | 6173.00 | 6198.13 | 6199.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 15:15:00 | 6135.00 | 6181.08 | 6191.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 10:15:00 | 6060.50 | 6058.43 | 6088.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 10:15:00 | 6060.50 | 6058.43 | 6088.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 6060.50 | 6058.43 | 6088.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 12:30:00 | 6041.00 | 6058.12 | 6083.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-31 11:45:00 | 6044.00 | 6065.04 | 6076.84 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-01 14:15:00 | 6113.00 | 6079.61 | 6077.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-01 14:15:00 | 6113.00 | 6079.61 | 6077.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 54 — BUY (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 14:15:00 | 6113.00 | 6079.61 | 6077.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-01 15:15:00 | 6118.00 | 6087.29 | 6080.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 11:15:00 | 6087.50 | 6092.59 | 6085.33 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 11:15:00 | 6087.50 | 6092.59 | 6085.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 6087.50 | 6092.59 | 6085.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:00:00 | 6087.50 | 6092.59 | 6085.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 6075.00 | 6089.07 | 6084.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:00:00 | 6075.00 | 6089.07 | 6084.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 6072.00 | 6085.66 | 6083.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:15:00 | 6076.00 | 6085.66 | 6083.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 14:15:00 | 6068.50 | 6082.23 | 6081.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 14:30:00 | 6058.00 | 6082.23 | 6081.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 55 — SELL (started 2026-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 15:15:00 | 6071.50 | 6080.08 | 6080.98 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 11:15:00 | 6102.00 | 6080.86 | 6080.72 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 13:15:00 | 6053.00 | 6076.43 | 6078.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 09:15:00 | 6028.50 | 6064.51 | 6072.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-07 09:15:00 | 6082.50 | 6021.15 | 6039.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 09:15:00 | 6082.50 | 6021.15 | 6039.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 6082.50 | 6021.15 | 6039.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:00:00 | 6082.50 | 6021.15 | 6039.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 6095.00 | 6035.92 | 6044.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:45:00 | 6113.50 | 6035.92 | 6044.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 11:15:00 | 6116.00 | 6051.94 | 6051.37 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 6037.00 | 6056.82 | 6058.19 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-12 09:15:00 | 5951.00 | 6024.05 | 6038.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 14:15:00 | 6010.50 | 5994.39 | 6014.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 15:00:00 | 6010.50 | 5994.39 | 6014.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 6012.00 | 5997.92 | 6014.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 6038.50 | 5997.92 | 6014.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 6033.00 | 6004.93 | 6016.29 | EMA400 retest candle locked (from downside) |

### Cycle 60 — BUY (started 2026-01-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-13 11:15:00 | 6084.50 | 6032.62 | 6027.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-13 15:15:00 | 6108.00 | 6061.07 | 6043.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-14 09:15:00 | 6033.00 | 6055.45 | 6042.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 6033.00 | 6055.45 | 6042.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 6033.00 | 6055.45 | 6042.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:00:00 | 6033.00 | 6055.45 | 6042.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 6024.00 | 6049.16 | 6040.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:30:00 | 6039.00 | 6049.16 | 6040.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 6049.00 | 6049.13 | 6041.39 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-16 09:15:00 | 6183.00 | 6039.84 | 6039.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 09:15:00 | 6007.50 | 6276.06 | 6238.64 | SL hit (close<static) qty=1.00 sl=6022.00 alert=retest2 |

### Cycle 61 — SELL (started 2026-01-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 11:15:00 | 5921.50 | 6157.30 | 6188.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 5798.00 | 5999.42 | 6091.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 5986.00 | 5894.07 | 5975.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 5986.00 | 5894.07 | 5975.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 5986.00 | 5894.07 | 5975.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:30:00 | 6010.00 | 5894.07 | 5975.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 5942.50 | 5903.75 | 5972.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:45:00 | 5986.50 | 5903.75 | 5972.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 15:15:00 | 5950.00 | 5923.12 | 5956.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 09:15:00 | 6013.00 | 5923.12 | 5956.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 5955.00 | 5929.49 | 5956.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 11:30:00 | 5897.00 | 5921.18 | 5947.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-28 09:15:00 | 5985.00 | 5931.30 | 5927.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — BUY (started 2026-01-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 09:15:00 | 5985.00 | 5931.30 | 5927.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 6010.00 | 5967.46 | 5948.01 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 5924.00 | 5966.53 | 5951.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 5924.00 | 5966.53 | 5951.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 5924.00 | 5966.53 | 5951.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:45:00 | 5913.00 | 5966.53 | 5951.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 5930.00 | 5959.23 | 5949.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 11:30:00 | 5966.00 | 5955.88 | 5948.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 10:30:00 | 5955.00 | 5959.75 | 5956.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 10:15:00 | 5955.00 | 5959.94 | 5958.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 5693.00 | 5971.36 | 5993.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 5693.00 | 5971.36 | 5993.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-04 09:15:00 | 5693.00 | 5971.36 | 5993.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 09:15:00 | 5693.00 | 5971.36 | 5993.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 09:15:00 | 5551.50 | 5663.95 | 5746.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 10:15:00 | 5605.00 | 5581.79 | 5645.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-09 11:00:00 | 5605.00 | 5581.79 | 5645.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 5635.00 | 5592.44 | 5644.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 11:30:00 | 5630.50 | 5592.44 | 5644.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 13:15:00 | 5627.00 | 5606.16 | 5642.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-09 13:30:00 | 5641.50 | 5606.16 | 5642.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 09:15:00 | 5648.00 | 5619.91 | 5640.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:00:00 | 5648.00 | 5619.91 | 5640.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 10:15:00 | 5656.50 | 5627.22 | 5641.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 10:45:00 | 5649.00 | 5627.22 | 5641.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 5659.00 | 5646.43 | 5647.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-10 15:00:00 | 5659.00 | 5646.43 | 5647.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 5615.50 | 5638.42 | 5643.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 12:00:00 | 5602.00 | 5630.59 | 5639.03 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-12 09:15:00 | 5321.90 | 5512.78 | 5576.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2026-02-13 09:15:00 | 5041.80 | 5231.08 | 5379.38 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 64 — BUY (started 2026-03-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 13:15:00 | 4338.90 | 4302.34 | 4301.71 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 4224.80 | 4291.33 | 4297.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 10:15:00 | 4184.60 | 4227.29 | 4256.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 4215.30 | 4206.61 | 4235.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 4215.70 | 4206.61 | 4235.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 4179.00 | 4197.97 | 4226.51 | EMA400 retest candle locked (from downside) |

### Cycle 66 — BUY (started 2026-03-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 09:15:00 | 4330.90 | 4226.60 | 4226.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 10:15:00 | 4347.10 | 4250.70 | 4237.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 4267.40 | 4288.02 | 4267.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 4267.40 | 4288.02 | 4267.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 4267.40 | 4288.02 | 4267.47 | EMA400 retest candle locked (from upside) |

### Cycle 67 — SELL (started 2026-03-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 13:15:00 | 4207.00 | 4253.54 | 4256.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 14:15:00 | 4188.00 | 4240.43 | 4250.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 13:15:00 | 4199.50 | 4190.23 | 4214.73 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-20 13:30:00 | 4195.00 | 4190.23 | 4214.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 14:15:00 | 4258.80 | 4203.94 | 4218.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-20 15:00:00 | 4258.80 | 4203.94 | 4218.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 68 — BUY (started 2026-03-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 15:15:00 | 4355.00 | 4234.15 | 4231.12 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2026-03-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 09:15:00 | 4131.70 | 4213.66 | 4222.08 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2026-03-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 14:15:00 | 4209.40 | 4180.11 | 4179.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 4249.00 | 4200.59 | 4189.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 10:15:00 | 4265.70 | 4267.39 | 4239.36 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-27 10:45:00 | 4260.30 | 4267.39 | 4239.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 12:15:00 | 4250.00 | 4266.11 | 4243.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 13:00:00 | 4250.00 | 4266.11 | 4243.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 13:15:00 | 4212.10 | 4255.30 | 4240.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 14:00:00 | 4212.10 | 4255.30 | 4240.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 4206.50 | 4245.54 | 4237.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 4206.50 | 4245.54 | 4237.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — SELL (started 2026-03-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 09:15:00 | 4130.60 | 4213.27 | 4223.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 10:15:00 | 4089.70 | 4188.55 | 4211.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 4228.20 | 4121.69 | 4159.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 4228.20 | 4121.69 | 4159.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 4228.20 | 4121.69 | 4159.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 4228.20 | 4121.69 | 4159.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 4197.30 | 4136.81 | 4162.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:30:00 | 4226.60 | 4136.81 | 4162.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 11:15:00 | 4184.80 | 4129.97 | 4143.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 11:45:00 | 4184.60 | 4129.97 | 4143.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 12:15:00 | 4240.50 | 4152.08 | 4152.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-02 13:00:00 | 4240.50 | 4152.08 | 4152.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 4300.30 | 4181.72 | 4166.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-02 14:15:00 | 4314.20 | 4208.22 | 4179.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 4432.30 | 4477.34 | 4417.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 10:00:00 | 4432.30 | 4477.34 | 4417.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 4467.10 | 4526.12 | 4477.92 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2026-04-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 13:15:00 | 4461.80 | 4472.63 | 4473.00 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 4578.00 | 4491.89 | 4481.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 14:15:00 | 4657.00 | 4577.81 | 4532.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 15:15:00 | 4700.00 | 4727.81 | 4681.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-20 09:15:00 | 4713.00 | 4727.81 | 4681.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 4713.00 | 4724.85 | 4684.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 11:45:00 | 4761.20 | 4729.84 | 4708.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 14:45:00 | 4760.60 | 4739.12 | 4718.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 09:15:00 | 4646.20 | 4723.46 | 4715.01 | SL hit (close<static) qty=1.00 sl=4680.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-22 09:15:00 | 4646.20 | 4723.46 | 4715.01 | SL hit (close<static) qty=1.00 sl=4680.70 alert=retest2 |

### Cycle 75 — SELL (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 10:15:00 | 4603.00 | 4699.37 | 4704.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-22 11:15:00 | 4575.00 | 4674.49 | 4693.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 11:15:00 | 4342.00 | 4341.75 | 4422.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-27 12:00:00 | 4342.00 | 4341.75 | 4422.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 4370.40 | 4349.59 | 4395.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 13:30:00 | 4334.90 | 4356.59 | 4374.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-06 09:15:00 | 4349.40 | 4285.61 | 4278.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — BUY (started 2026-05-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-06 09:15:00 | 4349.40 | 4285.61 | 4278.97 | EMA200 above EMA400 |

### Cycle 77 — SELL (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-07 11:15:00 | 4251.50 | 4280.41 | 4283.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-07 12:15:00 | 4244.70 | 4273.27 | 4279.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-08 09:15:00 | 4302.10 | 4267.93 | 4273.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-08 09:15:00 | 4302.10 | 4267.93 | 4273.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 09:15:00 | 4302.10 | 4267.93 | 4273.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-08 10:15:00 | 4346.30 | 4267.93 | 4273.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2026-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 10:15:00 | 4324.80 | 4279.31 | 4278.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-08 15:15:00 | 4360.00 | 4326.41 | 4304.44 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-12 09:15:00 | 4719.40 | 2025-05-20 09:15:00 | 5097.29 | TARGET_HIT | 1.00 | 8.01% |
| BUY | retest2 | 2025-05-27 11:15:00 | 5097.30 | 2025-05-30 11:15:00 | 5070.60 | STOP_HIT | 1.00 | -0.52% |
| BUY | retest2 | 2025-05-27 14:45:00 | 5109.40 | 2025-05-30 11:15:00 | 5070.60 | STOP_HIT | 1.00 | -0.76% |
| BUY | retest2 | 2025-05-28 11:30:00 | 5089.20 | 2025-05-30 11:15:00 | 5070.60 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-06-03 09:30:00 | 5032.00 | 2025-06-04 09:15:00 | 5136.00 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-06-13 10:15:00 | 5449.00 | 2025-06-19 09:15:00 | 5314.50 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-06-16 10:15:00 | 5457.00 | 2025-06-19 09:15:00 | 5314.50 | STOP_HIT | 1.00 | -2.61% |
| BUY | retest2 | 2025-06-16 12:15:00 | 5440.00 | 2025-06-19 09:15:00 | 5314.50 | STOP_HIT | 1.00 | -2.31% |
| BUY | retest2 | 2025-06-18 13:30:00 | 5449.50 | 2025-06-19 09:15:00 | 5314.50 | STOP_HIT | 1.00 | -2.48% |
| SELL | retest2 | 2025-06-23 09:45:00 | 5340.00 | 2025-06-25 10:15:00 | 5435.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-06-24 13:30:00 | 5356.00 | 2025-06-25 10:15:00 | 5435.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-06-24 14:00:00 | 5356.50 | 2025-06-25 10:15:00 | 5435.00 | STOP_HIT | 1.00 | -1.47% |
| SELL | retest2 | 2025-06-24 14:45:00 | 5344.50 | 2025-06-25 10:15:00 | 5435.00 | STOP_HIT | 1.00 | -1.69% |
| SELL | retest2 | 2025-07-02 10:15:00 | 5309.00 | 2025-07-03 10:15:00 | 5337.00 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-07-02 12:15:00 | 5307.00 | 2025-07-03 10:15:00 | 5337.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-07-03 09:15:00 | 5305.00 | 2025-07-03 10:15:00 | 5337.00 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-07-03 09:45:00 | 5314.00 | 2025-07-03 10:15:00 | 5337.00 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2025-07-08 15:15:00 | 5377.00 | 2025-07-09 09:15:00 | 5324.50 | STOP_HIT | 1.00 | -0.98% |
| BUY | retest2 | 2025-07-24 12:30:00 | 5244.00 | 2025-07-24 14:15:00 | 5201.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-08-08 09:15:00 | 5012.00 | 2025-08-11 13:15:00 | 5062.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-08-08 12:15:00 | 5030.50 | 2025-08-11 13:15:00 | 5062.00 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-08-13 14:45:00 | 5120.50 | 2025-08-18 11:15:00 | 5030.00 | STOP_HIT | 1.00 | -1.77% |
| BUY | retest2 | 2025-08-14 09:15:00 | 5178.50 | 2025-08-18 11:15:00 | 5030.00 | STOP_HIT | 1.00 | -2.87% |
| BUY | retest2 | 2025-08-21 10:30:00 | 5234.00 | 2025-08-26 12:15:00 | 5240.00 | STOP_HIT | 1.00 | 0.11% |
| BUY | retest2 | 2025-08-21 12:00:00 | 5218.00 | 2025-08-26 12:15:00 | 5240.00 | STOP_HIT | 1.00 | 0.42% |
| BUY | retest2 | 2025-08-21 13:00:00 | 5225.00 | 2025-08-26 14:15:00 | 5156.50 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-08-22 10:00:00 | 5225.50 | 2025-08-26 14:15:00 | 5156.50 | STOP_HIT | 1.00 | -1.32% |
| BUY | retest2 | 2025-08-26 10:30:00 | 5301.00 | 2025-08-26 14:15:00 | 5156.50 | STOP_HIT | 1.00 | -2.73% |
| BUY | retest2 | 2025-08-26 11:15:00 | 5300.00 | 2025-08-26 14:15:00 | 5156.50 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2025-09-11 12:15:00 | 5272.50 | 2025-09-22 10:15:00 | 5271.00 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest1 | 2025-09-24 09:15:00 | 5237.00 | 2025-09-29 14:15:00 | 5164.50 | STOP_HIT | 1.00 | 1.38% |
| SELL | retest2 | 2025-09-30 09:15:00 | 5122.50 | 2025-10-03 12:15:00 | 5145.50 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-10-01 09:15:00 | 5085.50 | 2025-10-03 12:15:00 | 5145.50 | STOP_HIT | 1.00 | -1.18% |
| BUY | retest1 | 2025-10-09 12:00:00 | 5377.00 | 2025-10-16 09:15:00 | 5645.85 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-10-09 13:45:00 | 5385.50 | 2025-10-16 10:15:00 | 5654.78 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-10-09 12:00:00 | 5377.00 | 2025-10-16 15:15:00 | 5615.00 | STOP_HIT | 0.50 | 4.43% |
| BUY | retest1 | 2025-10-09 13:45:00 | 5385.50 | 2025-10-16 15:15:00 | 5615.00 | STOP_HIT | 0.50 | 4.26% |
| BUY | retest2 | 2025-10-15 09:15:00 | 5615.00 | 2025-10-21 14:15:00 | 5545.50 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2025-11-03 09:15:00 | 5708.00 | 2025-11-04 10:15:00 | 5614.00 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-11-03 11:15:00 | 5711.00 | 2025-11-04 10:15:00 | 5614.00 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-11-03 13:00:00 | 5705.50 | 2025-11-04 10:15:00 | 5614.00 | STOP_HIT | 1.00 | -1.60% |
| SELL | retest2 | 2025-11-06 15:15:00 | 5641.00 | 2025-11-10 10:15:00 | 5682.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-11-14 11:45:00 | 5830.50 | 2025-11-14 12:15:00 | 5772.50 | STOP_HIT | 1.00 | -0.99% |
| BUY | retest2 | 2025-11-17 09:15:00 | 5836.00 | 2025-11-18 09:15:00 | 5788.00 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-11-17 11:45:00 | 5828.50 | 2025-11-18 09:15:00 | 5788.00 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-11-17 13:30:00 | 5835.00 | 2025-11-18 09:15:00 | 5788.00 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest1 | 2025-12-02 14:30:00 | 6160.50 | 2025-12-08 13:15:00 | 6263.00 | STOP_HIT | 1.00 | 1.66% |
| SELL | retest2 | 2025-12-22 12:15:00 | 6188.50 | 2025-12-24 09:15:00 | 6230.00 | STOP_HIT | 1.00 | -0.67% |
| SELL | retest2 | 2025-12-22 14:45:00 | 6196.00 | 2025-12-24 09:15:00 | 6230.00 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-12-23 09:15:00 | 6142.00 | 2025-12-24 09:15:00 | 6230.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2025-12-24 09:15:00 | 6194.50 | 2025-12-24 09:15:00 | 6230.00 | STOP_HIT | 1.00 | -0.57% |
| SELL | retest2 | 2025-12-30 12:30:00 | 6041.00 | 2026-01-01 14:15:00 | 6113.00 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-12-31 11:45:00 | 6044.00 | 2026-01-01 14:15:00 | 6113.00 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2026-01-16 09:15:00 | 6183.00 | 2026-01-20 09:15:00 | 6007.50 | STOP_HIT | 1.00 | -2.84% |
| SELL | retest2 | 2026-01-23 11:30:00 | 5897.00 | 2026-01-28 09:15:00 | 5985.00 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2026-01-29 11:30:00 | 5966.00 | 2026-02-04 09:15:00 | 5693.00 | STOP_HIT | 1.00 | -4.58% |
| BUY | retest2 | 2026-01-30 10:30:00 | 5955.00 | 2026-02-04 09:15:00 | 5693.00 | STOP_HIT | 1.00 | -4.40% |
| BUY | retest2 | 2026-02-01 10:15:00 | 5955.00 | 2026-02-04 09:15:00 | 5693.00 | STOP_HIT | 1.00 | -4.40% |
| SELL | retest2 | 2026-02-11 12:00:00 | 5602.00 | 2026-02-12 09:15:00 | 5321.90 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-11 12:00:00 | 5602.00 | 2026-02-13 09:15:00 | 5041.80 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2026-04-21 11:45:00 | 4761.20 | 2026-04-22 09:15:00 | 4646.20 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2026-04-21 14:45:00 | 4760.60 | 2026-04-22 09:15:00 | 4646.20 | STOP_HIT | 1.00 | -2.40% |
| SELL | retest2 | 2026-04-29 13:30:00 | 4334.90 | 2026-05-06 09:15:00 | 4349.40 | STOP_HIT | 1.00 | -0.33% |
