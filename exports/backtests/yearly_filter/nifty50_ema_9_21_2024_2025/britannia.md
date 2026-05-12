# BRITANNIA (BRITANNIA)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 5516.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 164 |
| ALERT1 | 107 |
| ALERT2 | 104 |
| ALERT2_SKIP | 58 |
| ALERT3 | 298 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 4 |
| ENTRY2 | 151 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 155 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 159 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 43 / 116
- **Target hits / Stop hits / Partials:** 0 / 155 / 4
- **Avg / median % per leg:** -0.19% / -0.73%
- **Sum % (uncompounded):** -30.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 78 | 20 | 25.6% | 0 | 78 | 0 | -0.23% | -17.8% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.62% | -1.9% |
| BUY @ 3rd Alert (retest2) | 75 | 20 | 26.7% | 0 | 75 | 0 | -0.21% | -15.9% |
| SELL (all) | 81 | 23 | 28.4% | 0 | 77 | 4 | -0.16% | -12.6% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -0.81% | -0.8% |
| SELL @ 3rd Alert (retest2) | 80 | 23 | 28.7% | 0 | 76 | 4 | -0.15% | -11.8% |
| retest1 (combined) | 4 | 0 | 0.0% | 0 | 4 | 0 | -0.67% | -2.7% |
| retest2 (combined) | 155 | 43 | 27.7% | 0 | 151 | 4 | -0.18% | -27.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-13 11:15:00 | 5136.05 | 5100.15 | 5098.90 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2024-05-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-14 10:15:00 | 5062.45 | 5103.87 | 5104.50 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 13:15:00 | 5132.80 | 5107.28 | 5105.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 5140.80 | 5113.98 | 5108.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 09:15:00 | 5095.40 | 5113.48 | 5109.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-15 09:15:00 | 5095.40 | 5113.48 | 5109.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 09:15:00 | 5095.40 | 5113.48 | 5109.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:00:00 | 5095.40 | 5113.48 | 5109.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 10:15:00 | 5091.70 | 5109.12 | 5107.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 10:30:00 | 5073.00 | 5109.12 | 5107.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — SELL (started 2024-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 11:15:00 | 5081.90 | 5103.68 | 5105.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 09:15:00 | 5046.40 | 5077.06 | 5090.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 11:15:00 | 5088.90 | 5073.98 | 5086.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 11:15:00 | 5088.90 | 5073.98 | 5086.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 5088.90 | 5073.98 | 5086.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-16 12:00:00 | 5088.90 | 5073.98 | 5086.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 12:15:00 | 5048.90 | 5068.97 | 5083.18 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-05-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-16 15:15:00 | 5142.80 | 5091.06 | 5089.92 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2024-05-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-17 13:15:00 | 5078.15 | 5089.29 | 5089.68 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2024-05-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-18 11:15:00 | 5102.50 | 5091.26 | 5090.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-21 12:15:00 | 5113.10 | 5100.99 | 5096.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-23 09:15:00 | 5201.10 | 5229.68 | 5188.26 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-23 09:15:00 | 5201.10 | 5229.68 | 5188.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-23 09:15:00 | 5201.10 | 5229.68 | 5188.26 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-23 10:00:00 | 5201.10 | 5229.68 | 5188.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 13:15:00 | 5241.25 | 5255.31 | 5235.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 14:00:00 | 5241.25 | 5255.31 | 5235.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 14:15:00 | 5246.00 | 5253.45 | 5236.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-24 15:00:00 | 5246.00 | 5253.45 | 5236.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 5159.75 | 5233.68 | 5230.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-27 10:00:00 | 5159.75 | 5233.68 | 5230.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2024-05-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-27 10:15:00 | 5150.15 | 5216.97 | 5222.93 | EMA200 below EMA400 |

### Cycle 9 — BUY (started 2024-05-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-28 10:15:00 | 5247.05 | 5221.76 | 5220.96 | EMA200 above EMA400 |

### Cycle 10 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 5203.70 | 5227.98 | 5230.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-03 13:15:00 | 5173.35 | 5186.52 | 5197.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-04 09:15:00 | 5223.05 | 5189.13 | 5195.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-04 09:15:00 | 5223.05 | 5189.13 | 5195.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 09:15:00 | 5223.05 | 5189.13 | 5195.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 09:45:00 | 5253.75 | 5189.13 | 5195.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 10:15:00 | 5165.00 | 5184.31 | 5192.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 10:30:00 | 5242.70 | 5184.31 | 5192.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-04 11:15:00 | 5201.60 | 5187.77 | 5193.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-04 11:45:00 | 5214.15 | 5187.77 | 5193.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 12:15:00 | 5278.05 | 5205.82 | 5200.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-04 13:15:00 | 5369.90 | 5238.64 | 5216.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 09:15:00 | 5430.00 | 5451.98 | 5378.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 5430.00 | 5451.98 | 5378.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 5430.00 | 5451.98 | 5378.30 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 10:30:00 | 5530.45 | 5467.68 | 5438.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 11:00:00 | 5534.50 | 5467.68 | 5438.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-11 09:15:00 | 5592.90 | 5491.25 | 5464.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-12 12:15:00 | 5452.85 | 5488.73 | 5490.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — SELL (started 2024-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 12:15:00 | 5452.85 | 5488.73 | 5490.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-12 14:15:00 | 5437.05 | 5470.82 | 5481.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 12:15:00 | 5403.00 | 5395.70 | 5410.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-18 13:00:00 | 5403.00 | 5395.70 | 5410.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 13:15:00 | 5383.00 | 5393.16 | 5407.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 09:45:00 | 5380.65 | 5390.08 | 5402.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-19 13:45:00 | 5381.20 | 5389.73 | 5398.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 10:00:00 | 5381.80 | 5377.22 | 5389.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-20 13:45:00 | 5364.70 | 5370.49 | 5381.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 5377.00 | 5371.79 | 5381.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-20 15:00:00 | 5377.00 | 5371.79 | 5381.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 15:15:00 | 5389.15 | 5375.26 | 5382.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-21 10:00:00 | 5368.40 | 5373.89 | 5380.80 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-25 15:15:00 | 5362.00 | 5343.49 | 5342.79 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 15:15:00 | 5362.00 | 5343.49 | 5342.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 10:15:00 | 5380.65 | 5354.33 | 5348.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 12:15:00 | 5403.15 | 5409.55 | 5387.69 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-27 13:00:00 | 5403.15 | 5409.55 | 5387.69 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 5420.00 | 5411.64 | 5390.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-27 15:00:00 | 5436.25 | 5416.56 | 5394.77 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 13:15:00 | 5391.05 | 5436.19 | 5439.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2024-07-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 13:15:00 | 5391.05 | 5436.19 | 5439.74 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2024-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-03 10:15:00 | 5486.20 | 5443.75 | 5441.09 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-04 11:15:00 | 5424.00 | 5442.05 | 5443.09 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-07-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 10:15:00 | 5471.70 | 5443.91 | 5442.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 11:15:00 | 5486.05 | 5452.34 | 5446.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 09:15:00 | 5742.20 | 5753.03 | 5714.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 09:15:00 | 5742.20 | 5753.03 | 5714.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 5742.20 | 5753.03 | 5714.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:30:00 | 5729.40 | 5753.03 | 5714.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 5813.25 | 5808.43 | 5782.01 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 11:15:00 | 5843.70 | 5813.14 | 5786.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-16 11:45:00 | 5839.90 | 5822.42 | 5793.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-24 12:15:00 | 5839.00 | 5884.76 | 5887.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 12:15:00 | 5839.00 | 5884.76 | 5887.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-24 13:15:00 | 5834.00 | 5874.61 | 5882.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 10:15:00 | 5840.85 | 5837.16 | 5850.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-26 11:00:00 | 5840.85 | 5837.16 | 5850.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 12:15:00 | 5870.00 | 5844.96 | 5851.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 13:00:00 | 5870.00 | 5844.96 | 5851.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 13:15:00 | 5867.15 | 5849.40 | 5853.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 14:00:00 | 5867.15 | 5849.40 | 5853.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2024-07-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 15:15:00 | 5860.00 | 5856.03 | 5855.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 10:15:00 | 5883.90 | 5862.10 | 5858.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 11:15:00 | 5883.15 | 5887.61 | 5876.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 11:15:00 | 5883.15 | 5887.61 | 5876.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 11:15:00 | 5883.15 | 5887.61 | 5876.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 11:45:00 | 5873.95 | 5887.61 | 5876.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 12:15:00 | 5886.20 | 5887.33 | 5876.92 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2024-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 15:15:00 | 5836.00 | 5868.38 | 5870.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 12:15:00 | 5821.80 | 5852.86 | 5861.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 11:15:00 | 5834.45 | 5823.20 | 5839.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 11:15:00 | 5834.45 | 5823.20 | 5839.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 5834.45 | 5823.20 | 5839.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-01 11:45:00 | 5838.60 | 5823.20 | 5839.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 12:15:00 | 5754.00 | 5809.36 | 5831.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-01 13:15:00 | 5744.95 | 5809.36 | 5831.65 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-02 14:45:00 | 5690.60 | 5723.07 | 5761.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 09:15:00 | 5695.60 | 5731.65 | 5762.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 10:45:00 | 5738.20 | 5739.75 | 5760.97 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 12:15:00 | 5741.55 | 5742.55 | 5758.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-05 15:00:00 | 5708.65 | 5740.30 | 5755.17 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-06 09:15:00 | 5824.00 | 5746.68 | 5754.84 | SL hit (close>static) qty=1.00 sl=5759.00 alert=retest2 |

### Cycle 21 — BUY (started 2024-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 11:15:00 | 5865.80 | 5781.17 | 5769.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-06 12:15:00 | 5919.70 | 5808.88 | 5783.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-07 11:15:00 | 5847.45 | 5852.97 | 5820.73 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-07 11:45:00 | 5847.05 | 5852.97 | 5820.73 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 5818.60 | 5845.49 | 5822.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-07 13:45:00 | 5832.60 | 5845.49 | 5822.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 14:15:00 | 5851.80 | 5846.75 | 5825.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-07 14:45:00 | 5809.15 | 5846.75 | 5825.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 5835.90 | 5844.58 | 5826.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 10:00:00 | 5825.45 | 5840.76 | 5826.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 10:15:00 | 5770.75 | 5826.76 | 5821.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 11:00:00 | 5770.75 | 5826.76 | 5821.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 11:15:00 | 5781.60 | 5817.72 | 5817.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-08 11:45:00 | 5763.65 | 5817.72 | 5817.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2024-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 12:15:00 | 5744.80 | 5803.14 | 5811.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 15:15:00 | 5730.20 | 5773.15 | 5794.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 09:15:00 | 5705.45 | 5687.26 | 5719.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-13 10:00:00 | 5705.45 | 5687.26 | 5719.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 5597.15 | 5653.94 | 5686.08 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 13:15:00 | 5708.00 | 5681.95 | 5680.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 14:15:00 | 5731.75 | 5691.91 | 5685.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 14:15:00 | 5728.45 | 5729.16 | 5711.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-19 14:30:00 | 5731.45 | 5729.16 | 5711.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 5791.95 | 5821.24 | 5810.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 15:00:00 | 5791.95 | 5821.24 | 5810.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 5789.00 | 5814.79 | 5808.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:15:00 | 5799.80 | 5814.79 | 5808.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 12:15:00 | 5808.05 | 5821.07 | 5814.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:00:00 | 5808.05 | 5821.07 | 5814.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 13:15:00 | 5812.00 | 5819.25 | 5814.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 13:45:00 | 5818.65 | 5819.25 | 5814.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 14:15:00 | 5794.65 | 5814.33 | 5812.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-26 14:45:00 | 5792.60 | 5814.33 | 5812.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 15:15:00 | 5805.00 | 5812.47 | 5811.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-27 09:15:00 | 5807.50 | 5812.47 | 5811.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 09:15:00 | 5778.00 | 5805.57 | 5808.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 09:15:00 | 5778.00 | 5805.57 | 5808.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 10:15:00 | 5767.40 | 5797.94 | 5804.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 09:15:00 | 5730.35 | 5723.76 | 5748.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-29 09:45:00 | 5743.40 | 5723.76 | 5748.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 5789.00 | 5739.25 | 5751.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 12:00:00 | 5789.00 | 5739.25 | 5751.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 12:15:00 | 5822.95 | 5755.99 | 5758.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-29 12:45:00 | 5824.85 | 5755.99 | 5758.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2024-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 13:15:00 | 5803.50 | 5765.49 | 5762.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-29 15:15:00 | 5842.85 | 5790.48 | 5774.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 10:15:00 | 5800.00 | 5803.05 | 5783.94 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-30 11:00:00 | 5800.00 | 5803.05 | 5783.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 5875.10 | 5900.87 | 5881.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-04 14:00:00 | 5916.85 | 5901.43 | 5887.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-05 12:15:00 | 5846.70 | 5881.73 | 5884.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 12:15:00 | 5846.70 | 5881.73 | 5884.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 15:15:00 | 5840.20 | 5856.48 | 5865.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 09:15:00 | 5887.55 | 5862.70 | 5867.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 09:15:00 | 5887.55 | 5862.70 | 5867.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 5887.55 | 5862.70 | 5867.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 09:45:00 | 5888.00 | 5862.70 | 5867.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 10:15:00 | 5896.95 | 5869.55 | 5870.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-09 10:30:00 | 5898.55 | 5869.55 | 5870.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 11:15:00 | 5920.10 | 5879.66 | 5875.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-09 14:15:00 | 5932.50 | 5901.61 | 5887.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 12:15:00 | 6068.70 | 6069.72 | 6037.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-13 13:00:00 | 6068.70 | 6069.72 | 6037.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 6072.20 | 6096.63 | 6063.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-16 10:30:00 | 6118.85 | 6095.50 | 6065.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-17 09:15:00 | 6202.10 | 6069.94 | 6061.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 13:45:00 | 6100.75 | 6117.54 | 6105.90 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-18 14:45:00 | 6116.55 | 6117.22 | 6106.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-18 15:15:00 | 6102.40 | 6114.26 | 6106.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-19 09:15:00 | 6146.35 | 6114.26 | 6106.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-20 09:15:00 | 6149.20 | 6141.88 | 6130.42 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-25 09:15:00 | 6134.50 | 6189.66 | 6190.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — SELL (started 2024-09-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 09:15:00 | 6134.50 | 6189.66 | 6190.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 11:15:00 | 6120.00 | 6165.94 | 6178.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-25 14:15:00 | 6186.95 | 6164.93 | 6174.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 14:15:00 | 6186.95 | 6164.93 | 6174.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 14:15:00 | 6186.95 | 6164.93 | 6174.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-25 15:00:00 | 6186.95 | 6164.93 | 6174.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 15:15:00 | 6170.00 | 6165.94 | 6174.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:15:00 | 6211.25 | 6165.94 | 6174.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 6161.50 | 6165.06 | 6173.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:45:00 | 6175.00 | 6165.06 | 6173.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 6165.35 | 6165.11 | 6172.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:30:00 | 6178.00 | 6165.11 | 6172.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 6192.90 | 6170.67 | 6174.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 12:00:00 | 6192.90 | 6170.67 | 6174.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 29 — BUY (started 2024-09-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 12:15:00 | 6217.50 | 6180.04 | 6178.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-26 14:15:00 | 6263.15 | 6201.61 | 6188.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-27 14:15:00 | 6269.25 | 6271.62 | 6237.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-27 15:00:00 | 6269.25 | 6271.62 | 6237.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 09:15:00 | 6316.85 | 6279.05 | 6246.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-30 14:15:00 | 6354.25 | 6309.37 | 6272.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 09:30:00 | 6358.00 | 6323.69 | 6289.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-01 11:45:00 | 6363.10 | 6337.81 | 6301.87 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-03 13:00:00 | 6366.70 | 6383.22 | 6353.51 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 13:15:00 | 6318.35 | 6370.25 | 6350.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-03 13:45:00 | 6305.70 | 6370.25 | 6350.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 14:15:00 | 6333.65 | 6362.93 | 6348.80 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-10-04 10:15:00 | 6305.00 | 6338.70 | 6340.03 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — SELL (started 2024-10-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 10:15:00 | 6305.00 | 6338.70 | 6340.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-04 12:15:00 | 6261.20 | 6319.01 | 6330.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 13:15:00 | 6141.60 | 6125.22 | 6175.23 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 13:45:00 | 6136.90 | 6125.22 | 6175.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 14:15:00 | 6209.90 | 6142.16 | 6178.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 15:00:00 | 6209.90 | 6142.16 | 6178.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 6198.00 | 6153.33 | 6180.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-09 09:15:00 | 6146.00 | 6153.33 | 6180.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-15 13:15:00 | 6055.50 | 6017.37 | 6016.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2024-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-15 13:15:00 | 6055.50 | 6017.37 | 6016.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-16 09:15:00 | 6098.90 | 6045.16 | 6030.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-16 11:15:00 | 6040.95 | 6045.94 | 6033.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-16 11:15:00 | 6040.95 | 6045.94 | 6033.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 11:15:00 | 6040.95 | 6045.94 | 6033.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 11:45:00 | 6037.25 | 6045.94 | 6033.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-16 12:15:00 | 6080.05 | 6052.76 | 6037.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-16 12:30:00 | 6046.05 | 6052.76 | 6037.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 09:15:00 | 6019.50 | 6056.15 | 6045.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-17 10:00:00 | 6019.50 | 6056.15 | 6045.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-17 10:15:00 | 6021.50 | 6049.22 | 6043.22 | EMA400 retest candle locked (from upside) |

### Cycle 32 — SELL (started 2024-10-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 12:15:00 | 5991.20 | 6033.58 | 6036.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-18 09:15:00 | 5953.55 | 6000.12 | 6018.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-23 09:15:00 | 5763.10 | 5760.16 | 5809.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-23 09:45:00 | 5773.30 | 5760.16 | 5809.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-25 15:15:00 | 5682.00 | 5656.45 | 5685.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 09:15:00 | 5680.90 | 5656.45 | 5685.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 09:15:00 | 5734.95 | 5672.15 | 5689.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 10:00:00 | 5734.95 | 5672.15 | 5689.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-28 10:15:00 | 5765.40 | 5690.80 | 5696.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-28 11:15:00 | 5768.50 | 5690.80 | 5696.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — BUY (started 2024-10-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-28 11:15:00 | 5769.45 | 5706.53 | 5703.14 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2024-10-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-29 09:15:00 | 5637.05 | 5705.69 | 5706.11 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 5784.15 | 5710.04 | 5702.94 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2024-11-01 18:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-01 18:15:00 | 5698.85 | 5727.88 | 5728.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-04 09:15:00 | 5626.20 | 5707.55 | 5718.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-05 09:15:00 | 5647.90 | 5642.03 | 5670.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-05 09:45:00 | 5662.30 | 5642.03 | 5670.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-05 10:15:00 | 5655.00 | 5644.62 | 5669.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-05 11:00:00 | 5655.00 | 5644.62 | 5669.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 09:15:00 | 5645.70 | 5627.31 | 5647.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 09:30:00 | 5657.70 | 5627.31 | 5647.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 10:15:00 | 5648.00 | 5631.45 | 5647.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 10:45:00 | 5663.80 | 5631.45 | 5647.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 11:15:00 | 5644.85 | 5634.13 | 5647.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 11:45:00 | 5651.95 | 5634.13 | 5647.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 12:15:00 | 5664.30 | 5640.16 | 5648.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 13:00:00 | 5664.30 | 5640.16 | 5648.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 13:15:00 | 5706.60 | 5653.45 | 5653.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-06 14:00:00 | 5706.60 | 5653.45 | 5653.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 37 — BUY (started 2024-11-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 14:15:00 | 5692.20 | 5661.20 | 5657.33 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2024-11-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 10:15:00 | 5632.80 | 5655.63 | 5656.02 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2024-11-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-07 11:15:00 | 5682.80 | 5661.06 | 5658.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-08 11:15:00 | 5722.50 | 5684.07 | 5672.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-11 10:15:00 | 5723.25 | 5734.89 | 5706.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-11 11:00:00 | 5723.25 | 5734.89 | 5706.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 11:15:00 | 5635.90 | 5715.09 | 5699.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-11 12:00:00 | 5635.90 | 5715.09 | 5699.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2024-11-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-11 12:15:00 | 5518.55 | 5675.78 | 5683.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-11 13:15:00 | 5436.60 | 5627.95 | 5661.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-19 09:15:00 | 4921.65 | 4917.23 | 4987.00 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2024-11-19 15:00:00 | 4884.05 | 4916.38 | 4960.86 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 12:15:00 | 4852.65 | 4827.78 | 4861.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 12:45:00 | 4860.10 | 4827.78 | 4861.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 13:15:00 | 4860.00 | 4834.22 | 4860.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:00:00 | 4860.00 | 4834.22 | 4860.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-22 14:15:00 | 4852.60 | 4837.90 | 4860.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-22 14:45:00 | 4864.75 | 4837.90 | 4860.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-25 09:15:00 | 4923.45 | 4856.94 | 4865.06 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-25 09:15:00 | 4923.45 | 4856.94 | 4865.06 | SL hit (close>ema400) qty=1.00 sl=4865.06 alert=retest1 |

### Cycle 41 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 4956.80 | 4876.91 | 4873.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 11:15:00 | 4976.60 | 4896.85 | 4882.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-25 14:15:00 | 4900.45 | 4911.10 | 4894.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-25 15:00:00 | 4900.45 | 4911.10 | 4894.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-26 09:15:00 | 4966.10 | 4923.20 | 4902.55 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-26 11:15:00 | 4983.65 | 4932.45 | 4908.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-27 13:00:00 | 4984.10 | 4974.66 | 4953.57 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-28 12:15:00 | 4913.00 | 4955.36 | 4955.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2024-11-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-28 12:15:00 | 4913.00 | 4955.36 | 4955.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-03 09:15:00 | 4893.45 | 4919.22 | 4929.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-03 14:15:00 | 4913.40 | 4909.73 | 4920.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-03 15:00:00 | 4913.40 | 4909.73 | 4920.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 09:15:00 | 4935.85 | 4915.16 | 4921.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-04 09:30:00 | 4948.00 | 4915.16 | 4921.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-04 10:15:00 | 4913.00 | 4914.72 | 4920.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-04 11:30:00 | 4903.35 | 4910.16 | 4917.75 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-11 11:15:00 | 4883.00 | 4836.02 | 4829.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 43 — BUY (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 11:15:00 | 4883.00 | 4836.02 | 4829.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-12 10:15:00 | 4900.00 | 4872.99 | 4853.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 11:15:00 | 4828.65 | 4864.12 | 4851.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-12 11:15:00 | 4828.65 | 4864.12 | 4851.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 11:15:00 | 4828.65 | 4864.12 | 4851.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 12:00:00 | 4828.65 | 4864.12 | 4851.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-12 12:15:00 | 4817.15 | 4854.73 | 4848.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-12 12:45:00 | 4817.00 | 4854.73 | 4848.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2024-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 15:15:00 | 4820.00 | 4840.47 | 4842.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-13 09:15:00 | 4794.00 | 4831.17 | 4838.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 12:15:00 | 4831.50 | 4825.70 | 4833.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 12:15:00 | 4831.50 | 4825.70 | 4833.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 4831.50 | 4825.70 | 4833.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-16 11:00:00 | 4793.50 | 4826.29 | 4832.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-16 14:15:00 | 4849.95 | 4832.74 | 4833.41 | SL hit (close>static) qty=1.00 sl=4839.35 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-26 11:15:00 | 4757.50 | 4739.81 | 4738.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-26 12:15:00 | 4781.05 | 4748.06 | 4742.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-27 14:15:00 | 4762.65 | 4768.54 | 4759.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-27 14:15:00 | 4762.65 | 4768.54 | 4759.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 4762.65 | 4768.54 | 4759.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-27 14:45:00 | 4766.00 | 4768.54 | 4759.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 4770.00 | 4768.83 | 4760.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:15:00 | 4773.50 | 4768.83 | 4760.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-30 09:15:00 | 4770.00 | 4769.07 | 4760.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 10:30:00 | 4784.45 | 4772.19 | 4763.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-30 14:45:00 | 4781.75 | 4775.26 | 4767.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 10:00:00 | 4782.20 | 4770.11 | 4767.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-02 10:15:00 | 4743.05 | 4774.87 | 4775.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — SELL (started 2025-01-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-02 10:15:00 | 4743.05 | 4774.87 | 4775.41 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-01-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 15:15:00 | 4785.00 | 4773.12 | 4773.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 09:15:00 | 4811.55 | 4780.81 | 4776.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-06 10:15:00 | 4810.00 | 4820.35 | 4805.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 11:00:00 | 4810.00 | 4820.35 | 4805.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 11:15:00 | 4827.05 | 4821.69 | 4807.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 12:45:00 | 4832.40 | 4821.12 | 4808.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 14:15:00 | 4784.00 | 4811.48 | 4805.90 | SL hit (close<static) qty=1.00 sl=4791.85 alert=retest2 |

### Cycle 48 — SELL (started 2025-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-08 11:15:00 | 4801.05 | 4818.91 | 4819.79 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-01-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-08 13:15:00 | 4840.30 | 4823.27 | 4821.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-08 14:15:00 | 4863.65 | 4831.35 | 4825.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-10 09:15:00 | 4901.70 | 4912.34 | 4884.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-10 09:15:00 | 4901.70 | 4912.34 | 4884.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-10 09:15:00 | 4901.70 | 4912.34 | 4884.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-10 09:30:00 | 4903.50 | 4912.34 | 4884.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 09:15:00 | 4977.00 | 4941.75 | 4914.65 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 12:15:00 | 4895.85 | 4915.40 | 4916.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-14 13:15:00 | 4872.65 | 4906.85 | 4912.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 14:15:00 | 4879.90 | 4873.37 | 4886.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-15 14:15:00 | 4879.90 | 4873.37 | 4886.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 14:15:00 | 4879.90 | 4873.37 | 4886.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-15 15:00:00 | 4879.90 | 4873.37 | 4886.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 15:15:00 | 4873.00 | 4873.29 | 4885.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 09:15:00 | 4861.75 | 4873.29 | 4885.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 4824.10 | 4863.45 | 4879.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 10:15:00 | 4805.15 | 4863.45 | 4879.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 12:45:00 | 4812.40 | 4837.56 | 4862.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-20 10:15:00 | 4881.60 | 4859.54 | 4859.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 10:15:00 | 4881.60 | 4859.54 | 4859.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-20 13:15:00 | 4907.85 | 4874.27 | 4866.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 15:15:00 | 4867.50 | 4874.99 | 4868.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 15:15:00 | 4867.50 | 4874.99 | 4868.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 15:15:00 | 4867.50 | 4874.99 | 4868.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 09:15:00 | 4900.00 | 4874.99 | 4868.20 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-21 09:45:00 | 4895.95 | 4878.98 | 4870.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-21 10:15:00 | 4856.80 | 4874.55 | 4869.37 | SL hit (close<static) qty=1.00 sl=4867.15 alert=retest2 |

### Cycle 52 — SELL (started 2025-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-29 09:15:00 | 5016.00 | 5080.03 | 5086.27 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2025-01-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-30 15:15:00 | 5104.95 | 5073.37 | 5069.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-31 11:15:00 | 5145.00 | 5091.60 | 5079.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 5125.75 | 5126.73 | 5107.52 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 5125.75 | 5126.73 | 5107.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 5125.75 | 5126.73 | 5107.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 5083.95 | 5126.73 | 5107.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 5354.00 | 5172.18 | 5129.93 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2025-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 12:15:00 | 5063.00 | 5127.20 | 5132.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-04 12:15:00 | 5002.60 | 5065.20 | 5095.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-06 15:15:00 | 4968.95 | 4955.42 | 4985.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 09:15:00 | 5029.90 | 4955.42 | 4985.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 4961.20 | 4956.57 | 4983.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 10:15:00 | 4915.80 | 4956.57 | 4983.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 11:45:00 | 4960.00 | 4933.75 | 4947.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-13 10:15:00 | 4932.00 | 4916.85 | 4914.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-02-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-13 10:15:00 | 4932.00 | 4916.85 | 4914.91 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-13 14:15:00 | 4890.05 | 4911.15 | 4913.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-13 15:15:00 | 4874.00 | 4903.72 | 4909.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-14 09:15:00 | 4939.50 | 4910.88 | 4912.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-14 09:15:00 | 4939.50 | 4910.88 | 4912.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 09:15:00 | 4939.50 | 4910.88 | 4912.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-14 10:00:00 | 4939.50 | 4910.88 | 4912.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-14 10:15:00 | 4906.75 | 4910.05 | 4911.91 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-14 11:15:00 | 4895.00 | 4910.05 | 4911.91 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-14 14:15:00 | 4943.15 | 4914.07 | 4912.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-02-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-14 14:15:00 | 4943.15 | 4914.07 | 4912.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-17 14:15:00 | 4968.00 | 4936.18 | 4925.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-18 09:15:00 | 4922.25 | 4936.02 | 4927.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-18 09:15:00 | 4922.25 | 4936.02 | 4927.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 09:15:00 | 4922.25 | 4936.02 | 4927.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 09:30:00 | 4914.00 | 4936.02 | 4927.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 10:15:00 | 4913.70 | 4931.56 | 4926.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 10:45:00 | 4917.20 | 4931.56 | 4926.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 12:15:00 | 4903.00 | 4926.64 | 4925.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 13:00:00 | 4903.00 | 4926.64 | 4925.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-18 13:15:00 | 4921.25 | 4925.56 | 4924.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-18 14:15:00 | 4895.05 | 4925.56 | 4924.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2025-02-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-18 14:15:00 | 4896.50 | 4919.75 | 4922.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-18 15:15:00 | 4880.00 | 4911.80 | 4918.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-20 14:15:00 | 4833.90 | 4830.20 | 4854.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-20 15:00:00 | 4833.90 | 4830.20 | 4854.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 4791.85 | 4822.98 | 4847.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 09:15:00 | 4768.55 | 4806.41 | 4818.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 13:15:00 | 4775.65 | 4802.18 | 4810.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 14:15:00 | 4772.25 | 4798.60 | 4807.72 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-28 09:15:00 | 4766.30 | 4797.39 | 4805.46 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 09:15:00 | 4530.12 | 4602.67 | 4657.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 09:15:00 | 4536.87 | 4602.67 | 4657.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 09:15:00 | 4533.64 | 4602.67 | 4657.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-03-04 09:15:00 | 4527.98 | 4602.67 | 4657.38 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-03-04 14:15:00 | 4574.90 | 4574.10 | 4619.80 | SL hit (close>ema200) qty=0.50 sl=4574.10 alert=retest2 |

### Cycle 59 — BUY (started 2025-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 13:15:00 | 4700.10 | 4637.03 | 4632.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-05 14:15:00 | 4725.05 | 4654.63 | 4641.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-06 14:15:00 | 4697.00 | 4724.22 | 4690.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-06 14:15:00 | 4697.00 | 4724.22 | 4690.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 14:15:00 | 4697.00 | 4724.22 | 4690.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-06 14:30:00 | 4724.00 | 4724.22 | 4690.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-06 15:15:00 | 4692.00 | 4717.78 | 4690.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:15:00 | 4691.80 | 4717.78 | 4690.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 09:15:00 | 4730.00 | 4720.22 | 4694.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-07 09:45:00 | 4692.00 | 4720.22 | 4694.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 4808.00 | 4756.17 | 4727.14 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 10:00:00 | 4829.00 | 4782.57 | 4767.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 11:00:00 | 4830.70 | 4792.20 | 4773.22 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-13 13:15:00 | 4727.15 | 4763.34 | 4763.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 60 — SELL (started 2025-03-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-13 13:15:00 | 4727.15 | 4763.34 | 4763.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-17 09:15:00 | 4665.95 | 4734.07 | 4749.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-17 15:15:00 | 4685.50 | 4683.88 | 4711.74 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-18 09:15:00 | 4739.40 | 4683.88 | 4711.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 4732.05 | 4693.52 | 4713.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 09:30:00 | 4754.15 | 4693.52 | 4713.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 10:15:00 | 4741.10 | 4703.03 | 4716.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-18 11:00:00 | 4741.10 | 4703.03 | 4716.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 12:15:00 | 4765.00 | 4723.78 | 4723.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-18 14:15:00 | 4774.50 | 4740.49 | 4731.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-19 10:15:00 | 4733.70 | 4743.00 | 4735.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-19 10:15:00 | 4733.70 | 4743.00 | 4735.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 10:15:00 | 4733.70 | 4743.00 | 4735.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 11:00:00 | 4733.70 | 4743.00 | 4735.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 11:15:00 | 4730.00 | 4740.40 | 4735.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 11:45:00 | 4725.50 | 4740.40 | 4735.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 12:15:00 | 4734.90 | 4739.30 | 4735.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 13:00:00 | 4734.90 | 4739.30 | 4735.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 13:15:00 | 4725.05 | 4736.45 | 4734.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 13:45:00 | 4724.90 | 4736.45 | 4734.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 14:15:00 | 4721.95 | 4733.55 | 4733.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-19 15:00:00 | 4721.95 | 4733.55 | 4733.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 4755.00 | 4738.59 | 4735.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 11:30:00 | 4793.50 | 4758.59 | 4745.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 10:00:00 | 4787.70 | 4807.86 | 4792.48 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-24 11:15:00 | 4799.00 | 4801.52 | 4790.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 09:30:00 | 4782.65 | 4797.42 | 4793.87 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-25 10:15:00 | 4803.75 | 4798.69 | 4794.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-25 12:00:00 | 4834.40 | 4805.83 | 4798.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-26 10:45:00 | 4836.95 | 4830.49 | 4816.68 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-27 09:30:00 | 4837.05 | 4847.10 | 4833.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-07 10:15:00 | 4966.40 | 5007.94 | 5009.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-04-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-07 10:15:00 | 4966.40 | 5007.94 | 5009.85 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-04-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-07 14:15:00 | 5055.00 | 5015.72 | 5011.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-08 10:15:00 | 5086.10 | 5043.09 | 5026.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-15 10:15:00 | 5315.20 | 5333.23 | 5280.11 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-15 11:00:00 | 5315.20 | 5333.23 | 5280.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 14:15:00 | 5389.10 | 5419.93 | 5409.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-21 15:00:00 | 5389.10 | 5419.93 | 5409.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 15:15:00 | 5394.00 | 5414.74 | 5407.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 09:15:00 | 5429.80 | 5414.74 | 5407.79 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 10:15:00 | 5396.30 | 5410.57 | 5406.53 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 11:45:00 | 5410.80 | 5414.71 | 5409.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-25 09:15:00 | 5409.10 | 5456.30 | 5456.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-04-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 09:15:00 | 5409.10 | 5456.30 | 5456.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 10:15:00 | 5386.00 | 5442.24 | 5450.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-25 14:15:00 | 5423.70 | 5423.47 | 5437.17 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-25 14:15:00 | 5423.70 | 5423.47 | 5437.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 14:15:00 | 5423.70 | 5423.47 | 5437.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-25 14:45:00 | 5430.90 | 5423.47 | 5437.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 09:15:00 | 5467.80 | 5430.98 | 5438.12 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2025-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-29 10:15:00 | 5501.60 | 5445.96 | 5441.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-30 09:15:00 | 5523.00 | 5484.73 | 5465.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-30 11:15:00 | 5480.10 | 5485.03 | 5469.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-30 12:00:00 | 5480.10 | 5485.03 | 5469.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 12:15:00 | 5449.00 | 5477.83 | 5467.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 13:00:00 | 5449.00 | 5477.83 | 5467.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-30 13:15:00 | 5448.50 | 5471.96 | 5465.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-30 13:45:00 | 5447.20 | 5471.96 | 5465.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2025-04-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 15:15:00 | 5415.10 | 5461.06 | 5461.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 09:15:00 | 5370.50 | 5442.95 | 5453.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 5358.00 | 5347.01 | 5388.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-05 10:00:00 | 5358.00 | 5347.01 | 5388.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 09:15:00 | 5335.50 | 5350.35 | 5370.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 09:30:00 | 5355.00 | 5350.35 | 5370.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 5369.00 | 5354.62 | 5367.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:00:00 | 5369.00 | 5354.62 | 5367.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 5395.00 | 5362.70 | 5370.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 13:45:00 | 5407.00 | 5362.70 | 5370.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 5397.50 | 5369.66 | 5372.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-06 15:00:00 | 5397.50 | 5369.66 | 5372.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 15:15:00 | 5390.00 | 5373.73 | 5374.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 09:15:00 | 5374.50 | 5373.73 | 5374.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 10:45:00 | 5371.50 | 5374.20 | 5374.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-07 11:15:00 | 5367.50 | 5374.20 | 5374.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-07 11:15:00 | 5375.50 | 5374.46 | 5374.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-07 11:15:00 | 5375.50 | 5374.46 | 5374.34 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-05-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-07 12:15:00 | 5353.00 | 5370.17 | 5372.40 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-05-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 11:15:00 | 5429.50 | 5376.71 | 5372.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-08 12:15:00 | 5456.00 | 5392.57 | 5380.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 15:15:00 | 5388.50 | 5394.57 | 5384.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-08 15:15:00 | 5388.50 | 5394.57 | 5384.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 15:15:00 | 5388.50 | 5394.57 | 5384.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-09 09:15:00 | 5457.00 | 5394.57 | 5384.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 09:15:00 | 5439.50 | 5403.55 | 5389.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 09:30:00 | 5490.00 | 5444.65 | 5420.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-12 10:00:00 | 5493.50 | 5444.65 | 5420.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 12:15:00 | 5488.50 | 5526.47 | 5491.83 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-13 14:30:00 | 5491.50 | 5502.99 | 5488.46 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 5488.00 | 5500.00 | 5488.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 09:15:00 | 5553.00 | 5500.00 | 5488.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 5542.00 | 5508.40 | 5493.29 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-05-14 13:15:00 | 5446.00 | 5483.30 | 5485.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2025-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 13:15:00 | 5446.00 | 5483.30 | 5485.48 | EMA200 below EMA400 |

### Cycle 71 — BUY (started 2025-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 10:15:00 | 5494.50 | 5487.91 | 5487.11 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2025-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 11:15:00 | 5475.00 | 5485.33 | 5486.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 13:15:00 | 5468.00 | 5481.97 | 5484.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 10:15:00 | 5495.50 | 5480.00 | 5482.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 10:15:00 | 5495.50 | 5480.00 | 5482.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 5495.50 | 5480.00 | 5482.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:45:00 | 5513.00 | 5480.00 | 5482.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-05-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 11:15:00 | 5501.00 | 5484.20 | 5483.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-16 12:15:00 | 5507.50 | 5488.86 | 5485.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 10:15:00 | 5493.00 | 5495.88 | 5491.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 10:15:00 | 5493.00 | 5495.88 | 5491.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 5493.00 | 5495.88 | 5491.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:45:00 | 5493.00 | 5495.88 | 5491.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 5475.00 | 5491.71 | 5489.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 11:30:00 | 5473.50 | 5491.71 | 5489.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 12:15:00 | 5500.50 | 5493.46 | 5490.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 14:15:00 | 5510.00 | 5495.77 | 5491.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 10:00:00 | 5510.50 | 5504.08 | 5497.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 12:00:00 | 5513.00 | 5513.13 | 5502.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 12:15:00 | 5461.50 | 5502.80 | 5499.15 | SL hit (close<static) qty=1.00 sl=5475.00 alert=retest2 |

### Cycle 74 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 5460.50 | 5494.34 | 5495.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 5429.00 | 5481.27 | 5489.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 5497.00 | 5474.62 | 5484.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 5497.00 | 5474.62 | 5484.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 5497.00 | 5474.62 | 5484.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 5497.00 | 5474.62 | 5484.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 5493.50 | 5478.39 | 5485.32 | EMA400 retest candle locked (from downside) |

### Cycle 75 — BUY (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 13:15:00 | 5503.00 | 5490.74 | 5489.73 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 5464.00 | 5484.41 | 5487.05 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-05-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 15:15:00 | 5490.00 | 5476.82 | 5475.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 5518.50 | 5485.16 | 5479.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 11:15:00 | 5520.00 | 5520.29 | 5505.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-27 12:00:00 | 5520.00 | 5520.29 | 5505.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 12:15:00 | 5488.00 | 5513.83 | 5504.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 13:00:00 | 5488.00 | 5513.83 | 5504.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 13:15:00 | 5482.50 | 5507.56 | 5502.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 14:00:00 | 5482.50 | 5507.56 | 5502.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 5504.00 | 5504.12 | 5501.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-28 09:15:00 | 5472.50 | 5504.12 | 5501.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 09:15:00 | 5446.00 | 5492.50 | 5496.46 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 09:15:00 | 5506.50 | 5492.19 | 5491.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 09:15:00 | 5672.00 | 5533.84 | 5512.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-03 09:15:00 | 5591.00 | 5595.56 | 5562.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-03 09:15:00 | 5591.00 | 5595.56 | 5562.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 09:15:00 | 5591.00 | 5595.56 | 5562.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 09:30:00 | 5567.00 | 5595.56 | 5562.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 10:15:00 | 5561.00 | 5588.65 | 5562.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 11:00:00 | 5561.00 | 5588.65 | 5562.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 11:15:00 | 5557.50 | 5582.42 | 5561.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-03 12:00:00 | 5557.50 | 5582.42 | 5561.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-03 12:15:00 | 5584.00 | 5582.74 | 5563.82 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2025-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 12:15:00 | 5545.00 | 5559.25 | 5560.03 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-06-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 12:15:00 | 5586.50 | 5556.40 | 5556.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-05 14:15:00 | 5599.50 | 5566.72 | 5561.06 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 11:15:00 | 5575.50 | 5578.96 | 5569.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 11:15:00 | 5575.50 | 5578.96 | 5569.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 11:15:00 | 5575.50 | 5578.96 | 5569.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 11:45:00 | 5574.50 | 5578.96 | 5569.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 12:15:00 | 5565.50 | 5576.27 | 5569.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 12:30:00 | 5566.00 | 5576.27 | 5569.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 13:15:00 | 5569.00 | 5574.81 | 5569.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-06 14:30:00 | 5581.00 | 5581.45 | 5572.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 11:15:00 | 5629.00 | 5649.82 | 5650.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 82 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 5629.00 | 5649.82 | 5650.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 5584.50 | 5633.35 | 5643.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 5584.50 | 5562.30 | 5591.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 14:15:00 | 5584.50 | 5562.30 | 5591.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 14:15:00 | 5584.50 | 5562.30 | 5591.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-13 15:00:00 | 5584.50 | 5562.30 | 5591.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 5565.00 | 5555.40 | 5570.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:15:00 | 5545.50 | 5555.40 | 5570.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 09:45:00 | 5551.00 | 5553.32 | 5568.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-18 09:15:00 | 5589.50 | 5563.01 | 5565.18 | SL hit (close>static) qty=1.00 sl=5572.00 alert=retest2 |

### Cycle 83 — BUY (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 11:15:00 | 5568.50 | 5566.82 | 5566.71 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2025-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 13:15:00 | 5560.50 | 5565.99 | 5566.37 | EMA200 below EMA400 |

### Cycle 85 — BUY (started 2025-06-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 14:15:00 | 5574.00 | 5567.59 | 5567.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-19 10:15:00 | 5585.00 | 5574.57 | 5570.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 12:15:00 | 5553.00 | 5575.44 | 5572.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-19 12:15:00 | 5553.00 | 5575.44 | 5572.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 5553.00 | 5575.44 | 5572.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 5553.00 | 5575.44 | 5572.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 5560.00 | 5572.35 | 5570.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 5547.00 | 5572.35 | 5570.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 86 — SELL (started 2025-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 15:15:00 | 5532.00 | 5563.19 | 5566.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 10:15:00 | 5520.00 | 5549.56 | 5559.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-20 12:15:00 | 5555.00 | 5550.24 | 5558.33 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-20 12:15:00 | 5555.00 | 5550.24 | 5558.33 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 12:15:00 | 5555.00 | 5550.24 | 5558.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 12:30:00 | 5559.50 | 5550.24 | 5558.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 13:15:00 | 5581.00 | 5556.39 | 5560.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 14:00:00 | 5581.00 | 5556.39 | 5560.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 5584.50 | 5562.01 | 5562.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 15:00:00 | 5584.50 | 5562.01 | 5562.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 15:15:00 | 5561.50 | 5561.91 | 5562.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-23 09:15:00 | 5490.00 | 5561.91 | 5562.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 15:15:00 | 5565.50 | 5557.34 | 5556.56 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 87 — BUY (started 2025-06-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 15:15:00 | 5565.50 | 5557.34 | 5556.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 5588.00 | 5563.48 | 5559.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 13:15:00 | 5581.00 | 5585.89 | 5573.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 13:15:00 | 5581.00 | 5585.89 | 5573.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 13:15:00 | 5581.00 | 5585.89 | 5573.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-24 13:45:00 | 5594.00 | 5585.89 | 5573.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 10:15:00 | 5768.50 | 5806.28 | 5786.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 11:00:00 | 5768.50 | 5806.28 | 5786.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-01 11:15:00 | 5758.50 | 5796.72 | 5783.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-01 12:00:00 | 5758.50 | 5796.72 | 5783.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — SELL (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 14:15:00 | 5741.00 | 5771.51 | 5774.24 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 10:15:00 | 5821.50 | 5784.40 | 5779.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-03 09:15:00 | 5867.00 | 5809.28 | 5795.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-03 13:15:00 | 5816.00 | 5822.60 | 5807.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-03 14:00:00 | 5816.00 | 5822.60 | 5807.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 14:15:00 | 5795.00 | 5817.08 | 5806.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-03 14:30:00 | 5794.50 | 5817.08 | 5806.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 15:15:00 | 5792.00 | 5812.06 | 5804.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-04 09:15:00 | 5838.00 | 5812.06 | 5804.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 12:15:00 | 5769.50 | 5801.75 | 5802.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 90 — SELL (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 12:15:00 | 5769.50 | 5801.75 | 5802.69 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2025-07-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-07 11:15:00 | 5822.50 | 5801.49 | 5800.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-07 13:15:00 | 5853.50 | 5814.13 | 5806.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-08 11:15:00 | 5846.50 | 5848.85 | 5829.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-08 11:30:00 | 5839.00 | 5848.85 | 5829.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 13:15:00 | 5830.00 | 5842.94 | 5830.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 13:30:00 | 5819.00 | 5842.94 | 5830.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 14:15:00 | 5840.50 | 5842.46 | 5831.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 14:45:00 | 5835.00 | 5842.46 | 5831.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 15:15:00 | 5825.00 | 5838.96 | 5830.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:15:00 | 5857.00 | 5838.96 | 5830.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 5872.00 | 5845.57 | 5834.52 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 10:30:00 | 5873.00 | 5845.46 | 5835.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 15:00:00 | 5896.50 | 5856.28 | 5843.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 11:00:00 | 5915.00 | 5873.42 | 5855.18 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-10 12:15:00 | 5890.00 | 5869.33 | 5854.98 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-10 12:15:00 | 5874.50 | 5870.37 | 5856.76 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-11 12:15:00 | 5734.00 | 5835.03 | 5847.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-07-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 12:15:00 | 5734.00 | 5835.03 | 5847.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-11 13:15:00 | 5730.50 | 5814.12 | 5836.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 13:15:00 | 5786.00 | 5783.47 | 5805.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-14 14:00:00 | 5786.00 | 5783.47 | 5805.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 5776.50 | 5782.07 | 5803.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 14:45:00 | 5778.00 | 5782.07 | 5803.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 5806.50 | 5785.03 | 5800.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:00:00 | 5806.50 | 5785.03 | 5800.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 10:15:00 | 5807.00 | 5789.42 | 5801.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:00:00 | 5807.00 | 5789.42 | 5801.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 11:15:00 | 5809.00 | 5793.34 | 5802.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 11:30:00 | 5814.50 | 5793.34 | 5802.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 12:15:00 | 5805.00 | 5795.67 | 5802.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 12:30:00 | 5815.50 | 5795.67 | 5802.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 13:15:00 | 5789.00 | 5794.34 | 5801.09 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 10:45:00 | 5775.50 | 5792.82 | 5798.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 11:15:00 | 5772.00 | 5792.82 | 5798.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 14:15:00 | 5775.00 | 5780.79 | 5790.23 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-16 15:15:00 | 5770.00 | 5781.13 | 5789.52 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 09:15:00 | 5833.50 | 5789.82 | 5791.91 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-07-17 09:15:00 | 5833.50 | 5789.82 | 5791.91 | SL hit (close>static) qty=1.00 sl=5805.50 alert=retest2 |

### Cycle 93 — BUY (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 10:15:00 | 5829.50 | 5797.76 | 5795.33 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 5776.50 | 5795.05 | 5797.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 14:15:00 | 5730.00 | 5775.95 | 5787.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-22 12:15:00 | 5734.50 | 5696.67 | 5721.76 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-22 12:15:00 | 5734.50 | 5696.67 | 5721.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 12:15:00 | 5734.50 | 5696.67 | 5721.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-22 13:00:00 | 5734.50 | 5696.67 | 5721.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 13:15:00 | 5729.50 | 5703.23 | 5722.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 15:00:00 | 5708.50 | 5704.29 | 5721.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:15:00 | 5702.50 | 5706.23 | 5720.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 5702.00 | 5705.99 | 5711.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 09:15:00 | 5684.00 | 5634.42 | 5633.75 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 95 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 5684.00 | 5634.42 | 5633.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 10:15:00 | 5707.00 | 5648.93 | 5640.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 5757.50 | 5766.25 | 5726.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 15:00:00 | 5757.50 | 5766.25 | 5726.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 5771.50 | 5803.49 | 5776.95 | EMA400 retest candle locked (from upside) |

### Cycle 96 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 5668.00 | 5760.01 | 5766.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 13:15:00 | 5631.50 | 5703.26 | 5734.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 09:15:00 | 5475.00 | 5468.97 | 5562.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-07 09:30:00 | 5524.50 | 5468.97 | 5562.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 5371.50 | 5341.19 | 5359.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 15:00:00 | 5371.50 | 5341.19 | 5359.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 15:15:00 | 5392.50 | 5351.45 | 5362.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 09:15:00 | 5372.00 | 5351.45 | 5362.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 10:15:00 | 5336.00 | 5349.17 | 5359.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-14 10:30:00 | 5344.00 | 5349.17 | 5359.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 97 — BUY (started 2025-08-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 09:15:00 | 5489.50 | 5357.78 | 5356.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 13:15:00 | 5702.50 | 5582.37 | 5510.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 13:15:00 | 5622.50 | 5639.89 | 5583.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 13:45:00 | 5622.00 | 5639.89 | 5583.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 5603.50 | 5626.31 | 5586.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 5571.00 | 5612.75 | 5584.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 10:15:00 | 5566.00 | 5603.40 | 5582.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 10:30:00 | 5565.00 | 5603.40 | 5582.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-22 12:15:00 | 5547.50 | 5583.68 | 5576.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 12:45:00 | 5538.00 | 5583.68 | 5576.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 98 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 5544.50 | 5570.93 | 5571.83 | EMA200 below EMA400 |

### Cycle 99 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 5594.50 | 5575.18 | 5573.57 | EMA200 above EMA400 |

### Cycle 100 — SELL (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 14:15:00 | 5549.50 | 5573.99 | 5574.73 | EMA200 below EMA400 |

### Cycle 101 — BUY (started 2025-08-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 09:15:00 | 5690.00 | 5593.27 | 5583.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-26 10:15:00 | 5699.00 | 5614.42 | 5593.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-28 14:15:00 | 5713.50 | 5730.43 | 5692.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-28 15:00:00 | 5713.50 | 5730.43 | 5692.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-29 09:15:00 | 5786.00 | 5740.03 | 5703.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 12:45:00 | 5911.50 | 5853.38 | 5818.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 14:30:00 | 5898.00 | 5864.86 | 5830.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-02 15:15:00 | 5895.00 | 5864.86 | 5830.40 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:45:00 | 5900.00 | 5878.31 | 5842.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 6014.50 | 6043.07 | 6001.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:45:00 | 6004.00 | 6043.07 | 6001.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 6203.00 | 6187.20 | 6154.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 09:30:00 | 6163.00 | 6187.20 | 6154.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 12:15:00 | 6242.50 | 6242.47 | 6213.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 13:30:00 | 6254.00 | 6243.57 | 6216.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 15:15:00 | 6251.50 | 6243.56 | 6219.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-15 14:15:00 | 6206.50 | 6230.71 | 6224.60 | SL hit (close<static) qty=1.00 sl=6207.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 12:15:00 | 6214.00 | 6220.31 | 6221.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 13:15:00 | 6199.00 | 6216.05 | 6219.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 15:15:00 | 6223.00 | 6215.51 | 6218.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 15:15:00 | 6223.00 | 6215.51 | 6218.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 6223.00 | 6215.51 | 6218.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 6106.00 | 6215.51 | 6218.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 11:15:00 | 6009.00 | 5948.08 | 5947.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 103 — BUY (started 2025-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-30 11:15:00 | 6009.00 | 5948.08 | 5947.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-30 12:15:00 | 6019.50 | 5962.36 | 5954.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-01 09:15:00 | 5925.50 | 5968.33 | 5961.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-01 09:15:00 | 5925.50 | 5968.33 | 5961.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 5925.50 | 5968.33 | 5961.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:00:00 | 5925.50 | 5968.33 | 5961.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 5920.00 | 5958.67 | 5957.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 5920.00 | 5958.67 | 5957.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 104 — SELL (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 11:15:00 | 5931.50 | 5953.23 | 5955.01 | EMA200 below EMA400 |

### Cycle 105 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 5973.00 | 5957.04 | 5956.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 09:15:00 | 5977.00 | 5961.04 | 5957.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-03 15:15:00 | 5985.00 | 5985.05 | 5973.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-03 15:15:00 | 5985.00 | 5985.05 | 5973.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 15:15:00 | 5985.00 | 5985.05 | 5973.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:30:00 | 5994.00 | 5983.44 | 5974.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 5965.50 | 5980.90 | 5974.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 11:30:00 | 5960.00 | 5980.90 | 5974.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 5974.00 | 5979.52 | 5974.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 13:45:00 | 5990.50 | 5982.82 | 5976.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 11:15:00 | 5927.00 | 5976.50 | 5977.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 106 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 5927.00 | 5976.50 | 5977.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 14:15:00 | 5890.00 | 5949.09 | 5963.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 12:15:00 | 5836.50 | 5835.50 | 5872.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 13:00:00 | 5836.50 | 5835.50 | 5872.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 5870.00 | 5842.40 | 5872.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:00:00 | 5870.00 | 5842.40 | 5872.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 5883.50 | 5850.62 | 5873.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 15:00:00 | 5883.50 | 5850.62 | 5873.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 15:15:00 | 5877.00 | 5855.90 | 5873.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:15:00 | 5895.50 | 5855.90 | 5873.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 5899.00 | 5864.52 | 5876.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-10 10:30:00 | 5868.00 | 5865.61 | 5875.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 11:00:00 | 5866.00 | 5874.27 | 5876.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 12:00:00 | 5866.50 | 5872.71 | 5875.53 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 12:30:00 | 5868.00 | 5872.27 | 5875.07 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 13:15:00 | 5864.50 | 5870.72 | 5874.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-14 10:00:00 | 5841.50 | 5863.00 | 5869.54 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 5904.50 | 5851.07 | 5845.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 5904.50 | 5851.07 | 5845.72 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 11:15:00 | 5944.50 | 5877.03 | 5858.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 14:15:00 | 6068.00 | 6095.04 | 6047.70 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 15:00:00 | 6068.00 | 6095.04 | 6047.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 6059.50 | 6094.06 | 6074.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 6059.50 | 6094.06 | 6074.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 6049.00 | 6085.05 | 6072.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 6005.50 | 6085.05 | 6072.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 6014.50 | 6070.94 | 6067.08 | EMA400 retest candle locked (from upside) |

### Cycle 108 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 6013.50 | 6059.45 | 6062.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 09:15:00 | 5964.00 | 6031.06 | 6045.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 15:15:00 | 5885.00 | 5872.20 | 5923.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-29 09:15:00 | 5876.50 | 5872.20 | 5923.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 15:15:00 | 5850.00 | 5841.21 | 5864.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 09:15:00 | 5878.50 | 5841.21 | 5864.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 09:15:00 | 5892.00 | 5851.37 | 5866.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-31 10:00:00 | 5892.00 | 5851.37 | 5866.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 10:15:00 | 5854.00 | 5851.89 | 5865.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 11:15:00 | 5847.00 | 5851.89 | 5865.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 12:30:00 | 5850.00 | 5854.11 | 5864.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 15:00:00 | 5836.50 | 5851.93 | 5861.66 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 11:15:00 | 5837.00 | 5835.03 | 5842.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 5860.00 | 5840.02 | 5843.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:00:00 | 5860.00 | 5840.02 | 5843.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 5856.50 | 5843.32 | 5845.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:30:00 | 5862.50 | 5843.32 | 5845.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-04 13:15:00 | 5873.00 | 5849.25 | 5847.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 13:15:00 | 5873.00 | 5849.25 | 5847.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 09:15:00 | 6076.00 | 5911.46 | 5877.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 13:15:00 | 6155.00 | 6158.81 | 6091.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 13:45:00 | 6162.00 | 6158.81 | 6091.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 5815.00 | 6083.19 | 6073.52 | EMA400 retest candle locked (from upside) |

### Cycle 110 — SELL (started 2025-11-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 10:15:00 | 5857.50 | 6038.05 | 6053.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 5800.00 | 5857.23 | 5900.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 09:15:00 | 5827.50 | 5817.18 | 5853.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 10:00:00 | 5827.50 | 5817.18 | 5853.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 11:15:00 | 5838.00 | 5821.32 | 5849.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 12:00:00 | 5838.00 | 5821.32 | 5849.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 5837.00 | 5828.58 | 5842.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:00:00 | 5837.00 | 5828.58 | 5842.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 11:15:00 | 5829.50 | 5828.99 | 5840.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 12:00:00 | 5829.50 | 5828.99 | 5840.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 13:15:00 | 5843.00 | 5831.71 | 5839.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 14:15:00 | 5835.00 | 5831.71 | 5839.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 5838.00 | 5832.97 | 5839.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-18 15:15:00 | 5852.00 | 5832.97 | 5839.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 5852.00 | 5836.78 | 5840.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 5827.00 | 5836.78 | 5840.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 10:00:00 | 5831.50 | 5835.72 | 5839.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 10:30:00 | 5831.50 | 5837.38 | 5840.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 11:15:00 | 5872.50 | 5844.40 | 5843.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 111 — BUY (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 11:15:00 | 5872.50 | 5844.40 | 5843.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 09:15:00 | 5885.00 | 5861.44 | 5852.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 11:15:00 | 5808.00 | 5853.00 | 5850.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 11:15:00 | 5808.00 | 5853.00 | 5850.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 5808.00 | 5853.00 | 5850.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:00:00 | 5808.00 | 5853.00 | 5850.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 112 — SELL (started 2025-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 12:15:00 | 5773.50 | 5837.10 | 5843.45 | EMA200 below EMA400 |

### Cycle 113 — BUY (started 2025-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 13:15:00 | 5853.50 | 5821.88 | 5819.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 14:15:00 | 5870.50 | 5831.60 | 5824.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 15:15:00 | 5868.00 | 5869.36 | 5852.28 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 09:15:00 | 5894.00 | 5869.36 | 5852.28 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 5844.00 | 5865.99 | 5855.15 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-27 11:15:00 | 5844.00 | 5865.99 | 5855.15 | SL hit (close<ema400) qty=1.00 sl=5855.15 alert=retest1 |

### Cycle 114 — SELL (started 2025-11-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 15:15:00 | 5820.00 | 5845.04 | 5847.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 09:15:00 | 5811.00 | 5838.23 | 5844.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 11:15:00 | 5835.00 | 5834.83 | 5841.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-28 12:00:00 | 5835.00 | 5834.83 | 5841.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 12:15:00 | 5838.50 | 5835.56 | 5841.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 10:15:00 | 5823.00 | 5840.71 | 5842.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 12:30:00 | 5823.50 | 5839.49 | 5841.30 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 14:30:00 | 5810.00 | 5833.05 | 5837.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 14:15:00 | 5877.50 | 5839.20 | 5836.96 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 115 — BUY (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 14:15:00 | 5877.50 | 5839.20 | 5836.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 15:15:00 | 5888.00 | 5848.96 | 5841.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 10:15:00 | 5843.00 | 5848.33 | 5842.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 10:15:00 | 5843.00 | 5848.33 | 5842.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 5843.00 | 5848.33 | 5842.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:30:00 | 5840.50 | 5848.33 | 5842.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 5856.00 | 5849.87 | 5843.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:45:00 | 5844.50 | 5849.87 | 5843.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 5816.50 | 5846.05 | 5843.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 5816.50 | 5846.05 | 5843.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 5840.00 | 5844.84 | 5843.62 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 09:15:00 | 5849.00 | 5844.84 | 5843.62 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-08 13:00:00 | 5844.00 | 5890.88 | 5885.18 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-08 14:15:00 | 5845.00 | 5874.60 | 5878.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 116 — SELL (started 2025-12-08 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 14:15:00 | 5845.00 | 5874.60 | 5878.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-09 09:15:00 | 5830.50 | 5864.73 | 5873.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 12:15:00 | 5860.00 | 5856.69 | 5866.81 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 13:00:00 | 5860.00 | 5856.69 | 5866.81 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 5869.00 | 5859.15 | 5867.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 14:00:00 | 5869.00 | 5859.15 | 5867.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 14:15:00 | 5882.00 | 5863.72 | 5868.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 15:00:00 | 5882.00 | 5863.72 | 5868.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 15:15:00 | 5887.00 | 5868.38 | 5870.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-10 09:15:00 | 5903.00 | 5868.38 | 5870.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 5955.00 | 5885.70 | 5877.79 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 11:15:00 | 5826.00 | 5873.65 | 5873.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 12:15:00 | 5820.50 | 5863.02 | 5868.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-11 11:15:00 | 5891.00 | 5854.44 | 5859.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-11 11:15:00 | 5891.00 | 5854.44 | 5859.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 11:15:00 | 5891.00 | 5854.44 | 5859.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-11 12:00:00 | 5891.00 | 5854.44 | 5859.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-11 12:15:00 | 5866.50 | 5856.85 | 5860.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-11 14:00:00 | 5844.50 | 5854.38 | 5858.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 09:45:00 | 5857.00 | 5847.79 | 5854.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 13:15:00 | 5900.00 | 5863.80 | 5859.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 119 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 5900.00 | 5863.80 | 5859.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 5914.00 | 5873.84 | 5864.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 15:15:00 | 6050.00 | 6056.06 | 6009.07 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 09:15:00 | 6085.50 | 6056.06 | 6009.07 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 10:00:00 | 6073.00 | 6083.21 | 6052.88 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 6048.50 | 6076.26 | 6052.48 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-18 10:15:00 | 6048.50 | 6076.26 | 6052.48 | SL hit (close<ema400) qty=1.00 sl=6052.48 alert=retest1 |

### Cycle 120 — SELL (started 2025-12-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 13:15:00 | 6042.00 | 6069.00 | 6071.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 10:15:00 | 6027.50 | 6053.92 | 6062.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 09:15:00 | 6045.00 | 6038.83 | 6050.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 09:15:00 | 6045.00 | 6038.83 | 6050.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 6045.00 | 6038.83 | 6050.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 11:45:00 | 6000.50 | 6025.94 | 6034.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-30 13:00:00 | 5983.50 | 6017.45 | 6029.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-31 12:15:00 | 6056.50 | 6034.59 | 6031.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — BUY (started 2025-12-31 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 12:15:00 | 6056.50 | 6034.59 | 6031.87 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 10:15:00 | 6025.00 | 6030.21 | 6030.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 11:15:00 | 6014.50 | 6027.06 | 6029.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-01 12:15:00 | 6028.00 | 6027.25 | 6029.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-01 12:15:00 | 6028.00 | 6027.25 | 6029.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 12:15:00 | 6028.00 | 6027.25 | 6029.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 13:00:00 | 6028.00 | 6027.25 | 6029.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2026-01-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-01 13:15:00 | 6042.00 | 6030.20 | 6030.20 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-01-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 14:15:00 | 6014.00 | 6026.96 | 6028.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 15:15:00 | 5999.00 | 6021.37 | 6026.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 10:15:00 | 6029.00 | 6022.68 | 6025.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 10:15:00 | 6029.00 | 6022.68 | 6025.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 10:15:00 | 6029.00 | 6022.68 | 6025.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:15:00 | 6004.00 | 6022.68 | 6025.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 5989.00 | 6015.94 | 6022.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-02 12:15:00 | 5953.00 | 6015.94 | 6022.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-05 09:15:00 | 5951.50 | 5993.68 | 6007.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-05 12:15:00 | 6047.50 | 6009.04 | 6010.92 | SL hit (close>static) qty=1.00 sl=6031.00 alert=retest2 |

### Cycle 125 — BUY (started 2026-01-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 13:15:00 | 6052.50 | 6017.73 | 6014.70 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 5958.00 | 6009.97 | 6012.86 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2026-01-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 12:15:00 | 6049.00 | 6020.66 | 6017.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-06 13:15:00 | 6091.00 | 6034.73 | 6024.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-08 09:15:00 | 6033.00 | 6142.90 | 6112.44 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 6033.00 | 6142.90 | 6112.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 6033.00 | 6142.90 | 6112.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 6033.00 | 6142.90 | 6112.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 6002.00 | 6114.72 | 6102.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:30:00 | 5964.50 | 6114.72 | 6102.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 128 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 6010.00 | 6093.78 | 6094.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 11:15:00 | 5986.50 | 6040.94 | 6063.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 11:15:00 | 5950.00 | 5940.58 | 5973.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-13 11:45:00 | 5960.50 | 5940.58 | 5973.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 10:15:00 | 5906.00 | 5913.36 | 5929.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 11:45:00 | 5890.00 | 5906.09 | 5925.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-16 13:00:00 | 5895.00 | 5903.87 | 5922.31 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 5890.00 | 5900.86 | 5916.05 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 12:45:00 | 5896.50 | 5909.06 | 5915.64 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 13:15:00 | 5949.00 | 5917.05 | 5918.67 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-01-19 13:15:00 | 5949.00 | 5917.05 | 5918.67 | SL hit (close>static) qty=1.00 sl=5944.00 alert=retest2 |

### Cycle 129 — BUY (started 2026-01-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-19 14:15:00 | 5939.50 | 5921.54 | 5920.56 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-20 10:15:00 | 5980.00 | 5942.23 | 5930.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-20 12:15:00 | 5902.50 | 5938.49 | 5931.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-20 12:15:00 | 5902.50 | 5938.49 | 5931.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 12:15:00 | 5902.50 | 5938.49 | 5931.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 13:00:00 | 5902.50 | 5938.49 | 5931.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 13:15:00 | 5910.00 | 5932.79 | 5929.50 | EMA400 retest candle locked (from upside) |

### Cycle 130 — SELL (started 2026-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 14:15:00 | 5879.50 | 5922.13 | 5924.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-21 09:15:00 | 5843.00 | 5901.96 | 5915.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 09:15:00 | 5896.00 | 5851.94 | 5876.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 09:15:00 | 5896.00 | 5851.94 | 5876.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 5896.00 | 5851.94 | 5876.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:45:00 | 5915.50 | 5851.94 | 5876.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 5907.00 | 5862.95 | 5879.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:45:00 | 5896.50 | 5862.95 | 5879.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 5921.50 | 5874.66 | 5883.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 12:15:00 | 5964.00 | 5874.66 | 5883.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 131 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 5960.00 | 5899.94 | 5893.89 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-01-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 13:15:00 | 5875.50 | 5892.54 | 5894.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 14:15:00 | 5825.00 | 5879.03 | 5888.44 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 5892.50 | 5877.08 | 5885.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 5892.50 | 5877.08 | 5885.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 5892.50 | 5877.08 | 5885.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 10:00:00 | 5892.50 | 5877.08 | 5885.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 5876.00 | 5876.86 | 5884.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-28 10:00:00 | 5780.00 | 5860.49 | 5874.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 10:15:00 | 5849.00 | 5753.36 | 5775.94 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-30 11:15:00 | 5848.00 | 5775.19 | 5783.81 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 11:15:00 | 5859.50 | 5792.05 | 5790.69 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — BUY (started 2026-01-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 11:15:00 | 5859.50 | 5792.05 | 5790.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 5905.00 | 5843.81 | 5819.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 5826.50 | 5840.35 | 5820.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 5826.50 | 5840.35 | 5820.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 5826.50 | 5840.35 | 5820.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 5826.50 | 5840.35 | 5820.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 5780.50 | 5828.38 | 5816.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:00:00 | 5780.50 | 5828.38 | 5816.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 5809.50 | 5824.60 | 5815.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 5804.50 | 5824.60 | 5815.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 5710.00 | 5792.87 | 5802.39 | EMA200 below EMA400 |

### Cycle 135 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 5891.00 | 5817.72 | 5808.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-02 15:15:00 | 5898.00 | 5833.77 | 5816.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 13:15:00 | 5879.00 | 5890.93 | 5867.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-04 13:15:00 | 5879.00 | 5890.93 | 5867.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 13:15:00 | 5879.00 | 5890.93 | 5867.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-04 13:30:00 | 5869.00 | 5890.93 | 5867.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 15:15:00 | 5877.00 | 5886.31 | 5869.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 09:15:00 | 5931.00 | 5886.31 | 5869.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 5885.50 | 5886.15 | 5870.51 | EMA400 retest candle locked (from upside) |

### Cycle 136 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 5823.50 | 5870.17 | 5870.58 | EMA200 below EMA400 |

### Cycle 137 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 5900.00 | 5873.98 | 5871.40 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-02-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-09 10:15:00 | 5812.50 | 5862.30 | 5867.87 | EMA200 below EMA400 |

### Cycle 139 — BUY (started 2026-02-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-10 12:15:00 | 5886.50 | 5859.58 | 5859.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-11 09:15:00 | 6057.50 | 5903.87 | 5880.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 6025.00 | 6071.13 | 6027.20 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 6025.00 | 6071.13 | 6027.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 6025.00 | 6071.13 | 6027.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:00:00 | 6025.00 | 6071.13 | 6027.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 6025.00 | 6061.90 | 6027.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 11:15:00 | 6036.00 | 6061.90 | 6027.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 6019.00 | 6053.32 | 6026.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:00:00 | 6019.00 | 6053.32 | 6026.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 6002.00 | 6043.06 | 6024.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 13:00:00 | 6002.00 | 6043.06 | 6024.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 13:15:00 | 6000.00 | 6034.45 | 6021.88 | EMA400 retest candle locked (from upside) |

### Cycle 140 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 5979.50 | 6013.95 | 6014.15 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 6042.50 | 6016.71 | 6015.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 11:15:00 | 6071.00 | 6027.56 | 6020.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 10:15:00 | 6138.50 | 6158.87 | 6131.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 10:30:00 | 6142.50 | 6158.87 | 6131.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 6109.00 | 6148.89 | 6129.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 6109.00 | 6148.89 | 6129.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 6115.00 | 6142.11 | 6128.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:00:00 | 6115.00 | 6142.11 | 6128.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 142 — SELL (started 2026-02-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 09:15:00 | 6096.00 | 6117.65 | 6119.52 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-23 10:15:00 | 6057.50 | 6092.15 | 6104.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-23 12:15:00 | 6093.00 | 6090.85 | 6101.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-23 12:15:00 | 6093.00 | 6090.85 | 6101.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 6093.00 | 6090.85 | 6101.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:45:00 | 6091.00 | 6090.85 | 6101.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 13:15:00 | 6114.00 | 6095.48 | 6102.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 13:45:00 | 6112.00 | 6095.48 | 6102.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 14:15:00 | 6125.50 | 6101.49 | 6104.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 15:00:00 | 6125.50 | 6101.49 | 6104.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 15:15:00 | 6106.00 | 6102.39 | 6104.88 | EMA400 retest candle locked (from downside) |

### Cycle 143 — BUY (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-24 09:15:00 | 6135.00 | 6108.91 | 6107.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-24 11:15:00 | 6145.50 | 6119.44 | 6112.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-25 12:15:00 | 6162.00 | 6164.47 | 6144.05 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-25 12:15:00 | 6162.00 | 6164.47 | 6144.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 12:15:00 | 6162.00 | 6164.47 | 6144.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 12:45:00 | 6154.00 | 6164.47 | 6144.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 14:15:00 | 6150.00 | 6162.14 | 6146.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-25 14:45:00 | 6145.50 | 6162.14 | 6146.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 15:15:00 | 6145.00 | 6158.71 | 6146.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:15:00 | 6111.50 | 6158.71 | 6146.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 6172.00 | 6161.37 | 6148.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-26 10:15:00 | 6180.50 | 6161.37 | 6148.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-26 11:15:00 | 6099.50 | 6141.90 | 6141.65 | SL hit (close<static) qty=1.00 sl=6101.50 alert=retest2 |

### Cycle 144 — SELL (started 2026-02-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-26 12:15:00 | 6099.50 | 6133.42 | 6137.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 09:15:00 | 6048.50 | 6112.02 | 6125.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 11:15:00 | 5924.50 | 5906.61 | 5947.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 12:00:00 | 5924.50 | 5906.61 | 5947.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 12:15:00 | 5948.00 | 5914.89 | 5947.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:00:00 | 5948.00 | 5914.89 | 5947.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 5940.00 | 5919.91 | 5946.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:30:00 | 5963.50 | 5919.91 | 5946.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 5962.50 | 5928.43 | 5947.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 5993.50 | 5928.43 | 5947.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 5954.00 | 5933.54 | 5948.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 5977.50 | 5933.54 | 5948.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 5991.00 | 5945.03 | 5952.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:00:00 | 5991.00 | 5945.03 | 5952.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 5985.00 | 5953.03 | 5955.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:30:00 | 5986.50 | 5953.03 | 5955.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — BUY (started 2026-03-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 11:15:00 | 5988.00 | 5960.02 | 5958.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-06 12:15:00 | 5995.50 | 5967.12 | 5961.70 | Break + close above crossover candle high |

### Cycle 146 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 5828.50 | 5948.41 | 5956.08 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 5973.00 | 5938.26 | 5937.53 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-03-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 12:15:00 | 5921.50 | 5941.34 | 5942.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 14:15:00 | 5908.00 | 5930.78 | 5937.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 10:15:00 | 5858.50 | 5825.78 | 5861.58 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 10:15:00 | 5858.50 | 5825.78 | 5861.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 5858.50 | 5825.78 | 5861.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:45:00 | 5868.00 | 5825.78 | 5861.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 5806.50 | 5821.93 | 5856.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:30:00 | 5858.00 | 5821.93 | 5856.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 5812.50 | 5822.06 | 5848.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 10:30:00 | 5790.00 | 5813.68 | 5837.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 12:15:00 | 5791.00 | 5810.44 | 5833.77 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 5852.00 | 5830.33 | 5831.33 | SL hit (close>static) qty=1.00 sl=5850.00 alert=retest2 |

### Cycle 149 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 5858.00 | 5835.87 | 5833.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 09:15:00 | 5874.50 | 5846.82 | 5839.29 | Break + close above crossover candle high |

### Cycle 150 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 5707.00 | 5842.24 | 5845.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 11:15:00 | 5694.50 | 5790.73 | 5820.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 5576.00 | 5513.29 | 5576.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 12:15:00 | 5576.00 | 5513.29 | 5576.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 5576.00 | 5513.29 | 5576.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 5576.00 | 5513.29 | 5576.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 5562.50 | 5523.13 | 5575.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:30:00 | 5584.50 | 5523.13 | 5575.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 5613.50 | 5540.62 | 5569.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 09:30:00 | 5610.00 | 5540.62 | 5569.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 5632.50 | 5559.00 | 5575.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 11:00:00 | 5632.50 | 5559.00 | 5575.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 151 — BUY (started 2026-03-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 12:15:00 | 5722.00 | 5611.36 | 5597.76 | EMA200 above EMA400 |

### Cycle 152 — SELL (started 2026-03-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 12:15:00 | 5560.50 | 5604.87 | 5605.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 5534.00 | 5590.69 | 5598.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 5490.00 | 5471.46 | 5513.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 5490.00 | 5471.46 | 5513.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 5490.00 | 5471.46 | 5513.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:30:00 | 5490.00 | 5471.46 | 5513.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 15:15:00 | 5496.50 | 5475.57 | 5496.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-02 09:15:00 | 5412.00 | 5475.57 | 5496.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 5431.00 | 5438.31 | 5460.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 11:00:00 | 5459.50 | 5444.58 | 5459.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 11:45:00 | 5455.00 | 5443.16 | 5457.55 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 12:15:00 | 5489.50 | 5452.43 | 5460.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-06 13:00:00 | 5489.50 | 5452.43 | 5460.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 13:15:00 | 5506.50 | 5463.24 | 5464.64 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-06 13:15:00 | 5506.50 | 5463.24 | 5464.64 | SL hit (close>static) qty=1.00 sl=5499.00 alert=retest2 |

### Cycle 153 — BUY (started 2026-04-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 14:15:00 | 5545.00 | 5479.60 | 5471.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 09:15:00 | 5624.50 | 5532.50 | 5505.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 11:15:00 | 5569.00 | 5586.65 | 5558.93 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 12:00:00 | 5569.00 | 5586.65 | 5558.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 12:15:00 | 5453.00 | 5559.92 | 5549.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 12:45:00 | 5460.00 | 5559.92 | 5549.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 13:15:00 | 5479.00 | 5543.74 | 5542.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 13:30:00 | 5464.50 | 5543.74 | 5542.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 154 — SELL (started 2026-04-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 14:15:00 | 5476.00 | 5530.19 | 5536.82 | EMA200 below EMA400 |

### Cycle 155 — BUY (started 2026-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 15:15:00 | 5565.50 | 5530.10 | 5528.83 | EMA200 above EMA400 |

### Cycle 156 — SELL (started 2026-04-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 09:15:00 | 5509.00 | 5525.88 | 5527.03 | EMA200 below EMA400 |

### Cycle 157 — BUY (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 10:15:00 | 5546.00 | 5529.90 | 5528.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 11:15:00 | 5564.00 | 5536.72 | 5531.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 09:15:00 | 5590.50 | 5623.58 | 5596.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 09:15:00 | 5590.50 | 5623.58 | 5596.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 5590.50 | 5623.58 | 5596.31 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 11:45:00 | 5723.00 | 5649.86 | 5619.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 13:30:00 | 5710.00 | 5697.21 | 5669.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:30:00 | 5745.00 | 5712.42 | 5683.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-23 09:30:00 | 5712.00 | 5745.61 | 5737.35 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 10:15:00 | 5673.50 | 5731.19 | 5731.55 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 158 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 5673.50 | 5731.19 | 5731.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 15:15:00 | 5656.50 | 5692.92 | 5710.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 11:15:00 | 5730.50 | 5696.65 | 5707.47 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 11:15:00 | 5730.50 | 5696.65 | 5707.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 5730.50 | 5696.65 | 5707.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:00:00 | 5730.50 | 5696.65 | 5707.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 12:15:00 | 5722.50 | 5701.82 | 5708.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 12:45:00 | 5745.00 | 5701.82 | 5708.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 14:15:00 | 5731.00 | 5705.53 | 5709.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-24 15:00:00 | 5731.00 | 5705.53 | 5709.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 15:15:00 | 5730.00 | 5710.42 | 5711.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:15:00 | 5732.00 | 5710.42 | 5711.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 5711.50 | 5709.61 | 5710.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 11:15:00 | 5698.50 | 5709.61 | 5710.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 159 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 5750.50 | 5717.79 | 5714.16 | EMA200 above EMA400 |

### Cycle 160 — SELL (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 10:15:00 | 5657.50 | 5704.80 | 5710.58 | EMA200 below EMA400 |

### Cycle 161 — BUY (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 14:15:00 | 5708.00 | 5699.40 | 5698.96 | EMA200 above EMA400 |

### Cycle 162 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 5676.50 | 5695.88 | 5697.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 5657.50 | 5688.20 | 5693.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 11:15:00 | 5692.00 | 5688.96 | 5693.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 11:15:00 | 5692.00 | 5688.96 | 5693.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 5692.00 | 5688.96 | 5693.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:30:00 | 5694.50 | 5688.96 | 5693.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 5703.50 | 5691.87 | 5694.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:30:00 | 5700.00 | 5691.87 | 5694.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 163 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 5730.00 | 5699.50 | 5697.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 5765.00 | 5720.90 | 5708.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 11:15:00 | 5738.00 | 5739.86 | 5720.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-04 12:00:00 | 5738.00 | 5739.86 | 5720.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 5787.50 | 5773.30 | 5746.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 09:30:00 | 5761.00 | 5773.30 | 5746.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 10:15:00 | 5791.50 | 5800.87 | 5778.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 10:30:00 | 5784.00 | 5800.87 | 5778.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 14:15:00 | 5780.00 | 5795.16 | 5782.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-06 15:00:00 | 5780.00 | 5795.16 | 5782.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-06 15:15:00 | 5790.00 | 5794.12 | 5783.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 09:15:00 | 5761.50 | 5794.12 | 5783.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 5762.50 | 5787.80 | 5781.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 10:00:00 | 5762.50 | 5787.80 | 5781.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 5807.00 | 5791.64 | 5783.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 11:45:00 | 5824.00 | 5795.71 | 5786.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 12:15:00 | 5825.00 | 5795.71 | 5786.13 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 13:00:00 | 5824.50 | 5801.47 | 5789.62 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 15:15:00 | 5885.50 | 5800.34 | 5791.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 5885.50 | 5817.37 | 5799.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:15:00 | 5537.00 | 5817.37 | 5799.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-05-08 09:15:00 | 5615.50 | 5777.00 | 5782.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 164 — SELL (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 09:15:00 | 5615.50 | 5777.00 | 5782.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 14:15:00 | 5523.00 | 5612.08 | 5688.41 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-06-10 10:30:00 | 5530.45 | 2024-06-12 12:15:00 | 5452.85 | STOP_HIT | 1.00 | -1.40% |
| BUY | retest2 | 2024-06-10 11:00:00 | 5534.50 | 2024-06-12 12:15:00 | 5452.85 | STOP_HIT | 1.00 | -1.48% |
| BUY | retest2 | 2024-06-11 09:15:00 | 5592.90 | 2024-06-12 12:15:00 | 5452.85 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2024-06-19 09:45:00 | 5380.65 | 2024-06-25 15:15:00 | 5362.00 | STOP_HIT | 1.00 | 0.35% |
| SELL | retest2 | 2024-06-19 13:45:00 | 5381.20 | 2024-06-25 15:15:00 | 5362.00 | STOP_HIT | 1.00 | 0.36% |
| SELL | retest2 | 2024-06-20 10:00:00 | 5381.80 | 2024-06-25 15:15:00 | 5362.00 | STOP_HIT | 1.00 | 0.37% |
| SELL | retest2 | 2024-06-20 13:45:00 | 5364.70 | 2024-06-25 15:15:00 | 5362.00 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2024-06-21 10:00:00 | 5368.40 | 2024-06-25 15:15:00 | 5362.00 | STOP_HIT | 1.00 | 0.12% |
| BUY | retest2 | 2024-06-27 15:00:00 | 5436.25 | 2024-07-02 13:15:00 | 5391.05 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-07-16 11:15:00 | 5843.70 | 2024-07-24 12:15:00 | 5839.00 | STOP_HIT | 1.00 | -0.08% |
| BUY | retest2 | 2024-07-16 11:45:00 | 5839.90 | 2024-07-24 12:15:00 | 5839.00 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2024-08-01 13:15:00 | 5744.95 | 2024-08-06 09:15:00 | 5824.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-08-02 14:45:00 | 5690.60 | 2024-08-06 11:15:00 | 5865.80 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2024-08-05 09:15:00 | 5695.60 | 2024-08-06 11:15:00 | 5865.80 | STOP_HIT | 1.00 | -2.99% |
| SELL | retest2 | 2024-08-05 10:45:00 | 5738.20 | 2024-08-06 11:15:00 | 5865.80 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2024-08-05 15:00:00 | 5708.65 | 2024-08-06 11:15:00 | 5865.80 | STOP_HIT | 1.00 | -2.75% |
| BUY | retest2 | 2024-08-27 09:15:00 | 5807.50 | 2024-08-27 09:15:00 | 5778.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2024-09-04 14:00:00 | 5916.85 | 2024-09-05 12:15:00 | 5846.70 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2024-09-16 10:30:00 | 6118.85 | 2024-09-25 09:15:00 | 6134.50 | STOP_HIT | 1.00 | 0.26% |
| BUY | retest2 | 2024-09-17 09:15:00 | 6202.10 | 2024-09-25 09:15:00 | 6134.50 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-09-18 13:45:00 | 6100.75 | 2024-09-25 09:15:00 | 6134.50 | STOP_HIT | 1.00 | 0.55% |
| BUY | retest2 | 2024-09-18 14:45:00 | 6116.55 | 2024-09-25 09:15:00 | 6134.50 | STOP_HIT | 1.00 | 0.29% |
| BUY | retest2 | 2024-09-19 09:15:00 | 6146.35 | 2024-09-25 09:15:00 | 6134.50 | STOP_HIT | 1.00 | -0.19% |
| BUY | retest2 | 2024-09-20 09:15:00 | 6149.20 | 2024-09-25 09:15:00 | 6134.50 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2024-09-30 14:15:00 | 6354.25 | 2024-10-04 10:15:00 | 6305.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2024-10-01 09:30:00 | 6358.00 | 2024-10-04 10:15:00 | 6305.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2024-10-01 11:45:00 | 6363.10 | 2024-10-04 10:15:00 | 6305.00 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2024-10-03 13:00:00 | 6366.70 | 2024-10-04 10:15:00 | 6305.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2024-10-09 09:15:00 | 6146.00 | 2024-10-15 13:15:00 | 6055.50 | STOP_HIT | 1.00 | 1.47% |
| SELL | retest1 | 2024-11-19 15:00:00 | 4884.05 | 2024-11-25 09:15:00 | 4923.45 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2024-11-26 11:15:00 | 4983.65 | 2024-11-28 12:15:00 | 4913.00 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2024-11-27 13:00:00 | 4984.10 | 2024-11-28 12:15:00 | 4913.00 | STOP_HIT | 1.00 | -1.43% |
| SELL | retest2 | 2024-12-04 11:30:00 | 4903.35 | 2024-12-11 11:15:00 | 4883.00 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2024-12-16 11:00:00 | 4793.50 | 2024-12-16 14:15:00 | 4849.95 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2024-12-17 10:15:00 | 4785.00 | 2024-12-26 11:15:00 | 4757.50 | STOP_HIT | 1.00 | 0.57% |
| SELL | retest2 | 2024-12-17 11:30:00 | 4792.80 | 2024-12-26 11:15:00 | 4757.50 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2024-12-17 12:00:00 | 4798.85 | 2024-12-26 11:15:00 | 4757.50 | STOP_HIT | 1.00 | 0.86% |
| SELL | retest2 | 2024-12-20 13:00:00 | 4737.90 | 2024-12-26 11:15:00 | 4757.50 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2024-12-24 13:45:00 | 4750.30 | 2024-12-26 11:15:00 | 4757.50 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2024-12-24 15:00:00 | 4741.15 | 2024-12-26 11:15:00 | 4757.50 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2024-12-30 10:30:00 | 4784.45 | 2025-01-02 10:15:00 | 4743.05 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2024-12-30 14:45:00 | 4781.75 | 2025-01-02 10:15:00 | 4743.05 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-01-01 10:00:00 | 4782.20 | 2025-01-02 10:15:00 | 4743.05 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-01-06 12:45:00 | 4832.40 | 2025-01-06 14:15:00 | 4784.00 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-01-07 09:15:00 | 4845.75 | 2025-01-08 09:15:00 | 4790.95 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-01-16 10:15:00 | 4805.15 | 2025-01-20 10:15:00 | 4881.60 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-01-16 12:45:00 | 4812.40 | 2025-01-20 10:15:00 | 4881.60 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-01-21 09:15:00 | 4900.00 | 2025-01-21 10:15:00 | 4856.80 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-01-21 09:45:00 | 4895.95 | 2025-01-21 10:15:00 | 4856.80 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-01-21 10:45:00 | 4889.40 | 2025-01-29 09:15:00 | 5016.00 | STOP_HIT | 1.00 | 2.59% |
| BUY | retest2 | 2025-01-21 11:30:00 | 4894.90 | 2025-01-29 09:15:00 | 5016.00 | STOP_HIT | 1.00 | 2.47% |
| BUY | retest2 | 2025-01-23 10:15:00 | 4991.45 | 2025-01-29 09:15:00 | 5016.00 | STOP_HIT | 1.00 | 0.49% |
| SELL | retest2 | 2025-02-07 10:15:00 | 4915.80 | 2025-02-13 10:15:00 | 4932.00 | STOP_HIT | 1.00 | -0.33% |
| SELL | retest2 | 2025-02-10 11:45:00 | 4960.00 | 2025-02-13 10:15:00 | 4932.00 | STOP_HIT | 1.00 | 0.56% |
| SELL | retest2 | 2025-02-14 11:15:00 | 4895.00 | 2025-02-14 14:15:00 | 4943.15 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-02-25 09:15:00 | 4768.55 | 2025-03-04 09:15:00 | 4530.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 13:15:00 | 4775.65 | 2025-03-04 09:15:00 | 4536.87 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 14:15:00 | 4772.25 | 2025-03-04 09:15:00 | 4533.64 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-28 09:15:00 | 4766.30 | 2025-03-04 09:15:00 | 4527.98 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 09:15:00 | 4768.55 | 2025-03-04 14:15:00 | 4574.90 | STOP_HIT | 0.50 | 4.06% |
| SELL | retest2 | 2025-02-27 13:15:00 | 4775.65 | 2025-03-04 14:15:00 | 4574.90 | STOP_HIT | 0.50 | 4.20% |
| SELL | retest2 | 2025-02-27 14:15:00 | 4772.25 | 2025-03-04 14:15:00 | 4574.90 | STOP_HIT | 0.50 | 4.14% |
| SELL | retest2 | 2025-02-28 09:15:00 | 4766.30 | 2025-03-04 14:15:00 | 4574.90 | STOP_HIT | 0.50 | 4.02% |
| BUY | retest2 | 2025-03-13 10:00:00 | 4829.00 | 2025-03-13 13:15:00 | 4727.15 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-03-13 11:00:00 | 4830.70 | 2025-03-13 13:15:00 | 4727.15 | STOP_HIT | 1.00 | -2.14% |
| BUY | retest2 | 2025-03-20 11:30:00 | 4793.50 | 2025-04-07 10:15:00 | 4966.40 | STOP_HIT | 1.00 | 3.61% |
| BUY | retest2 | 2025-03-24 10:00:00 | 4787.70 | 2025-04-07 10:15:00 | 4966.40 | STOP_HIT | 1.00 | 3.73% |
| BUY | retest2 | 2025-03-24 11:15:00 | 4799.00 | 2025-04-07 10:15:00 | 4966.40 | STOP_HIT | 1.00 | 3.49% |
| BUY | retest2 | 2025-03-25 09:30:00 | 4782.65 | 2025-04-07 10:15:00 | 4966.40 | STOP_HIT | 1.00 | 3.84% |
| BUY | retest2 | 2025-03-25 12:00:00 | 4834.40 | 2025-04-07 10:15:00 | 4966.40 | STOP_HIT | 1.00 | 2.73% |
| BUY | retest2 | 2025-03-26 10:45:00 | 4836.95 | 2025-04-07 10:15:00 | 4966.40 | STOP_HIT | 1.00 | 2.68% |
| BUY | retest2 | 2025-03-27 09:30:00 | 4837.05 | 2025-04-07 10:15:00 | 4966.40 | STOP_HIT | 1.00 | 2.67% |
| BUY | retest2 | 2025-04-22 09:15:00 | 5429.80 | 2025-04-25 09:15:00 | 5409.10 | STOP_HIT | 1.00 | -0.38% |
| BUY | retest2 | 2025-04-22 10:15:00 | 5396.30 | 2025-04-25 09:15:00 | 5409.10 | STOP_HIT | 1.00 | 0.24% |
| BUY | retest2 | 2025-04-22 11:45:00 | 5410.80 | 2025-04-25 09:15:00 | 5409.10 | STOP_HIT | 1.00 | -0.03% |
| SELL | retest2 | 2025-05-07 09:15:00 | 5374.50 | 2025-05-07 11:15:00 | 5375.50 | STOP_HIT | 1.00 | -0.02% |
| SELL | retest2 | 2025-05-07 10:45:00 | 5371.50 | 2025-05-07 11:15:00 | 5375.50 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-05-07 11:15:00 | 5367.50 | 2025-05-07 11:15:00 | 5375.50 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2025-05-12 09:30:00 | 5490.00 | 2025-05-14 13:15:00 | 5446.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-05-12 10:00:00 | 5493.50 | 2025-05-14 13:15:00 | 5446.00 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2025-05-13 12:15:00 | 5488.50 | 2025-05-14 13:15:00 | 5446.00 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2025-05-13 14:30:00 | 5491.50 | 2025-05-14 13:15:00 | 5446.00 | STOP_HIT | 1.00 | -0.83% |
| BUY | retest2 | 2025-05-19 14:15:00 | 5510.00 | 2025-05-20 12:15:00 | 5461.50 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-05-20 10:00:00 | 5510.50 | 2025-05-20 12:15:00 | 5461.50 | STOP_HIT | 1.00 | -0.89% |
| BUY | retest2 | 2025-05-20 12:00:00 | 5513.00 | 2025-05-20 12:15:00 | 5461.50 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-06-06 14:30:00 | 5581.00 | 2025-06-12 11:15:00 | 5629.00 | STOP_HIT | 1.00 | 0.86% |
| SELL | retest2 | 2025-06-17 09:15:00 | 5545.50 | 2025-06-18 09:15:00 | 5589.50 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-06-17 09:45:00 | 5551.00 | 2025-06-18 09:15:00 | 5589.50 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-06-23 09:15:00 | 5490.00 | 2025-06-23 15:15:00 | 5565.50 | STOP_HIT | 1.00 | -1.38% |
| BUY | retest2 | 2025-07-04 09:15:00 | 5838.00 | 2025-07-04 12:15:00 | 5769.50 | STOP_HIT | 1.00 | -1.17% |
| BUY | retest2 | 2025-07-09 10:30:00 | 5873.00 | 2025-07-11 12:15:00 | 5734.00 | STOP_HIT | 1.00 | -2.37% |
| BUY | retest2 | 2025-07-09 15:00:00 | 5896.50 | 2025-07-11 12:15:00 | 5734.00 | STOP_HIT | 1.00 | -2.76% |
| BUY | retest2 | 2025-07-10 11:00:00 | 5915.00 | 2025-07-11 12:15:00 | 5734.00 | STOP_HIT | 1.00 | -3.06% |
| BUY | retest2 | 2025-07-10 12:15:00 | 5890.00 | 2025-07-11 12:15:00 | 5734.00 | STOP_HIT | 1.00 | -2.65% |
| SELL | retest2 | 2025-07-16 10:45:00 | 5775.50 | 2025-07-17 09:15:00 | 5833.50 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2025-07-16 11:15:00 | 5772.00 | 2025-07-17 09:15:00 | 5833.50 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-07-16 14:15:00 | 5775.00 | 2025-07-17 09:15:00 | 5833.50 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2025-07-16 15:15:00 | 5770.00 | 2025-07-17 09:15:00 | 5833.50 | STOP_HIT | 1.00 | -1.10% |
| SELL | retest2 | 2025-07-22 15:00:00 | 5708.50 | 2025-07-30 09:15:00 | 5684.00 | STOP_HIT | 1.00 | 0.43% |
| SELL | retest2 | 2025-07-23 09:15:00 | 5702.50 | 2025-07-30 09:15:00 | 5684.00 | STOP_HIT | 1.00 | 0.32% |
| SELL | retest2 | 2025-07-24 09:15:00 | 5702.00 | 2025-07-30 09:15:00 | 5684.00 | STOP_HIT | 1.00 | 0.32% |
| BUY | retest2 | 2025-09-02 12:45:00 | 5911.50 | 2025-09-15 14:15:00 | 6206.50 | STOP_HIT | 1.00 | 4.99% |
| BUY | retest2 | 2025-09-02 14:30:00 | 5898.00 | 2025-09-15 14:15:00 | 6206.50 | STOP_HIT | 1.00 | 5.23% |
| BUY | retest2 | 2025-09-02 15:15:00 | 5895.00 | 2025-09-16 12:15:00 | 6214.00 | STOP_HIT | 1.00 | 5.41% |
| BUY | retest2 | 2025-09-03 09:45:00 | 5900.00 | 2025-09-16 12:15:00 | 6214.00 | STOP_HIT | 1.00 | 5.32% |
| BUY | retest2 | 2025-09-12 13:30:00 | 6254.00 | 2025-09-16 12:15:00 | 6214.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-09-12 15:15:00 | 6251.50 | 2025-09-16 12:15:00 | 6214.00 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-09-17 09:15:00 | 6106.00 | 2025-09-30 11:15:00 | 6009.00 | STOP_HIT | 1.00 | 1.59% |
| BUY | retest2 | 2025-10-06 13:45:00 | 5990.50 | 2025-10-07 11:15:00 | 5927.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-10-10 10:30:00 | 5868.00 | 2025-10-16 09:15:00 | 5904.50 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-10-13 11:00:00 | 5866.00 | 2025-10-16 09:15:00 | 5904.50 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-10-13 12:00:00 | 5866.50 | 2025-10-16 09:15:00 | 5904.50 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-10-13 12:30:00 | 5868.00 | 2025-10-16 09:15:00 | 5904.50 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-10-14 10:00:00 | 5841.50 | 2025-10-16 09:15:00 | 5904.50 | STOP_HIT | 1.00 | -1.08% |
| SELL | retest2 | 2025-10-31 11:15:00 | 5847.00 | 2025-11-04 13:15:00 | 5873.00 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-10-31 12:30:00 | 5850.00 | 2025-11-04 13:15:00 | 5873.00 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-10-31 15:00:00 | 5836.50 | 2025-11-04 13:15:00 | 5873.00 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-11-04 11:15:00 | 5837.00 | 2025-11-04 13:15:00 | 5873.00 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-11-19 09:15:00 | 5827.00 | 2025-11-19 11:15:00 | 5872.50 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-11-19 10:00:00 | 5831.50 | 2025-11-19 11:15:00 | 5872.50 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-11-19 10:30:00 | 5831.50 | 2025-11-19 11:15:00 | 5872.50 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest1 | 2025-11-27 09:15:00 | 5894.00 | 2025-11-27 11:15:00 | 5844.00 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-12-01 10:15:00 | 5823.00 | 2025-12-02 14:15:00 | 5877.50 | STOP_HIT | 1.00 | -0.94% |
| SELL | retest2 | 2025-12-01 12:30:00 | 5823.50 | 2025-12-02 14:15:00 | 5877.50 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-12-01 14:30:00 | 5810.00 | 2025-12-02 14:15:00 | 5877.50 | STOP_HIT | 1.00 | -1.16% |
| BUY | retest2 | 2025-12-04 09:15:00 | 5849.00 | 2025-12-08 14:15:00 | 5845.00 | STOP_HIT | 1.00 | -0.07% |
| BUY | retest2 | 2025-12-08 13:00:00 | 5844.00 | 2025-12-08 14:15:00 | 5845.00 | STOP_HIT | 1.00 | 0.02% |
| SELL | retest2 | 2025-12-11 14:00:00 | 5844.50 | 2025-12-12 13:15:00 | 5900.00 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2025-12-12 09:45:00 | 5857.00 | 2025-12-12 13:15:00 | 5900.00 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest1 | 2025-12-17 09:15:00 | 6085.50 | 2025-12-18 10:15:00 | 6048.50 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest1 | 2025-12-18 10:00:00 | 6073.00 | 2025-12-18 10:15:00 | 6048.50 | STOP_HIT | 1.00 | -0.40% |
| BUY | retest2 | 2025-12-18 15:00:00 | 6043.00 | 2025-12-23 13:15:00 | 6042.00 | STOP_HIT | 1.00 | -0.02% |
| BUY | retest2 | 2025-12-19 10:00:00 | 6050.50 | 2025-12-23 13:15:00 | 6042.00 | STOP_HIT | 1.00 | -0.14% |
| SELL | retest2 | 2025-12-30 11:45:00 | 6000.50 | 2025-12-31 12:15:00 | 6056.50 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-12-30 13:00:00 | 5983.50 | 2025-12-31 12:15:00 | 6056.50 | STOP_HIT | 1.00 | -1.22% |
| SELL | retest2 | 2026-01-02 12:15:00 | 5953.00 | 2026-01-05 12:15:00 | 6047.50 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2026-01-05 09:15:00 | 5951.50 | 2026-01-05 12:15:00 | 6047.50 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-01-16 11:45:00 | 5890.00 | 2026-01-19 13:15:00 | 5949.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-01-16 13:00:00 | 5895.00 | 2026-01-19 13:15:00 | 5949.00 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2026-01-19 09:15:00 | 5890.00 | 2026-01-19 13:15:00 | 5949.00 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-01-19 12:45:00 | 5896.50 | 2026-01-19 13:15:00 | 5949.00 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2026-01-28 10:00:00 | 5780.00 | 2026-01-30 11:15:00 | 5859.50 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2026-01-30 10:15:00 | 5849.00 | 2026-01-30 11:15:00 | 5859.50 | STOP_HIT | 1.00 | -0.18% |
| SELL | retest2 | 2026-01-30 11:15:00 | 5848.00 | 2026-01-30 11:15:00 | 5859.50 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2026-02-26 10:15:00 | 6180.50 | 2026-02-26 11:15:00 | 6099.50 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-03-16 10:30:00 | 5790.00 | 2026-03-17 13:15:00 | 5852.00 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2026-03-16 12:15:00 | 5791.00 | 2026-03-17 13:15:00 | 5852.00 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2026-04-02 09:15:00 | 5412.00 | 2026-04-06 13:15:00 | 5506.50 | STOP_HIT | 1.00 | -1.75% |
| SELL | retest2 | 2026-04-06 09:15:00 | 5431.00 | 2026-04-06 13:15:00 | 5506.50 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2026-04-06 11:00:00 | 5459.50 | 2026-04-06 13:15:00 | 5506.50 | STOP_HIT | 1.00 | -0.86% |
| SELL | retest2 | 2026-04-06 11:45:00 | 5455.00 | 2026-04-06 13:15:00 | 5506.50 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2026-04-17 11:45:00 | 5723.00 | 2026-04-23 10:15:00 | 5673.50 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-04-20 13:30:00 | 5710.00 | 2026-04-23 10:15:00 | 5673.50 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2026-04-21 09:30:00 | 5745.00 | 2026-04-23 10:15:00 | 5673.50 | STOP_HIT | 1.00 | -1.24% |
| BUY | retest2 | 2026-04-23 09:30:00 | 5712.00 | 2026-04-23 10:15:00 | 5673.50 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2026-05-07 11:45:00 | 5824.00 | 2026-05-08 09:15:00 | 5615.50 | STOP_HIT | 1.00 | -3.58% |
| BUY | retest2 | 2026-05-07 12:15:00 | 5825.00 | 2026-05-08 09:15:00 | 5615.50 | STOP_HIT | 1.00 | -3.60% |
| BUY | retest2 | 2026-05-07 13:00:00 | 5824.50 | 2026-05-08 09:15:00 | 5615.50 | STOP_HIT | 1.00 | -3.59% |
| BUY | retest2 | 2026-05-07 15:15:00 | 5885.50 | 2026-05-08 09:15:00 | 5615.50 | STOP_HIT | 1.00 | -4.59% |
