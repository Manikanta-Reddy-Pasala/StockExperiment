# BRITANNIA (BRITANNIA)

## Backtest Summary

- **Window:** 2024-03-12 09:15:00 → 2024-09-17 15:15:00 (892 bars)
- **Last close:** 6111.95
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 27 |
| ALERT1 | 18 |
| ALERT2 | 18 |
| ALERT2_SKIP | 18 |
| ALERT3 | 18 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 0 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 0 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 0 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 0 / 0
- **Target hits / Stop hits / Partials:** 0 / 0 / 0
- **Avg / median % per leg:** 0.00% / 0.00%
- **Sum % (uncompounded):** 0.00%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |

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

### Cycle 4 — SELL (started 2024-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-15 11:15:00 | 5081.90 | 5103.68 | 5105.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-16 09:15:00 | 5046.40 | 5077.06 | 5090.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-16 11:15:00 | 5088.90 | 5073.98 | 5086.61 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-16 11:15:00 | 5088.90 | 5073.98 | 5086.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-16 11:15:00 | 5088.90 | 5073.98 | 5086.61 | EMA400 retest candle locked (from downside) |

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

### Cycle 11 — BUY (started 2024-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 12:15:00 | 5278.05 | 5205.82 | 5200.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-04 13:15:00 | 5369.90 | 5238.64 | 5216.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 09:15:00 | 5430.00 | 5451.98 | 5378.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 5430.00 | 5451.98 | 5378.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 5430.00 | 5451.98 | 5378.30 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2024-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 12:15:00 | 5452.85 | 5488.73 | 5490.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-12 14:15:00 | 5437.05 | 5470.82 | 5481.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 12:15:00 | 5403.00 | 5395.70 | 5410.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 13:15:00 | 5383.00 | 5393.16 | 5407.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 13:15:00 | 5383.00 | 5393.16 | 5407.69 | EMA400 retest candle locked (from downside) |

### Cycle 13 — BUY (started 2024-06-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 15:15:00 | 5362.00 | 5343.49 | 5342.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-26 10:15:00 | 5380.65 | 5354.33 | 5348.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-27 12:15:00 | 5403.15 | 5409.55 | 5387.69 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-27 13:15:00 | 5420.00 | 5411.64 | 5390.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-27 13:15:00 | 5420.00 | 5411.64 | 5390.62 | EMA400 retest candle locked (from upside) |

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

### Cycle 18 — SELL (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 12:15:00 | 5839.00 | 5884.76 | 5887.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-24 13:15:00 | 5834.00 | 5874.61 | 5882.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 10:15:00 | 5840.85 | 5837.16 | 5850.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 12:15:00 | 5870.00 | 5844.96 | 5851.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 12:15:00 | 5870.00 | 5844.96 | 5851.95 | EMA400 retest candle locked (from downside) |

### Cycle 19 — BUY (started 2024-07-26 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-26 15:15:00 | 5860.00 | 5856.03 | 5855.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-29 10:15:00 | 5883.90 | 5862.10 | 5858.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 11:15:00 | 5883.15 | 5887.61 | 5876.00 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 11:15:00 | 5883.15 | 5887.61 | 5876.00 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 11:15:00 | 5883.15 | 5887.61 | 5876.00 | EMA400 retest candle locked (from upside) |

### Cycle 20 — SELL (started 2024-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 15:15:00 | 5836.00 | 5868.38 | 5870.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-31 12:15:00 | 5821.80 | 5852.86 | 5861.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-01 11:15:00 | 5834.45 | 5823.20 | 5839.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-01 11:15:00 | 5834.45 | 5823.20 | 5839.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 11:15:00 | 5834.45 | 5823.20 | 5839.42 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2024-08-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-06 11:15:00 | 5865.80 | 5781.17 | 5769.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-06 12:15:00 | 5919.70 | 5808.88 | 5783.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-07 11:15:00 | 5847.45 | 5852.97 | 5820.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-07 13:15:00 | 5818.60 | 5845.49 | 5822.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 13:15:00 | 5818.60 | 5845.49 | 5822.89 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2024-08-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-08 12:15:00 | 5744.80 | 5803.14 | 5811.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-08 15:15:00 | 5730.20 | 5773.15 | 5794.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 09:15:00 | 5705.45 | 5687.26 | 5719.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-14 09:15:00 | 5597.15 | 5653.94 | 5686.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-14 09:15:00 | 5597.15 | 5653.94 | 5686.08 | EMA400 retest candle locked (from downside) |

### Cycle 23 — BUY (started 2024-08-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 13:15:00 | 5708.00 | 5681.95 | 5680.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 14:15:00 | 5731.75 | 5691.91 | 5685.58 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-19 14:15:00 | 5728.45 | 5729.16 | 5711.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-23 14:15:00 | 5791.95 | 5821.24 | 5810.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 14:15:00 | 5791.95 | 5821.24 | 5810.60 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 09:15:00 | 5778.00 | 5805.57 | 5808.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 10:15:00 | 5767.40 | 5797.94 | 5804.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-29 09:15:00 | 5730.35 | 5723.76 | 5748.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-29 11:15:00 | 5789.00 | 5739.25 | 5751.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-29 11:15:00 | 5789.00 | 5739.25 | 5751.78 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2024-08-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-29 13:15:00 | 5803.50 | 5765.49 | 5762.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-29 15:15:00 | 5842.85 | 5790.48 | 5774.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-30 10:15:00 | 5800.00 | 5803.05 | 5783.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-04 09:15:00 | 5875.10 | 5900.87 | 5881.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 09:15:00 | 5875.10 | 5900.87 | 5881.19 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2024-09-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-05 12:15:00 | 5846.70 | 5881.73 | 5884.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-06 15:15:00 | 5840.20 | 5856.48 | 5865.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-09 09:15:00 | 5887.55 | 5862.70 | 5867.90 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-09 09:15:00 | 5887.55 | 5862.70 | 5867.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-09 09:15:00 | 5887.55 | 5862.70 | 5867.90 | EMA400 retest candle locked (from downside) |

### Cycle 27 — BUY (started 2024-09-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-09 11:15:00 | 5920.10 | 5879.66 | 5875.05 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-09 14:15:00 | 5932.50 | 5901.61 | 5887.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-13 12:15:00 | 6068.70 | 6069.72 | 6037.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-16 09:15:00 | 6072.20 | 6096.63 | 6063.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-16 09:15:00 | 6072.20 | 6096.63 | 6063.05 | EMA400 retest candle locked (from upside) |

