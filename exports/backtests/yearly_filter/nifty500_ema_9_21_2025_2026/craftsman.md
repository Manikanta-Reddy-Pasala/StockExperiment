# Craftsman Automation Ltd. (CRAFTSMAN)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 8990.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 82 |
| ALERT1 | 51 |
| ALERT2 | 50 |
| ALERT2_SKIP | 25 |
| ALERT3 | 142 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 60 |
| PARTIAL | 3 |
| TARGET_HIT | 1 |
| STOP_HIT | 62 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 66 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 15 / 51
- **Target hits / Stop hits / Partials:** 1 / 62 / 3
- **Avg / median % per leg:** -0.57% / -0.88%
- **Sum % (uncompounded):** -37.61%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 35 | 7 | 20.0% | 1 | 34 | 0 | -0.21% | -7.4% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.88% | -2.6% |
| BUY @ 3rd Alert (retest2) | 32 | 7 | 21.9% | 1 | 31 | 0 | -0.15% | -4.7% |
| SELL (all) | 31 | 8 | 25.8% | 0 | 28 | 3 | -0.98% | -30.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 31 | 8 | 25.8% | 0 | 28 | 3 | -0.98% | -30.2% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.88% | -2.6% |
| retest2 (combined) | 63 | 15 | 23.8% | 1 | 59 | 3 | -0.56% | -35.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 5266.90 | 5301.14 | 5304.84 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-05-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 09:15:00 | 5430.10 | 5326.94 | 5316.23 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-21 14:15:00 | 5285.90 | 5310.47 | 5313.01 | EMA200 below EMA400 |

### Cycle 4 — BUY (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-22 09:15:00 | 5388.00 | 5328.94 | 5321.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-22 10:15:00 | 5579.90 | 5379.13 | 5344.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-23 12:15:00 | 5521.40 | 5526.98 | 5464.47 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-23 13:00:00 | 5521.40 | 5526.98 | 5464.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 15:15:00 | 5477.00 | 5514.59 | 5474.20 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-26 09:30:00 | 5518.60 | 5512.18 | 5476.78 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 13:15:00 | 5430.00 | 5482.25 | 5473.00 | SL hit (close<static) qty=1.00 sl=5455.60 alert=retest2 |

### Cycle 5 — SELL (started 2025-05-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 15:15:00 | 5609.90 | 5626.81 | 5628.31 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-06-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 12:15:00 | 5638.00 | 5630.34 | 5629.58 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-06-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 13:15:00 | 5600.00 | 5624.27 | 5626.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-03 09:15:00 | 5500.50 | 5589.65 | 5609.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 10:15:00 | 5571.50 | 5533.29 | 5560.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 10:15:00 | 5571.50 | 5533.29 | 5560.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 5571.50 | 5533.29 | 5560.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 5571.50 | 5533.29 | 5560.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 5551.50 | 5536.93 | 5559.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:30:00 | 5570.00 | 5536.93 | 5559.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 14:15:00 | 5517.00 | 5534.90 | 5553.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 14:30:00 | 5550.00 | 5534.90 | 5553.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 5483.00 | 5518.94 | 5542.60 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2025-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 12:15:00 | 5682.50 | 5570.33 | 5555.11 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2025-06-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 11:15:00 | 5533.50 | 5569.84 | 5572.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 12:15:00 | 5500.00 | 5555.87 | 5565.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 5302.50 | 5290.36 | 5345.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 15:00:00 | 5302.50 | 5290.36 | 5345.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 5332.50 | 5277.06 | 5299.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 09:45:00 | 5251.50 | 5285.36 | 5295.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:45:00 | 5245.00 | 5266.61 | 5283.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 15:00:00 | 5247.50 | 5258.69 | 5276.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-19 10:15:00 | 5237.00 | 5254.74 | 5271.41 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 09:15:00 | 5233.00 | 5216.98 | 5240.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 15:00:00 | 5163.50 | 5194.87 | 5219.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 10:15:00 | 5268.50 | 5213.92 | 5222.40 | SL hit (close>static) qty=1.00 sl=5249.00 alert=retest2 |

### Cycle 10 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 5278.00 | 5234.91 | 5230.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 12:15:00 | 5295.50 | 5255.85 | 5246.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-27 09:15:00 | 5429.00 | 5436.90 | 5382.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-27 10:00:00 | 5429.00 | 5436.90 | 5382.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 10:15:00 | 5414.00 | 5432.32 | 5385.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 10:30:00 | 5392.50 | 5432.32 | 5385.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 09:15:00 | 5833.00 | 5867.18 | 5839.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:00:00 | 5833.00 | 5867.18 | 5839.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 5790.00 | 5851.75 | 5834.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-08 11:00:00 | 5790.00 | 5851.75 | 5834.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 11:15:00 | 5867.00 | 5854.80 | 5837.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 09:15:00 | 5930.00 | 5845.59 | 5837.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-09 10:15:00 | 5985.00 | 5852.77 | 5841.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 13:15:00 | 5928.50 | 5931.26 | 5927.78 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-11 14:45:00 | 5937.00 | 5929.61 | 5927.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 5933.50 | 5931.89 | 5928.93 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-14 12:15:00 | 5910.00 | 5924.55 | 5926.07 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2025-07-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 12:15:00 | 5910.00 | 5924.55 | 5926.07 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-07-14 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 14:15:00 | 5942.00 | 5929.39 | 5928.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 6124.00 | 5971.53 | 5947.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 10:15:00 | 6361.00 | 6403.15 | 6300.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-18 10:30:00 | 6396.00 | 6403.15 | 6300.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 09:15:00 | 6453.00 | 6544.86 | 6499.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 10:00:00 | 6453.00 | 6544.86 | 6499.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 10:15:00 | 6400.50 | 6515.99 | 6490.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 11:00:00 | 6400.50 | 6515.99 | 6490.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 13 — SELL (started 2025-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-23 12:15:00 | 6400.00 | 6474.55 | 6474.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 13:15:00 | 6386.50 | 6456.94 | 6466.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-24 09:15:00 | 6440.00 | 6438.08 | 6454.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-24 09:15:00 | 6440.00 | 6438.08 | 6454.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 6440.00 | 6438.08 | 6454.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 10:30:00 | 6395.00 | 6416.17 | 6442.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 13:15:00 | 6075.25 | 6180.22 | 6240.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 14:15:00 | 6475.00 | 6239.17 | 6261.92 | SL hit (close>ema200) qty=0.50 sl=6239.17 alert=retest2 |

### Cycle 14 — BUY (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 15:15:00 | 6465.00 | 6284.34 | 6280.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 6815.00 | 6390.47 | 6328.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 11:15:00 | 6698.00 | 6698.92 | 6573.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 12:00:00 | 6698.00 | 6698.92 | 6573.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 6568.00 | 6662.07 | 6602.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 6568.00 | 6662.07 | 6602.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 6576.00 | 6644.86 | 6600.25 | EMA400 retest candle locked (from upside) |

### Cycle 15 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 6492.00 | 6581.00 | 6581.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 15:15:00 | 6425.00 | 6549.80 | 6567.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 12:15:00 | 6549.50 | 6534.28 | 6552.79 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-04 13:00:00 | 6549.50 | 6534.28 | 6552.79 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 6626.00 | 6552.63 | 6559.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-04 14:00:00 | 6626.00 | 6552.63 | 6559.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 6646.00 | 6571.30 | 6567.31 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 6486.00 | 6564.35 | 6565.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-05 10:15:00 | 6445.50 | 6540.58 | 6554.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 12:15:00 | 6374.00 | 6352.33 | 6399.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 12:15:00 | 6374.00 | 6352.33 | 6399.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 12:15:00 | 6374.00 | 6352.33 | 6399.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 12:45:00 | 6372.00 | 6352.33 | 6399.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 6403.50 | 6364.83 | 6397.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 6403.50 | 6364.83 | 6397.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 6423.50 | 6376.57 | 6399.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 6425.00 | 6376.57 | 6399.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 6445.50 | 6390.35 | 6403.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:45:00 | 6440.00 | 6390.35 | 6403.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 10:15:00 | 6439.50 | 6400.18 | 6406.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 11:15:00 | 6484.50 | 6400.18 | 6406.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 18 — BUY (started 2025-08-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-08 11:15:00 | 6517.50 | 6423.65 | 6417.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-11 09:15:00 | 6615.00 | 6492.85 | 6456.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 15:15:00 | 7033.00 | 7066.94 | 6970.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-18 09:15:00 | 6997.00 | 7066.94 | 6970.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 7014.00 | 7056.36 | 6974.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:45:00 | 6995.00 | 7056.36 | 6974.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 6917.00 | 7019.47 | 6971.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:00:00 | 6917.00 | 7019.47 | 6971.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 12:15:00 | 6850.50 | 6985.67 | 6960.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-18 12:30:00 | 6873.50 | 6985.67 | 6960.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — SELL (started 2025-08-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 14:15:00 | 6867.50 | 6934.17 | 6939.68 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-08-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-18 15:15:00 | 6990.00 | 6945.34 | 6944.25 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2025-08-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-19 09:15:00 | 6805.00 | 6917.27 | 6931.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-26 11:15:00 | 6786.00 | 6826.76 | 6847.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 12:15:00 | 6880.50 | 6837.51 | 6850.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 12:15:00 | 6880.50 | 6837.51 | 6850.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 6880.50 | 6837.51 | 6850.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:00:00 | 6880.50 | 6837.51 | 6850.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 6856.50 | 6841.31 | 6850.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 14:15:00 | 6900.00 | 6841.31 | 6850.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — BUY (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-26 14:15:00 | 7162.50 | 6905.55 | 6879.03 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2025-08-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-29 12:15:00 | 6877.00 | 6922.91 | 6923.90 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 7027.50 | 6930.05 | 6922.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 7180.00 | 6980.04 | 6946.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 09:15:00 | 6985.50 | 7010.17 | 6967.57 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 09:15:00 | 6985.50 | 7010.17 | 6967.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 6985.50 | 7010.17 | 6967.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:45:00 | 6972.00 | 7010.17 | 6967.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 11:15:00 | 6993.00 | 7002.55 | 6971.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 11:30:00 | 6977.00 | 7002.55 | 6971.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 6972.00 | 6996.44 | 6971.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:00:00 | 6972.00 | 6996.44 | 6971.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 6935.00 | 6984.15 | 6968.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:45:00 | 6931.00 | 6984.15 | 6968.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 14:15:00 | 6985.00 | 6984.32 | 6969.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 15:15:00 | 6948.00 | 6984.32 | 6969.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 15:15:00 | 6948.00 | 6977.06 | 6967.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:15:00 | 6867.50 | 6977.06 | 6967.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — SELL (started 2025-09-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-03 09:15:00 | 6869.00 | 6955.44 | 6958.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 14:15:00 | 6829.50 | 6883.29 | 6907.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 14:15:00 | 6896.50 | 6854.64 | 6875.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 14:15:00 | 6896.50 | 6854.64 | 6875.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 6896.50 | 6854.64 | 6875.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:45:00 | 6877.50 | 6854.64 | 6875.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 6859.50 | 6855.61 | 6873.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 09:15:00 | 6848.50 | 6855.61 | 6873.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 13:00:00 | 6842.00 | 6858.61 | 6870.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 14:15:00 | 6844.00 | 6857.39 | 6868.87 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 10:15:00 | 6802.50 | 6770.05 | 6766.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — BUY (started 2025-09-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-12 10:15:00 | 6802.50 | 6770.05 | 6766.20 | EMA200 above EMA400 |

### Cycle 27 — SELL (started 2025-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 13:15:00 | 6700.00 | 6757.43 | 6761.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 09:15:00 | 6571.00 | 6702.37 | 6734.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 09:15:00 | 6700.00 | 6615.47 | 6659.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 09:15:00 | 6700.00 | 6615.47 | 6659.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 09:15:00 | 6700.00 | 6615.47 | 6659.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-16 10:00:00 | 6700.00 | 6615.47 | 6659.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 10:15:00 | 6680.50 | 6628.47 | 6661.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-16 11:30:00 | 6666.00 | 6640.78 | 6664.19 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-16 13:15:00 | 6713.00 | 6661.50 | 6669.93 | SL hit (close>static) qty=1.00 sl=6712.50 alert=retest2 |

### Cycle 28 — BUY (started 2025-09-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 14:15:00 | 6752.50 | 6679.70 | 6677.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-17 09:15:00 | 6783.50 | 6711.63 | 6693.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-18 11:15:00 | 6766.50 | 6770.83 | 6744.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-18 12:00:00 | 6766.50 | 6770.83 | 6744.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 6799.50 | 6777.69 | 6757.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 10:45:00 | 6836.50 | 6782.36 | 6761.55 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 13:15:00 | 6734.50 | 6772.31 | 6762.07 | SL hit (close<static) qty=1.00 sl=6740.50 alert=retest2 |

### Cycle 29 — SELL (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 10:15:00 | 6746.50 | 6781.37 | 6781.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 12:15:00 | 6692.50 | 6756.74 | 6770.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 13:15:00 | 6798.50 | 6765.09 | 6772.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 13:15:00 | 6798.50 | 6765.09 | 6772.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 6798.50 | 6765.09 | 6772.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 14:00:00 | 6798.50 | 6765.09 | 6772.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 6829.00 | 6777.87 | 6777.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 15:00:00 | 6829.00 | 6777.87 | 6777.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — BUY (started 2025-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 15:15:00 | 6820.00 | 6786.30 | 6781.74 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 11:15:00 | 6757.50 | 6776.76 | 6778.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 14:15:00 | 6740.00 | 6767.64 | 6773.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-25 15:15:00 | 6778.00 | 6769.71 | 6773.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 15:15:00 | 6778.00 | 6769.71 | 6773.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 6778.00 | 6769.71 | 6773.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 6589.50 | 6769.71 | 6773.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 09:15:00 | 6623.00 | 6740.37 | 6760.13 | EMA400 retest candle locked (from downside) |

### Cycle 32 — BUY (started 2025-09-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-29 13:15:00 | 6815.00 | 6747.39 | 6739.52 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2025-09-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-29 15:15:00 | 6671.00 | 6728.21 | 6731.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-30 09:15:00 | 6626.50 | 6707.87 | 6722.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 14:15:00 | 6800.00 | 6691.67 | 6704.91 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-30 14:15:00 | 6800.00 | 6691.67 | 6704.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 6800.00 | 6691.67 | 6704.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 6800.00 | 6691.67 | 6704.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 6752.50 | 6703.83 | 6709.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 6712.50 | 6703.83 | 6709.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 6718.00 | 6696.17 | 6702.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:00:00 | 6718.00 | 6696.17 | 6702.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 13:15:00 | 6741.00 | 6705.13 | 6706.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 14:00:00 | 6741.00 | 6705.13 | 6706.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 6723.00 | 6708.28 | 6707.56 | EMA200 above EMA400 |

### Cycle 35 — SELL (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-03 10:15:00 | 6694.00 | 6707.30 | 6707.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-03 12:15:00 | 6683.00 | 6700.55 | 6704.16 | Break + close below crossover candle low |

### Cycle 36 — BUY (started 2025-10-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 13:15:00 | 6738.50 | 6708.14 | 6707.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 15:15:00 | 6798.50 | 6736.43 | 6720.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 15:15:00 | 6780.00 | 6785.96 | 6759.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 09:15:00 | 6775.00 | 6785.96 | 6759.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 09:15:00 | 6773.50 | 6783.47 | 6761.22 | EMA400 retest candle locked (from upside) |

### Cycle 37 — SELL (started 2025-10-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 13:15:00 | 6708.00 | 6746.61 | 6748.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 14:15:00 | 6692.50 | 6735.79 | 6743.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-10 14:15:00 | 6563.00 | 6552.60 | 6591.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-10 14:45:00 | 6551.00 | 6552.60 | 6591.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 6506.00 | 6487.09 | 6523.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 09:30:00 | 6574.50 | 6487.09 | 6523.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 6462.00 | 6466.53 | 6498.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 14:45:00 | 6592.00 | 6466.53 | 6498.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 6465.00 | 6459.55 | 6486.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 10:30:00 | 6500.00 | 6459.55 | 6486.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 6477.50 | 6463.14 | 6485.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:30:00 | 6491.00 | 6463.14 | 6485.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 6441.00 | 6458.71 | 6481.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 14:00:00 | 6425.00 | 6451.97 | 6476.55 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 15:15:00 | 6409.00 | 6448.67 | 6472.82 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-16 09:15:00 | 6790.00 | 6510.59 | 6496.38 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 38 — BUY (started 2025-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 09:15:00 | 6790.00 | 6510.59 | 6496.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 11:15:00 | 6800.50 | 6611.92 | 6547.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 10:15:00 | 6677.50 | 6711.09 | 6638.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 11:00:00 | 6677.50 | 6711.09 | 6638.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 6668.50 | 6705.99 | 6671.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:00:00 | 6668.50 | 6705.99 | 6671.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 6674.00 | 6699.59 | 6671.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:30:00 | 6692.50 | 6699.59 | 6671.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 6670.00 | 6691.98 | 6676.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 6723.00 | 6695.58 | 6679.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 6683.00 | 6693.77 | 6681.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 6683.00 | 6693.77 | 6681.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 6676.00 | 6690.22 | 6681.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:00:00 | 6676.00 | 6690.22 | 6681.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 6677.00 | 6687.57 | 6680.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 12:30:00 | 6705.00 | 6691.76 | 6683.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-23 14:15:00 | 6614.00 | 6677.69 | 6678.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 39 — SELL (started 2025-10-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 14:15:00 | 6614.00 | 6677.69 | 6678.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 11:15:00 | 6592.00 | 6637.52 | 6657.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 09:15:00 | 6504.00 | 6500.13 | 6546.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 09:45:00 | 6513.50 | 6500.13 | 6546.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 10:15:00 | 6549.00 | 6509.91 | 6546.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-28 11:00:00 | 6549.00 | 6509.91 | 6546.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-28 11:15:00 | 6487.50 | 6505.43 | 6541.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 12:15:00 | 6480.00 | 6505.43 | 6541.39 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 13:30:00 | 6468.00 | 6487.87 | 6526.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-28 14:00:00 | 6434.00 | 6487.87 | 6526.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-29 15:00:00 | 6463.00 | 6441.58 | 6473.40 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 15:15:00 | 6452.00 | 6443.66 | 6471.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:15:00 | 6590.00 | 6443.66 | 6471.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 6550.00 | 6464.93 | 6478.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-30 09:30:00 | 6540.50 | 6464.93 | 6478.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-30 10:15:00 | 6583.00 | 6488.54 | 6488.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — BUY (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-30 10:15:00 | 6583.00 | 6488.54 | 6488.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 11:15:00 | 6584.50 | 6507.73 | 6496.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-31 14:15:00 | 6666.50 | 6682.86 | 6619.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-31 15:00:00 | 6666.50 | 6682.86 | 6619.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 15:15:00 | 6815.00 | 6840.01 | 6787.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:15:00 | 6798.50 | 6840.01 | 6787.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 6794.00 | 6830.81 | 6787.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 09:45:00 | 6773.50 | 6830.81 | 6787.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 10:15:00 | 6789.50 | 6822.55 | 6788.05 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2025-11-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-07 09:15:00 | 6681.50 | 6768.42 | 6773.01 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2025-11-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 15:15:00 | 6799.50 | 6769.97 | 6769.34 | EMA200 above EMA400 |

### Cycle 43 — SELL (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 09:15:00 | 6657.50 | 6747.48 | 6759.17 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2025-11-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 15:15:00 | 6793.50 | 6762.29 | 6758.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 09:15:00 | 7112.00 | 6832.23 | 6791.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-12 09:15:00 | 7031.50 | 7084.90 | 6973.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-12 09:30:00 | 7034.50 | 7084.90 | 6973.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 6927.50 | 7033.84 | 6999.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 6927.50 | 7033.84 | 6999.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 6880.50 | 7003.17 | 6989.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 6880.50 | 7003.17 | 6989.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 45 — SELL (started 2025-11-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 11:15:00 | 6867.50 | 6976.04 | 6978.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 15:15:00 | 6812.00 | 6898.99 | 6936.85 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 12:15:00 | 6800.00 | 6789.17 | 6833.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-17 13:00:00 | 6800.00 | 6789.17 | 6833.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 6691.00 | 6771.64 | 6812.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 10:30:00 | 6678.50 | 6754.31 | 6800.49 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 11:45:00 | 6678.00 | 6743.25 | 6791.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-21 11:00:00 | 6640.00 | 6715.58 | 6736.70 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-24 09:30:00 | 6588.50 | 6680.05 | 6710.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 6684.50 | 6658.04 | 6687.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 13:45:00 | 6684.00 | 6658.04 | 6687.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 6662.00 | 6658.83 | 6685.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:15:00 | 6759.00 | 6658.83 | 6685.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 6759.00 | 6678.87 | 6692.19 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-25 09:15:00 | 6942.50 | 6731.59 | 6714.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 46 — BUY (started 2025-11-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 09:15:00 | 6942.50 | 6731.59 | 6714.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 10:15:00 | 7065.00 | 6798.28 | 6746.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 09:15:00 | 7025.50 | 7058.16 | 6979.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-27 09:30:00 | 7020.00 | 7058.16 | 6979.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 6973.50 | 7041.23 | 6978.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:00:00 | 6973.50 | 7041.23 | 6978.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 6985.00 | 7029.98 | 6979.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-27 12:15:00 | 7042.00 | 7029.98 | 6979.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 11:45:00 | 7013.50 | 7036.50 | 7009.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-28 13:00:00 | 7005.00 | 7030.20 | 7008.71 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 12:30:00 | 6995.50 | 7018.57 | 7013.24 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 7048.00 | 7022.77 | 7015.97 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-02 10:15:00 | 6980.00 | 7011.32 | 7012.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 6980.00 | 7011.32 | 7012.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-02 11:15:00 | 6953.00 | 6999.66 | 7007.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 14:15:00 | 7069.50 | 6961.07 | 6970.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 14:15:00 | 7069.50 | 6961.07 | 6970.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 14:15:00 | 7069.50 | 6961.07 | 6970.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 15:00:00 | 7069.50 | 6961.07 | 6970.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 15:15:00 | 7011.00 | 6971.06 | 6974.51 | EMA400 retest candle locked (from downside) |

### Cycle 48 — BUY (started 2025-12-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 09:15:00 | 7063.50 | 6989.55 | 6982.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 09:15:00 | 7098.50 | 7047.73 | 7021.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 11:15:00 | 7085.50 | 7090.50 | 7064.27 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 12:00:00 | 7085.50 | 7090.50 | 7064.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 12:15:00 | 6970.00 | 7066.40 | 7055.70 | EMA400 retest candle locked (from upside) |

### Cycle 49 — SELL (started 2025-12-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 13:15:00 | 6968.50 | 7046.82 | 7047.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 14:15:00 | 6914.50 | 7020.36 | 7035.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 6941.50 | 6932.15 | 6981.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 6941.50 | 6932.15 | 6981.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 12:15:00 | 7024.50 | 6950.62 | 6985.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:00:00 | 7024.50 | 6950.62 | 6985.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 7029.00 | 6966.30 | 6989.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-09 13:30:00 | 7007.50 | 6966.30 | 6989.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 7059.00 | 7012.34 | 7007.50 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2025-12-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 12:15:00 | 6975.50 | 7003.05 | 7004.43 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2025-12-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 13:15:00 | 7044.50 | 7011.34 | 7008.07 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2025-12-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 15:15:00 | 6935.00 | 6992.74 | 6999.97 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2025-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 11:15:00 | 7086.50 | 7013.49 | 7007.84 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2025-12-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 13:15:00 | 7042.50 | 7050.91 | 7051.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-17 09:15:00 | 6982.00 | 7035.59 | 7043.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-17 15:15:00 | 7047.00 | 7012.96 | 7025.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-17 15:15:00 | 7047.00 | 7012.96 | 7025.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 15:15:00 | 7047.00 | 7012.96 | 7025.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 6922.50 | 7012.96 | 7025.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-18 15:15:00 | 7076.50 | 7005.57 | 7007.82 | SL hit (close>static) qty=1.00 sl=7065.00 alert=retest2 |

### Cycle 56 — BUY (started 2025-12-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 09:15:00 | 7069.00 | 7018.26 | 7013.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 11:15:00 | 7103.00 | 7068.72 | 7046.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-22 14:15:00 | 7051.50 | 7077.20 | 7056.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-22 14:15:00 | 7051.50 | 7077.20 | 7056.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 14:15:00 | 7051.50 | 7077.20 | 7056.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-22 15:00:00 | 7051.50 | 7077.20 | 7056.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-22 15:15:00 | 7060.00 | 7073.76 | 7057.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:15:00 | 7076.50 | 7073.76 | 7057.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:45:00 | 7066.50 | 7073.11 | 7058.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 11:15:00 | 7098.00 | 7065.89 | 7056.44 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 09:15:00 | 7078.50 | 7079.46 | 7069.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 7085.00 | 7080.57 | 7070.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-24 13:45:00 | 7102.50 | 7098.18 | 7082.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-24 15:15:00 | 7002.50 | 7071.66 | 7072.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — SELL (started 2025-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 15:15:00 | 7002.50 | 7071.66 | 7072.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-26 10:15:00 | 6959.50 | 7040.00 | 7057.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-26 12:15:00 | 7068.00 | 7032.80 | 7050.30 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-26 12:15:00 | 7068.00 | 7032.80 | 7050.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 12:15:00 | 7068.00 | 7032.80 | 7050.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-26 13:00:00 | 7068.00 | 7032.80 | 7050.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2025-12-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 13:15:00 | 7279.00 | 7082.04 | 7071.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 14:15:00 | 7392.00 | 7144.03 | 7100.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 10:15:00 | 7158.00 | 7189.10 | 7136.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 10:15:00 | 7158.00 | 7189.10 | 7136.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 7158.00 | 7189.10 | 7136.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:00:00 | 7158.00 | 7189.10 | 7136.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 11:15:00 | 7139.50 | 7179.18 | 7136.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 12:15:00 | 7122.00 | 7179.18 | 7136.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 12:15:00 | 7089.50 | 7161.25 | 7132.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 12:45:00 | 7099.50 | 7161.25 | 7132.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 7096.00 | 7148.20 | 7129.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:15:00 | 7080.50 | 7148.20 | 7129.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 7071.00 | 7132.76 | 7123.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:45:00 | 7097.00 | 7132.76 | 7123.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2025-12-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-29 15:15:00 | 7004.50 | 7107.11 | 7112.89 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 10:15:00 | 7138.00 | 7119.11 | 7117.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 10:15:00 | 7545.00 | 7236.74 | 7177.39 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-01 09:15:00 | 7466.00 | 7524.24 | 7379.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-01 10:00:00 | 7466.00 | 7524.24 | 7379.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 09:15:00 | 7991.50 | 7998.54 | 7931.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 09:30:00 | 7977.50 | 7998.54 | 7931.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 7809.00 | 7960.63 | 7919.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 7809.00 | 7960.63 | 7919.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 7839.00 | 7936.31 | 7912.58 | EMA400 retest candle locked (from upside) |

### Cycle 61 — SELL (started 2026-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 14:15:00 | 7834.00 | 7890.19 | 7895.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 15:15:00 | 7788.50 | 7869.85 | 7885.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-08 09:15:00 | 7881.50 | 7872.18 | 7885.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-08 09:15:00 | 7881.50 | 7872.18 | 7885.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 09:15:00 | 7881.50 | 7872.18 | 7885.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-08 10:00:00 | 7881.50 | 7872.18 | 7885.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-08 10:15:00 | 7735.50 | 7844.84 | 7871.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:15:00 | 7703.00 | 7844.84 | 7871.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 12:45:00 | 7721.00 | 7800.50 | 7845.33 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 13:15:00 | 7709.00 | 7800.50 | 7845.33 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 14:00:00 | 7716.00 | 7783.60 | 7833.58 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 09:15:00 | 7770.00 | 7745.19 | 7800.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:00:00 | 7770.00 | 7745.19 | 7800.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 7745.00 | 7745.15 | 7795.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 10:30:00 | 7804.00 | 7745.15 | 7795.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-12 11:15:00 | 7334.95 | 7556.93 | 7660.95 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-12 12:15:00 | 7566.00 | 7558.74 | 7652.32 | SL hit (close>ema200) qty=0.50 sl=7558.74 alert=retest2 |

### Cycle 62 — BUY (started 2026-01-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 09:15:00 | 7754.00 | 7616.38 | 7613.91 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 12:15:00 | 7814.50 | 7695.21 | 7654.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 7756.50 | 7789.63 | 7739.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 14:00:00 | 7756.50 | 7789.63 | 7739.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-16 15:15:00 | 7739.50 | 7776.46 | 7742.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:15:00 | 7674.50 | 7776.46 | 7742.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 7725.00 | 7766.17 | 7740.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 7684.00 | 7766.17 | 7740.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 7674.00 | 7747.74 | 7734.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:30:00 | 7669.50 | 7747.74 | 7734.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 11:15:00 | 7678.50 | 7733.89 | 7729.51 | EMA400 retest candle locked (from upside) |

### Cycle 63 — SELL (started 2026-01-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 12:15:00 | 7633.50 | 7713.81 | 7720.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 13:15:00 | 7601.00 | 7691.25 | 7709.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 11:15:00 | 7439.50 | 7425.26 | 7521.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 12:00:00 | 7439.50 | 7425.26 | 7521.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 7582.00 | 7442.78 | 7491.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 10:00:00 | 7582.00 | 7442.78 | 7491.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 7505.00 | 7455.22 | 7492.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 11:15:00 | 7488.00 | 7455.22 | 7492.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-22 14:15:00 | 7616.50 | 7519.50 | 7513.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — BUY (started 2026-01-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 14:15:00 | 7616.50 | 7519.50 | 7513.64 | EMA200 above EMA400 |

### Cycle 65 — SELL (started 2026-01-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 11:15:00 | 7415.50 | 7497.31 | 7506.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 12:15:00 | 7361.50 | 7470.15 | 7493.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 7420.00 | 7367.89 | 7428.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 7420.00 | 7367.89 | 7428.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 7420.00 | 7367.89 | 7428.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:45:00 | 7433.00 | 7367.89 | 7428.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 7554.00 | 7405.11 | 7440.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 11:00:00 | 7554.00 | 7405.11 | 7440.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 11:15:00 | 7541.50 | 7432.39 | 7449.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 11:30:00 | 7558.00 | 7432.39 | 7449.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — BUY (started 2026-01-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 13:15:00 | 7544.00 | 7470.01 | 7464.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-27 14:15:00 | 7568.00 | 7489.61 | 7473.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 7492.50 | 7657.87 | 7598.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-29 09:15:00 | 7492.50 | 7657.87 | 7598.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 09:15:00 | 7492.50 | 7657.87 | 7598.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 09:30:00 | 7522.00 | 7657.87 | 7598.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 7357.50 | 7597.79 | 7576.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:30:00 | 7349.50 | 7597.79 | 7576.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2026-01-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 11:15:00 | 7255.50 | 7529.33 | 7547.34 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2026-02-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 14:15:00 | 7512.00 | 7401.14 | 7390.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 7606.50 | 7459.23 | 7419.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 7761.50 | 7779.75 | 7684.77 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 11:15:00 | 7828.00 | 7784.80 | 7695.70 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 11:45:00 | 7822.00 | 7793.34 | 7707.68 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-02-05 13:45:00 | 7823.50 | 7802.24 | 7726.76 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 7805.00 | 7812.55 | 7750.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:30:00 | 7753.00 | 7812.55 | 7750.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 7755.50 | 7807.73 | 7764.65 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-02-06 12:15:00 | 7755.50 | 7807.73 | 7764.65 | SL hit (close<ema400) qty=1.00 sl=7764.65 alert=retest1 |

### Cycle 69 — SELL (started 2026-02-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 09:15:00 | 7734.00 | 7950.23 | 7954.36 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2026-02-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-18 11:15:00 | 7937.00 | 7815.08 | 7805.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-19 09:15:00 | 8085.00 | 7948.11 | 7881.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-20 09:15:00 | 7961.00 | 8015.00 | 7958.53 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-20 09:15:00 | 7961.00 | 8015.00 | 7958.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 7961.00 | 8015.00 | 7958.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:30:00 | 7984.00 | 8015.00 | 7958.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 8052.50 | 8022.50 | 7967.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-23 09:45:00 | 8099.50 | 8046.15 | 8003.07 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 09:15:00 | 7909.00 | 7987.20 | 7990.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 7909.00 | 7987.20 | 7990.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-25 09:15:00 | 7730.00 | 7874.05 | 7924.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 14:15:00 | 7799.50 | 7797.95 | 7861.18 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 15:00:00 | 7799.50 | 7797.95 | 7861.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 7859.00 | 7805.69 | 7853.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 10:00:00 | 7859.00 | 7805.69 | 7853.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 10:15:00 | 7767.50 | 7798.05 | 7845.63 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-26 11:15:00 | 7750.50 | 7798.05 | 7845.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-02 09:15:00 | 7362.97 | 7548.53 | 7647.35 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 10:15:00 | 7634.50 | 7565.72 | 7646.18 | SL hit (close>ema200) qty=0.50 sl=7565.72 alert=retest2 |

### Cycle 72 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 6849.50 | 6759.23 | 6747.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 7003.50 | 6808.08 | 6770.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 6915.50 | 6971.06 | 6895.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 6915.50 | 6971.06 | 6895.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 6915.50 | 6971.06 | 6895.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:45:00 | 6900.50 | 6971.06 | 6895.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 10:15:00 | 6950.50 | 6966.95 | 6900.13 | EMA400 retest candle locked (from upside) |

### Cycle 73 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 6749.50 | 6875.50 | 6885.45 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2026-03-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 13:15:00 | 6987.00 | 6907.61 | 6897.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 09:15:00 | 7069.50 | 6935.19 | 6912.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-01 11:15:00 | 6913.00 | 6940.96 | 6920.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 11:15:00 | 6913.00 | 6940.96 | 6920.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 11:15:00 | 6913.00 | 6940.96 | 6920.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-01 11:45:00 | 6889.50 | 6940.96 | 6920.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 12:15:00 | 7030.00 | 6958.77 | 6930.08 | EMA400 retest candle locked (from upside) |

### Cycle 75 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 6767.50 | 6889.65 | 6904.61 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2026-04-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 10:15:00 | 7023.00 | 6823.16 | 6796.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 12:15:00 | 7269.50 | 6951.04 | 6861.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 7369.00 | 7422.33 | 7326.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 7369.00 | 7422.33 | 7326.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 7369.00 | 7422.33 | 7326.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 7390.50 | 7415.56 | 7332.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 11:15:00 | 7405.50 | 7415.56 | 7332.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 7466.00 | 7408.10 | 7360.41 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 14:15:00 | 7411.50 | 7414.54 | 7383.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 14:15:00 | 7431.50 | 7417.93 | 7388.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 15:15:00 | 7450.00 | 7417.93 | 7388.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-24 10:15:00 | 7611.50 | 7681.58 | 7686.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 77 — SELL (started 2026-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 10:15:00 | 7611.50 | 7681.58 | 7686.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 11:15:00 | 7595.50 | 7664.36 | 7677.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 7686.00 | 7618.94 | 7644.96 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 7686.00 | 7618.94 | 7644.96 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 7686.00 | 7618.94 | 7644.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:00:00 | 7686.00 | 7618.94 | 7644.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 7593.00 | 7613.75 | 7640.23 | EMA400 retest candle locked (from downside) |

### Cycle 78 — BUY (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 13:15:00 | 7709.00 | 7653.93 | 7653.69 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2026-04-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 14:15:00 | 7621.50 | 7647.44 | 7650.76 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-28 09:15:00 | 7557.00 | 7623.84 | 7639.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-28 15:15:00 | 7596.00 | 7582.71 | 7606.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-29 09:15:00 | 7596.00 | 7582.71 | 7606.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 7583.50 | 7582.86 | 7604.25 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 10:30:00 | 7553.50 | 7583.99 | 7602.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 11:15:00 | 7644.50 | 7596.09 | 7606.61 | SL hit (close>static) qty=1.00 sl=7636.00 alert=retest2 |

### Cycle 80 — BUY (started 2026-04-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 12:15:00 | 7719.50 | 7620.77 | 7616.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 13:15:00 | 7763.00 | 7649.22 | 7630.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 10:15:00 | 7679.50 | 7694.93 | 7661.65 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 10:15:00 | 7679.50 | 7694.93 | 7661.65 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 7679.50 | 7694.93 | 7661.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:00:00 | 7679.50 | 7694.93 | 7661.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 7675.00 | 7690.94 | 7662.87 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 7785.00 | 7674.08 | 7662.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-05 10:15:00 | 7616.50 | 7672.61 | 7676.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — SELL (started 2026-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 10:15:00 | 7616.50 | 7672.61 | 7676.89 | EMA200 below EMA400 |

### Cycle 82 — BUY (started 2026-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-05 12:15:00 | 7743.50 | 7691.81 | 7685.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 09:15:00 | 7812.00 | 7736.77 | 7710.38 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-07 10:15:00 | 7710.50 | 7745.50 | 7731.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-07 10:15:00 | 7710.50 | 7745.50 | 7731.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 10:15:00 | 7710.50 | 7745.50 | 7731.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 10:30:00 | 7722.00 | 7745.50 | 7731.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 11:15:00 | 7807.50 | 7757.90 | 7738.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-07 11:30:00 | 7704.50 | 7757.90 | 7738.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 12:15:00 | 8348.00 | 7875.92 | 7793.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 15:15:00 | 8777.00 | 8162.19 | 7946.28 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-08 10:15:00 | 9654.70 | 8688.18 | 8261.93 | Target hit (10%) qty=1.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-16 11:30:00 | 5399.00 | 2025-05-16 14:15:00 | 5249.30 | STOP_HIT | 1.00 | -2.77% |
| BUY | retest2 | 2025-05-19 09:15:00 | 5365.50 | 2025-05-20 13:15:00 | 5263.00 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2025-05-19 13:30:00 | 5313.20 | 2025-05-20 13:15:00 | 5263.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-05-19 14:30:00 | 5313.50 | 2025-05-20 13:15:00 | 5263.00 | STOP_HIT | 1.00 | -0.95% |
| BUY | retest2 | 2025-05-26 09:30:00 | 5518.60 | 2025-05-26 13:15:00 | 5430.00 | STOP_HIT | 1.00 | -1.61% |
| BUY | retest2 | 2025-05-27 09:45:00 | 5520.10 | 2025-05-30 15:15:00 | 5609.90 | STOP_HIT | 1.00 | 1.63% |
| SELL | retest2 | 2025-06-18 09:45:00 | 5251.50 | 2025-06-23 10:15:00 | 5268.50 | STOP_HIT | 1.00 | -0.32% |
| SELL | retest2 | 2025-06-18 12:45:00 | 5245.00 | 2025-06-23 12:15:00 | 5278.00 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-06-18 15:00:00 | 5247.50 | 2025-06-23 12:15:00 | 5278.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-06-19 10:15:00 | 5237.00 | 2025-06-23 12:15:00 | 5278.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-06-20 15:00:00 | 5163.50 | 2025-06-23 12:15:00 | 5278.00 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-07-09 09:15:00 | 5930.00 | 2025-07-14 12:15:00 | 5910.00 | STOP_HIT | 1.00 | -0.34% |
| BUY | retest2 | 2025-07-09 10:15:00 | 5985.00 | 2025-07-14 12:15:00 | 5910.00 | STOP_HIT | 1.00 | -1.25% |
| BUY | retest2 | 2025-07-11 13:15:00 | 5928.50 | 2025-07-14 12:15:00 | 5910.00 | STOP_HIT | 1.00 | -0.31% |
| BUY | retest2 | 2025-07-11 14:45:00 | 5937.00 | 2025-07-14 12:15:00 | 5910.00 | STOP_HIT | 1.00 | -0.45% |
| SELL | retest2 | 2025-07-24 10:30:00 | 6395.00 | 2025-07-29 13:15:00 | 6075.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 10:30:00 | 6395.00 | 2025-07-29 14:15:00 | 6475.00 | STOP_HIT | 0.50 | -1.25% |
| SELL | retest2 | 2025-09-08 09:15:00 | 6848.50 | 2025-09-12 10:15:00 | 6802.50 | STOP_HIT | 1.00 | 0.67% |
| SELL | retest2 | 2025-09-08 13:00:00 | 6842.00 | 2025-09-12 10:15:00 | 6802.50 | STOP_HIT | 1.00 | 0.58% |
| SELL | retest2 | 2025-09-08 14:15:00 | 6844.00 | 2025-09-12 10:15:00 | 6802.50 | STOP_HIT | 1.00 | 0.61% |
| SELL | retest2 | 2025-09-16 11:30:00 | 6666.00 | 2025-09-16 13:15:00 | 6713.00 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-09-19 10:45:00 | 6836.50 | 2025-09-19 13:15:00 | 6734.50 | STOP_HIT | 1.00 | -1.49% |
| BUY | retest2 | 2025-09-23 09:15:00 | 6846.00 | 2025-09-24 10:15:00 | 6746.50 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-09-23 14:30:00 | 6816.00 | 2025-09-24 10:15:00 | 6746.50 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2025-10-15 14:00:00 | 6425.00 | 2025-10-16 09:15:00 | 6790.00 | STOP_HIT | 1.00 | -5.68% |
| SELL | retest2 | 2025-10-15 15:15:00 | 6409.00 | 2025-10-16 09:15:00 | 6790.00 | STOP_HIT | 1.00 | -5.94% |
| BUY | retest2 | 2025-10-23 12:30:00 | 6705.00 | 2025-10-23 14:15:00 | 6614.00 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-10-28 12:15:00 | 6480.00 | 2025-10-30 10:15:00 | 6583.00 | STOP_HIT | 1.00 | -1.59% |
| SELL | retest2 | 2025-10-28 13:30:00 | 6468.00 | 2025-10-30 10:15:00 | 6583.00 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-10-28 14:00:00 | 6434.00 | 2025-10-30 10:15:00 | 6583.00 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-10-29 15:00:00 | 6463.00 | 2025-10-30 10:15:00 | 6583.00 | STOP_HIT | 1.00 | -1.86% |
| SELL | retest2 | 2025-11-18 10:30:00 | 6678.50 | 2025-11-25 09:15:00 | 6942.50 | STOP_HIT | 1.00 | -3.95% |
| SELL | retest2 | 2025-11-18 11:45:00 | 6678.00 | 2025-11-25 09:15:00 | 6942.50 | STOP_HIT | 1.00 | -3.96% |
| SELL | retest2 | 2025-11-21 11:00:00 | 6640.00 | 2025-11-25 09:15:00 | 6942.50 | STOP_HIT | 1.00 | -4.56% |
| SELL | retest2 | 2025-11-24 09:30:00 | 6588.50 | 2025-11-25 09:15:00 | 6942.50 | STOP_HIT | 1.00 | -5.37% |
| BUY | retest2 | 2025-11-27 12:15:00 | 7042.00 | 2025-12-02 10:15:00 | 6980.00 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-11-28 11:45:00 | 7013.50 | 2025-12-02 10:15:00 | 6980.00 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2025-11-28 13:00:00 | 7005.00 | 2025-12-02 10:15:00 | 6980.00 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-12-01 12:30:00 | 6995.50 | 2025-12-02 10:15:00 | 6980.00 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-12-18 09:15:00 | 6922.50 | 2025-12-18 15:15:00 | 7076.50 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-12-23 09:15:00 | 7076.50 | 2025-12-24 15:15:00 | 7002.50 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-12-23 09:45:00 | 7066.50 | 2025-12-24 15:15:00 | 7002.50 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-12-23 11:15:00 | 7098.00 | 2025-12-24 15:15:00 | 7002.50 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2025-12-24 09:15:00 | 7078.50 | 2025-12-24 15:15:00 | 7002.50 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-12-24 13:45:00 | 7102.50 | 2025-12-24 15:15:00 | 7002.50 | STOP_HIT | 1.00 | -1.41% |
| SELL | retest2 | 2026-01-08 11:15:00 | 7703.00 | 2026-01-12 11:15:00 | 7334.95 | PARTIAL | 0.50 | 4.78% |
| SELL | retest2 | 2026-01-08 11:15:00 | 7703.00 | 2026-01-12 12:15:00 | 7566.00 | STOP_HIT | 0.50 | 1.78% |
| SELL | retest2 | 2026-01-08 12:45:00 | 7721.00 | 2026-01-14 09:15:00 | 7754.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2026-01-08 13:15:00 | 7709.00 | 2026-01-14 09:15:00 | 7754.00 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-01-08 14:00:00 | 7716.00 | 2026-01-14 09:15:00 | 7754.00 | STOP_HIT | 1.00 | -0.49% |
| SELL | retest2 | 2026-01-22 11:15:00 | 7488.00 | 2026-01-22 14:15:00 | 7616.50 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest1 | 2026-02-05 11:15:00 | 7828.00 | 2026-02-06 12:15:00 | 7755.50 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest1 | 2026-02-05 11:45:00 | 7822.00 | 2026-02-06 12:15:00 | 7755.50 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest1 | 2026-02-05 13:45:00 | 7823.50 | 2026-02-06 12:15:00 | 7755.50 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-02-06 14:45:00 | 7800.00 | 2026-02-13 09:15:00 | 7734.00 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2026-02-23 09:45:00 | 8099.50 | 2026-02-24 09:15:00 | 7909.00 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2026-02-26 11:15:00 | 7750.50 | 2026-03-02 09:15:00 | 7362.97 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-26 11:15:00 | 7750.50 | 2026-03-02 10:15:00 | 7634.50 | STOP_HIT | 0.50 | 1.50% |
| BUY | retest2 | 2026-04-13 10:45:00 | 7390.50 | 2026-04-24 10:15:00 | 7611.50 | STOP_HIT | 1.00 | 2.99% |
| BUY | retest2 | 2026-04-13 11:15:00 | 7405.50 | 2026-04-24 10:15:00 | 7611.50 | STOP_HIT | 1.00 | 2.78% |
| BUY | retest2 | 2026-04-15 09:15:00 | 7466.00 | 2026-04-24 10:15:00 | 7611.50 | STOP_HIT | 1.00 | 1.95% |
| BUY | retest2 | 2026-04-15 14:15:00 | 7411.50 | 2026-04-24 10:15:00 | 7611.50 | STOP_HIT | 1.00 | 2.70% |
| BUY | retest2 | 2026-04-15 15:15:00 | 7450.00 | 2026-04-24 10:15:00 | 7611.50 | STOP_HIT | 1.00 | 2.17% |
| SELL | retest2 | 2026-04-29 10:30:00 | 7553.50 | 2026-04-29 11:15:00 | 7644.50 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-05-04 09:15:00 | 7785.00 | 2026-05-05 10:15:00 | 7616.50 | STOP_HIT | 1.00 | -2.16% |
| BUY | retest2 | 2026-05-07 15:15:00 | 8777.00 | 2026-05-08 10:15:00 | 9654.70 | TARGET_HIT | 1.00 | 10.00% |
