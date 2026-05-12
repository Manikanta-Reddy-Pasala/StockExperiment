# Bayer Cropscience Ltd. (BAYERCROP)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 4600.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 72 |
| ALERT1 | 45 |
| ALERT2 | 45 |
| ALERT2_SKIP | 19 |
| ALERT3 | 113 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 80 |
| PARTIAL | 10 |
| TARGET_HIT | 5 |
| STOP_HIT | 78 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 91 (incl. partial bookings)
- **Trades open at end:** 4
- **Winners / losers:** 33 / 58
- **Target hits / Stop hits / Partials:** 5 / 76 / 10
- **Avg / median % per leg:** 0.77% / -0.75%
- **Sum % (uncompounded):** 70.39%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 33 | 9 | 27.3% | 5 | 27 | 1 | 0.91% | 30.0% |
| BUY @ 2nd Alert (retest1) | 6 | 2 | 33.3% | 0 | 5 | 1 | 1.39% | 8.3% |
| BUY @ 3rd Alert (retest2) | 27 | 7 | 25.9% | 5 | 22 | 0 | 0.80% | 21.7% |
| SELL (all) | 58 | 24 | 41.4% | 0 | 49 | 9 | 0.70% | 40.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 58 | 24 | 41.4% | 0 | 49 | 9 | 0.70% | 40.4% |
| retest1 (combined) | 6 | 2 | 33.3% | 0 | 5 | 1 | 1.39% | 8.3% |
| retest2 (combined) | 85 | 31 | 36.5% | 5 | 71 | 9 | 0.73% | 62.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 4731.10 | 4656.42 | 4646.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 4764.50 | 4685.68 | 4661.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 09:15:00 | 4723.50 | 4728.49 | 4693.16 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-14 09:15:00 | 4812.00 | 4755.21 | 4724.37 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-16 14:15:00 | 5052.60 | 4993.03 | 4935.55 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 5110.20 | 5071.07 | 5018.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:30:00 | 5168.00 | 5106.87 | 5064.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 13:30:00 | 5150.00 | 5132.01 | 5091.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 09:15:00 | 5127.20 | 5135.91 | 5104.13 | SL hit (close<ema200) qty=0.50 sl=5135.91 alert=retest1 |

### Cycle 2 — SELL (started 2025-05-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 15:15:00 | 5047.50 | 5087.34 | 5091.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-23 12:15:00 | 5033.00 | 5066.51 | 5079.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-26 09:15:00 | 5086.00 | 5037.51 | 5058.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-26 09:15:00 | 5086.00 | 5037.51 | 5058.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-26 09:15:00 | 5086.00 | 5037.51 | 5058.23 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 14:15:00 | 5136.40 | 5080.86 | 5073.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 09:15:00 | 5754.00 | 5220.15 | 5138.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-29 11:15:00 | 5693.00 | 5707.12 | 5590.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-29 12:00:00 | 5693.00 | 5707.12 | 5590.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 5712.90 | 5699.89 | 5629.87 | EMA400 retest candle locked (from upside) |

### Cycle 4 — SELL (started 2025-06-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 15:15:00 | 5613.00 | 5637.08 | 5640.30 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-06-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-03 09:15:00 | 5694.00 | 5648.47 | 5645.18 | EMA200 above EMA400 |

### Cycle 6 — SELL (started 2025-06-03 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-03 15:15:00 | 5607.00 | 5641.32 | 5644.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 11:15:00 | 5596.50 | 5622.60 | 5634.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 14:15:00 | 5496.50 | 5490.50 | 5526.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-09 14:45:00 | 5497.50 | 5490.50 | 5526.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-10 09:15:00 | 5484.50 | 5490.82 | 5520.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 13:30:00 | 5438.50 | 5483.19 | 5507.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 13:00:00 | 5469.00 | 5481.77 | 5494.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 11:00:00 | 5465.00 | 5442.10 | 5457.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 11:45:00 | 5460.00 | 5444.28 | 5456.86 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 12:15:00 | 5450.00 | 5445.43 | 5456.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-13 15:00:00 | 5439.00 | 5446.47 | 5454.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-16 09:15:00 | 5439.50 | 5449.38 | 5455.52 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 10:15:00 | 5518.00 | 5460.08 | 5459.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 10:15:00 | 5518.00 | 5460.08 | 5459.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-17 11:15:00 | 5586.00 | 5529.50 | 5498.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-18 11:15:00 | 5607.00 | 5609.68 | 5563.57 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-18 12:00:00 | 5607.00 | 5609.68 | 5563.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 5600.00 | 5646.71 | 5612.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 5600.00 | 5646.71 | 5612.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 5591.50 | 5635.67 | 5610.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:45:00 | 5589.00 | 5635.67 | 5610.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 5577.00 | 5610.14 | 5605.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:45:00 | 5568.00 | 5610.14 | 5605.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 5576.50 | 5603.41 | 5602.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:30:00 | 5569.00 | 5603.41 | 5602.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — SELL (started 2025-06-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 12:15:00 | 5545.50 | 5591.83 | 5597.26 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-23 10:15:00 | 5537.50 | 5565.96 | 5580.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-24 10:15:00 | 5595.00 | 5542.43 | 5556.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 10:15:00 | 5595.00 | 5542.43 | 5556.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 10:15:00 | 5595.00 | 5542.43 | 5556.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:00:00 | 5595.00 | 5542.43 | 5556.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 11:15:00 | 5659.00 | 5565.74 | 5565.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-24 11:30:00 | 5659.50 | 5565.74 | 5565.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2025-06-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 12:15:00 | 5690.00 | 5590.59 | 5577.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-25 14:15:00 | 5750.00 | 5689.71 | 5646.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 15:15:00 | 6350.00 | 6370.40 | 6286.15 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 10:00:00 | 6429.00 | 6382.12 | 6299.14 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 12:30:00 | 6432.00 | 6404.30 | 6331.16 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-03 14:30:00 | 6430.50 | 6411.81 | 6347.38 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-07-04 09:15:00 | 6436.50 | 6407.05 | 6351.08 | BUY ENTRY1 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 15:15:00 | 6380.50 | 6437.92 | 6419.28 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-07-07 15:15:00 | 6380.50 | 6437.92 | 6419.28 | SL hit (close<ema400) qty=1.00 sl=6419.28 alert=retest1 |

### Cycle 10 — SELL (started 2025-07-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 09:15:00 | 6320.00 | 6430.75 | 6437.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-10 10:15:00 | 6283.00 | 6401.20 | 6423.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-14 14:15:00 | 6225.50 | 6197.36 | 6237.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-14 14:15:00 | 6225.50 | 6197.36 | 6237.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 14:15:00 | 6225.50 | 6197.36 | 6237.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 15:00:00 | 6225.50 | 6197.36 | 6237.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 6233.00 | 6208.75 | 6235.78 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 6344.50 | 6258.58 | 6255.14 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 10:15:00 | 6280.00 | 6314.81 | 6315.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 13:15:00 | 6272.50 | 6300.03 | 6308.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 15:15:00 | 6304.00 | 6297.22 | 6305.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-22 09:15:00 | 6336.50 | 6297.22 | 6305.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 6285.00 | 6294.78 | 6303.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:15:00 | 6270.00 | 6294.78 | 6303.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 13:45:00 | 6268.00 | 6274.90 | 6289.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-23 09:15:00 | 6257.50 | 6277.74 | 6288.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-23 10:15:00 | 6396.00 | 6302.47 | 6298.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2025-07-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-23 10:15:00 | 6396.00 | 6302.47 | 6298.02 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-07-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 10:15:00 | 6218.00 | 6306.89 | 6315.42 | EMA200 below EMA400 |

### Cycle 15 — BUY (started 2025-07-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-28 09:15:00 | 6470.00 | 6326.31 | 6316.84 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-30 12:15:00 | 6314.50 | 6352.85 | 6352.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-30 13:15:00 | 6266.50 | 6335.58 | 6345.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-31 09:15:00 | 6408.00 | 6345.96 | 6347.19 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-31 09:15:00 | 6408.00 | 6345.96 | 6347.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-31 09:15:00 | 6408.00 | 6345.96 | 6347.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-31 09:45:00 | 6417.50 | 6345.96 | 6347.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2025-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-31 10:15:00 | 6419.50 | 6360.67 | 6353.77 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-31 14:15:00 | 6327.00 | 6350.12 | 6350.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-01 10:15:00 | 6317.50 | 6341.16 | 6346.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-04 15:15:00 | 6265.00 | 6247.64 | 6276.55 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-05 09:15:00 | 6230.00 | 6247.64 | 6276.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 09:15:00 | 6223.50 | 6242.81 | 6271.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-06 11:30:00 | 6200.00 | 6266.87 | 6273.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 12:15:00 | 6333.00 | 6280.10 | 6279.26 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 12:15:00 | 6333.00 | 6280.10 | 6279.26 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-08-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 14:15:00 | 5994.50 | 6226.00 | 6255.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 15:15:00 | 5946.50 | 6170.10 | 6226.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 14:15:00 | 5643.00 | 5630.88 | 5720.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 15:00:00 | 5643.00 | 5630.88 | 5720.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 10:15:00 | 5599.50 | 5628.90 | 5666.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 13:45:00 | 5592.00 | 5619.10 | 5652.24 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 11:15:00 | 5592.00 | 5626.05 | 5646.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 5688.00 | 5644.45 | 5645.59 | SL hit (close>static) qty=1.00 sl=5667.50 alert=retest2 |

### Cycle 21 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 5614.00 | 5581.71 | 5579.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-21 10:15:00 | 5643.50 | 5594.07 | 5585.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-21 13:15:00 | 5604.50 | 5604.66 | 5593.10 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-21 14:00:00 | 5604.50 | 5604.66 | 5593.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 14:15:00 | 5589.50 | 5601.63 | 5592.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-21 15:00:00 | 5589.50 | 5601.63 | 5592.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 15:15:00 | 5624.00 | 5606.10 | 5595.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-22 09:30:00 | 5565.00 | 5600.78 | 5594.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2025-08-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 10:15:00 | 5524.50 | 5585.53 | 5587.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 11:15:00 | 5510.00 | 5570.42 | 5580.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 5565.00 | 5543.81 | 5560.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 5565.00 | 5543.81 | 5560.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 5565.00 | 5543.81 | 5560.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:45:00 | 5568.00 | 5543.81 | 5560.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 5520.00 | 5539.05 | 5557.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 14:00:00 | 5499.50 | 5527.09 | 5546.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 10:15:00 | 5224.52 | 5285.00 | 5353.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-01 09:15:00 | 5243.00 | 5240.48 | 5297.05 | SL hit (close>ema200) qty=0.50 sl=5240.48 alert=retest2 |

### Cycle 23 — BUY (started 2025-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 15:15:00 | 5125.00 | 5108.78 | 5107.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 12:15:00 | 5134.50 | 5119.01 | 5113.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 14:15:00 | 5115.00 | 5120.05 | 5114.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 14:15:00 | 5115.00 | 5120.05 | 5114.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 5115.00 | 5120.05 | 5114.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 15:00:00 | 5115.00 | 5120.05 | 5114.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 15:15:00 | 5109.00 | 5117.84 | 5114.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 09:15:00 | 5139.00 | 5117.84 | 5114.43 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-12 12:15:00 | 5103.00 | 5120.79 | 5117.86 | SL hit (close<static) qty=1.00 sl=5103.50 alert=retest2 |

### Cycle 24 — SELL (started 2025-09-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 13:15:00 | 5082.00 | 5113.03 | 5114.60 | EMA200 below EMA400 |

### Cycle 25 — BUY (started 2025-09-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-16 09:15:00 | 5175.00 | 5112.75 | 5108.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-16 10:15:00 | 5209.50 | 5132.10 | 5117.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-17 09:15:00 | 5196.50 | 5197.11 | 5163.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-17 10:00:00 | 5196.50 | 5197.11 | 5163.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-17 10:15:00 | 5185.00 | 5194.69 | 5165.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-18 09:30:00 | 5214.50 | 5200.80 | 5180.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-19 09:30:00 | 5223.50 | 5200.82 | 5188.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 10:15:00 | 5135.50 | 5222.11 | 5228.21 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 5135.50 | 5222.11 | 5228.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 11:15:00 | 5099.50 | 5197.59 | 5216.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-26 09:15:00 | 5001.00 | 4994.09 | 5039.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-26 10:00:00 | 5001.00 | 4994.09 | 5039.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 10:15:00 | 5021.00 | 4999.47 | 5037.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-26 11:00:00 | 5021.00 | 4999.47 | 5037.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-26 11:15:00 | 5051.50 | 5009.88 | 5038.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 09:30:00 | 5005.00 | 5030.66 | 5040.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 10:15:00 | 5006.50 | 5030.66 | 5040.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 10:15:00 | 4964.50 | 4911.43 | 4911.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-10-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 10:15:00 | 4964.50 | 4911.43 | 4911.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 14:15:00 | 5032.00 | 4958.50 | 4935.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 12:15:00 | 5000.00 | 5006.70 | 4972.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 12:45:00 | 4998.00 | 5006.70 | 4972.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 15:15:00 | 4989.00 | 5004.19 | 4980.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 10:00:00 | 5007.30 | 5004.81 | 4982.88 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 11:15:00 | 4977.00 | 4996.88 | 4982.93 | SL hit (close<static) qty=1.00 sl=4980.00 alert=retest2 |

### Cycle 28 — SELL (started 2025-10-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 10:15:00 | 4987.30 | 5003.71 | 5003.93 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-10-15 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 14:15:00 | 5046.10 | 5000.95 | 4996.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 15:15:00 | 5104.30 | 5021.62 | 5006.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-17 11:15:00 | 5051.40 | 5073.06 | 5051.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-17 12:00:00 | 5051.40 | 5073.06 | 5051.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 12:15:00 | 5036.30 | 5065.71 | 5049.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:00:00 | 5036.30 | 5065.71 | 5049.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 13:15:00 | 5035.30 | 5059.63 | 5048.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 13:45:00 | 5038.20 | 5059.63 | 5048.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-17 14:15:00 | 5027.90 | 5053.28 | 5046.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-17 14:30:00 | 5025.00 | 5053.28 | 5046.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 30 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 4999.50 | 5038.80 | 5041.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-20 11:15:00 | 4984.50 | 5020.12 | 5031.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-21 13:15:00 | 5017.20 | 5004.96 | 5018.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-21 13:15:00 | 5017.20 | 5004.96 | 5018.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 5017.20 | 5004.96 | 5018.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 5014.90 | 5004.96 | 5018.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 09:15:00 | 5041.70 | 5014.40 | 5020.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-23 10:00:00 | 5041.70 | 5014.40 | 5020.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 10:15:00 | 5018.00 | 5015.12 | 5020.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 11:45:00 | 4982.40 | 5010.47 | 5017.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 13:00:00 | 4999.90 | 5008.36 | 5016.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 14:00:00 | 4996.80 | 5006.05 | 5014.56 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-23 15:00:00 | 4999.70 | 5004.78 | 5013.21 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 5020.00 | 5004.98 | 5011.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:00:00 | 5020.00 | 5004.98 | 5011.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 5007.40 | 5005.46 | 5011.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 11:15:00 | 5000.60 | 5005.46 | 5011.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 13:45:00 | 4999.90 | 5009.82 | 5012.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 5053.30 | 5021.55 | 5017.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 5053.30 | 5021.55 | 5017.23 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-10-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-28 10:15:00 | 4999.40 | 5019.79 | 5020.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-28 13:15:00 | 4977.90 | 5005.48 | 5013.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-29 15:15:00 | 4959.00 | 4934.77 | 4966.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-30 09:15:00 | 4913.80 | 4934.77 | 4966.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 4905.80 | 4928.97 | 4960.56 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 13:45:00 | 4881.80 | 4910.81 | 4940.70 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 11:45:00 | 4879.50 | 4901.73 | 4925.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 12:15:00 | 4878.40 | 4901.73 | 4925.12 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 12:45:00 | 4878.10 | 4896.38 | 4920.57 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 4896.60 | 4888.70 | 4908.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 09:30:00 | 4909.60 | 4888.70 | 4908.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 14:15:00 | 4869.90 | 4879.36 | 4896.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 14:45:00 | 4883.90 | 4879.36 | 4896.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 10:15:00 | 4637.71 | 4736.61 | 4789.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 10:15:00 | 4635.52 | 4736.61 | 4789.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 10:15:00 | 4634.48 | 4736.61 | 4789.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-07 10:15:00 | 4634.19 | 4736.61 | 4789.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-10 14:15:00 | 4642.10 | 4617.21 | 4668.49 | SL hit (close>ema200) qty=0.50 sl=4617.21 alert=retest2 |

### Cycle 33 — BUY (started 2025-11-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 12:15:00 | 4555.00 | 4523.98 | 4521.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 14:15:00 | 4557.90 | 4535.57 | 4527.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 4512.60 | 4537.12 | 4529.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 4512.60 | 4537.12 | 4529.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 4512.60 | 4537.12 | 4529.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:45:00 | 4514.50 | 4537.12 | 4529.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 10:15:00 | 4503.90 | 4530.48 | 4527.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 10:45:00 | 4503.30 | 4530.48 | 4527.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 4549.00 | 4549.29 | 4539.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 12:15:00 | 4580.00 | 4552.25 | 4545.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-20 14:15:00 | 4577.00 | 4556.02 | 4548.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-21 13:45:00 | 4574.30 | 4553.36 | 4550.37 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-21 15:15:00 | 4528.30 | 4547.15 | 4547.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — SELL (started 2025-11-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 15:15:00 | 4528.30 | 4547.15 | 4547.99 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 11:15:00 | 4512.90 | 4539.59 | 4544.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 14:15:00 | 4534.90 | 4532.91 | 4539.51 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 14:15:00 | 4534.90 | 4532.91 | 4539.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 4534.90 | 4532.91 | 4539.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 4534.90 | 4532.91 | 4539.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 15:15:00 | 4534.00 | 4533.13 | 4539.01 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 09:15:00 | 4530.10 | 4533.13 | 4539.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 4535.80 | 4533.66 | 4538.72 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2025-11-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 15:15:00 | 4590.00 | 4548.83 | 4544.06 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 11:15:00 | 4466.50 | 4549.36 | 4554.70 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-11-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 09:15:00 | 4576.00 | 4554.08 | 4553.41 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-28 10:15:00 | 4593.90 | 4562.04 | 4557.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 11:15:00 | 4612.40 | 4620.57 | 4597.68 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-01 12:00:00 | 4612.40 | 4620.57 | 4597.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 4585.00 | 4613.46 | 4596.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 13:00:00 | 4585.00 | 4613.46 | 4596.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 13:15:00 | 4565.30 | 4603.82 | 4593.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 14:00:00 | 4565.30 | 4603.82 | 4593.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 4599.00 | 4602.25 | 4594.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 09:15:00 | 4605.00 | 4602.25 | 4594.69 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 13:45:00 | 4614.60 | 4606.21 | 4599.95 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 15:00:00 | 4607.40 | 4606.45 | 4600.63 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 4571.00 | 4598.17 | 4597.80 | SL hit (close<static) qty=1.00 sl=4581.50 alert=retest2 |

### Cycle 38 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 4568.30 | 4592.19 | 4595.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 11:15:00 | 4545.00 | 4582.76 | 4590.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 11:15:00 | 4556.20 | 4539.72 | 4559.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 12:00:00 | 4556.20 | 4539.72 | 4559.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 4550.80 | 4541.94 | 4558.45 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 13:45:00 | 4511.60 | 4536.33 | 4554.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-18 15:15:00 | 4286.02 | 4316.70 | 4343.68 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-19 11:15:00 | 4315.00 | 4312.73 | 4334.88 | SL hit (close>ema200) qty=0.50 sl=4312.73 alert=retest2 |

### Cycle 39 — BUY (started 2025-12-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-22 13:15:00 | 4355.30 | 4336.83 | 4334.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-22 14:15:00 | 4369.80 | 4343.42 | 4337.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 12:15:00 | 4555.50 | 4560.03 | 4524.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-29 13:15:00 | 4552.50 | 4560.03 | 4524.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 13:15:00 | 4513.10 | 4550.65 | 4523.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:00:00 | 4513.10 | 4550.65 | 4523.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 4545.40 | 4549.60 | 4525.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:45:00 | 4532.10 | 4549.60 | 4525.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 4530.00 | 4545.68 | 4526.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:45:00 | 4538.40 | 4542.34 | 4526.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 4481.90 | 4530.25 | 4522.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:00:00 | 4481.90 | 4530.25 | 4522.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 4518.40 | 4527.88 | 4521.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 11:30:00 | 4507.50 | 4527.88 | 4521.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 4512.70 | 4524.85 | 4521.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:30:00 | 4528.80 | 4524.85 | 4521.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 4551.20 | 4530.12 | 4523.84 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2026-01-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 12:15:00 | 4530.00 | 4545.33 | 4546.09 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2026-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 14:15:00 | 4567.10 | 4548.99 | 4547.58 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 09:15:00 | 4520.00 | 4543.39 | 4545.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-06 11:15:00 | 4503.00 | 4531.57 | 4539.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-06 14:15:00 | 4525.20 | 4524.39 | 4533.66 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-06 15:00:00 | 4525.20 | 4524.39 | 4533.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 4529.80 | 4521.66 | 4529.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 10:30:00 | 4537.10 | 4521.66 | 4529.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 4532.10 | 4523.75 | 4529.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:30:00 | 4528.00 | 4523.75 | 4529.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 4529.80 | 4524.96 | 4529.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 09:15:00 | 4515.70 | 4528.88 | 4530.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:00:00 | 4515.00 | 4525.32 | 4528.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-08 11:45:00 | 4513.20 | 4521.02 | 4526.31 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-20 14:15:00 | 4289.91 | 4368.39 | 4384.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 14:15:00 | 4398.00 | 4368.39 | 4384.80 | SL hit (close>static) qty=0.50 sl=4368.39 alert=retest2 |

### Cycle 43 — BUY (started 2026-01-21 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-21 15:15:00 | 4399.90 | 4387.18 | 4385.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 09:15:00 | 4435.50 | 4396.85 | 4390.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-22 10:15:00 | 4391.60 | 4395.80 | 4390.56 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-22 10:15:00 | 4391.60 | 4395.80 | 4390.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 4391.60 | 4395.80 | 4390.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 4391.60 | 4395.80 | 4390.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 4375.00 | 4391.64 | 4389.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:45:00 | 4361.10 | 4391.64 | 4389.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2026-01-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-22 12:15:00 | 4360.70 | 4385.45 | 4386.56 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2026-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-27 09:15:00 | 4404.90 | 4384.00 | 4383.37 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2026-01-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 11:15:00 | 4368.40 | 4381.52 | 4382.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 12:15:00 | 4363.80 | 4377.97 | 4380.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 4369.00 | 4368.94 | 4375.28 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 15:15:00 | 4369.00 | 4368.94 | 4375.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 4369.00 | 4368.94 | 4375.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 4394.60 | 4368.94 | 4375.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 4372.90 | 4369.73 | 4375.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 10:45:00 | 4334.90 | 4365.41 | 4371.53 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-29 11:45:00 | 4337.90 | 4364.19 | 4370.42 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-30 12:15:00 | 4430.00 | 4377.92 | 4373.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 12:15:00 | 4430.00 | 4377.92 | 4373.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 13:15:00 | 4479.10 | 4398.15 | 4382.74 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 4409.20 | 4426.98 | 4406.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 4409.20 | 4426.98 | 4406.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 4409.20 | 4426.98 | 4406.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 11:45:00 | 4413.50 | 4426.98 | 4406.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 4386.50 | 4418.88 | 4404.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 4391.60 | 4418.88 | 4404.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 13:15:00 | 4385.40 | 4412.19 | 4402.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 13:30:00 | 4372.90 | 4412.19 | 4402.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2026-02-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 15:15:00 | 4351.20 | 4396.26 | 4397.06 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-02-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-02 15:15:00 | 4440.00 | 4399.52 | 4395.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-03 09:15:00 | 4450.00 | 4409.61 | 4400.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-03 14:15:00 | 4423.80 | 4427.26 | 4414.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 14:15:00 | 4423.80 | 4427.26 | 4414.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 14:15:00 | 4423.80 | 4427.26 | 4414.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-03 14:30:00 | 4423.00 | 4427.26 | 4414.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 15:15:00 | 4415.40 | 4424.89 | 4414.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 09:30:00 | 4452.30 | 4426.95 | 4416.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 10:30:00 | 4445.10 | 4430.70 | 4419.05 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-04 13:00:00 | 4447.50 | 4434.73 | 4422.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 14:45:00 | 4484.00 | 4467.08 | 4453.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 15:15:00 | 4464.00 | 4466.46 | 4454.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:15:00 | 4428.50 | 4466.46 | 4454.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 4436.40 | 4460.45 | 4452.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 13:00:00 | 4485.30 | 4463.80 | 4455.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-02-12 09:15:00 | 4897.53 | 4743.46 | 4684.39 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-02-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-16 12:15:00 | 4690.00 | 4743.11 | 4746.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-16 15:15:00 | 4670.00 | 4711.52 | 4729.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 4751.00 | 4719.42 | 4731.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 4751.00 | 4719.42 | 4731.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 4751.00 | 4719.42 | 4731.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 4751.00 | 4719.42 | 4731.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 4749.40 | 4725.41 | 4733.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 4751.80 | 4725.41 | 4733.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 51 — BUY (started 2026-02-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 13:15:00 | 4758.00 | 4740.01 | 4738.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 09:15:00 | 4833.90 | 4763.69 | 4750.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 09:15:00 | 4772.30 | 4802.54 | 4782.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 09:15:00 | 4772.30 | 4802.54 | 4782.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 09:15:00 | 4772.30 | 4802.54 | 4782.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 09:30:00 | 4773.10 | 4802.54 | 4782.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 10:15:00 | 4770.30 | 4796.10 | 4781.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 10:30:00 | 4761.50 | 4796.10 | 4781.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 4751.10 | 4781.32 | 4776.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:00:00 | 4751.10 | 4781.32 | 4776.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 4775.00 | 4779.99 | 4776.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 4775.00 | 4779.99 | 4776.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 4779.80 | 4779.95 | 4777.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 4740.00 | 4779.95 | 4777.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 4766.80 | 4777.32 | 4776.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:30:00 | 4757.80 | 4777.32 | 4776.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2026-02-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 10:15:00 | 4752.70 | 4772.40 | 4774.09 | EMA200 below EMA400 |

### Cycle 53 — BUY (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-20 12:15:00 | 4800.10 | 4775.31 | 4774.96 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-02-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 09:15:00 | 4747.70 | 4782.96 | 4784.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-24 12:15:00 | 4707.50 | 4751.91 | 4768.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-25 09:15:00 | 4738.50 | 4735.26 | 4753.94 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-25 10:00:00 | 4738.50 | 4735.26 | 4753.94 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-26 09:15:00 | 4736.20 | 4730.05 | 4740.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-26 09:45:00 | 4743.30 | 4730.05 | 4740.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 09:15:00 | 4567.40 | 4558.16 | 4588.12 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 12:45:00 | 4500.80 | 4547.28 | 4575.95 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-06 14:15:00 | 4600.00 | 4566.58 | 4566.30 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2026-03-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 14:15:00 | 4600.00 | 4566.58 | 4566.30 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 4488.30 | 4556.64 | 4562.18 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-03-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 12:15:00 | 4615.60 | 4550.56 | 4544.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-11 09:15:00 | 4625.00 | 4585.99 | 4564.99 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-11 11:15:00 | 4580.20 | 4587.07 | 4569.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-11 11:45:00 | 4575.10 | 4587.07 | 4569.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 12:15:00 | 4607.40 | 4591.14 | 4572.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 12:45:00 | 4607.20 | 4591.14 | 4572.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 13:15:00 | 4584.70 | 4589.85 | 4573.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 13:30:00 | 4602.00 | 4589.85 | 4573.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-11 14:15:00 | 4514.70 | 4574.82 | 4568.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-11 15:00:00 | 4514.70 | 4574.82 | 4568.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — SELL (started 2026-03-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 15:15:00 | 4500.00 | 4559.86 | 4562.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 09:15:00 | 4463.00 | 4507.94 | 4529.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 13:15:00 | 4513.90 | 4493.12 | 4513.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 13:15:00 | 4513.90 | 4493.12 | 4513.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 13:15:00 | 4513.90 | 4493.12 | 4513.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 13:45:00 | 4500.00 | 4493.12 | 4513.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 14:15:00 | 4511.00 | 4496.70 | 4513.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 14:30:00 | 4499.80 | 4496.70 | 4513.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 15:15:00 | 4511.00 | 4499.56 | 4513.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-16 09:15:00 | 4473.60 | 4499.56 | 4513.37 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 11:15:00 | 4446.80 | 4478.05 | 4486.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 15:15:00 | 4466.40 | 4472.52 | 4479.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 12:15:00 | 4499.40 | 4483.53 | 4482.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 4499.40 | 4483.53 | 4482.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-19 09:15:00 | 4550.10 | 4497.58 | 4489.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 11:15:00 | 4495.00 | 4498.96 | 4491.85 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-19 12:00:00 | 4495.00 | 4498.96 | 4491.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 12:15:00 | 4511.30 | 4501.43 | 4493.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 12:30:00 | 4500.80 | 4501.43 | 4493.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 13:15:00 | 4496.00 | 4500.34 | 4493.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 13:30:00 | 4492.50 | 4500.34 | 4493.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 14:15:00 | 4460.60 | 4492.39 | 4490.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 15:00:00 | 4460.60 | 4492.39 | 4490.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2026-03-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 15:15:00 | 4478.00 | 4489.51 | 4489.65 | EMA200 below EMA400 |

### Cycle 61 — BUY (started 2026-03-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 12:15:00 | 4584.80 | 4507.30 | 4496.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-20 13:15:00 | 4686.00 | 4543.04 | 4514.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-23 09:15:00 | 4553.00 | 4589.07 | 4546.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-23 10:00:00 | 4553.00 | 4589.07 | 4546.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 10:15:00 | 4490.00 | 4569.25 | 4541.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 10:45:00 | 4475.20 | 4569.25 | 4541.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 11:15:00 | 4550.60 | 4565.52 | 4542.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 11:30:00 | 4486.00 | 4565.52 | 4542.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 12:15:00 | 4426.90 | 4537.80 | 4531.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-23 13:00:00 | 4426.90 | 4537.80 | 4531.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-23 13:15:00 | 4500.00 | 4530.24 | 4528.71 | EMA400 retest candle locked (from upside) |

### Cycle 62 — SELL (started 2026-03-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-23 14:15:00 | 4500.00 | 4524.19 | 4526.10 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2026-03-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 11:15:00 | 4539.10 | 4526.71 | 4526.01 | EMA200 above EMA400 |

### Cycle 64 — SELL (started 2026-03-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-24 12:15:00 | 4507.70 | 4522.91 | 4524.35 | EMA200 below EMA400 |

### Cycle 65 — BUY (started 2026-03-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 13:15:00 | 4594.00 | 4537.13 | 4530.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 4608.00 | 4555.97 | 4541.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 12:15:00 | 4537.20 | 4558.19 | 4546.72 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 12:15:00 | 4537.20 | 4558.19 | 4546.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 12:15:00 | 4537.20 | 4558.19 | 4546.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 12:45:00 | 4542.90 | 4558.19 | 4546.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 13:15:00 | 4533.00 | 4553.15 | 4545.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 14:00:00 | 4533.00 | 4553.15 | 4545.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 4505.10 | 4543.54 | 4541.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 15:00:00 | 4505.10 | 4543.54 | 4541.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 66 — SELL (started 2026-03-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-25 15:15:00 | 4494.80 | 4533.79 | 4537.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 09:15:00 | 4449.80 | 4516.99 | 4529.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-27 14:15:00 | 4512.20 | 4488.47 | 4507.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 14:15:00 | 4512.20 | 4488.47 | 4507.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 4512.20 | 4488.47 | 4507.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 4512.20 | 4488.47 | 4507.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 4460.00 | 4482.78 | 4502.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-30 09:15:00 | 4422.00 | 4482.78 | 4502.93 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-30 13:15:00 | 4565.00 | 4471.54 | 4485.59 | SL hit (close>static) qty=1.00 sl=4515.30 alert=retest2 |

### Cycle 67 — BUY (started 2026-03-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-30 14:15:00 | 4641.50 | 4505.54 | 4499.76 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 4695.40 | 4606.37 | 4556.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 09:15:00 | 4611.90 | 4661.13 | 4604.45 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-02 10:00:00 | 4611.90 | 4661.13 | 4604.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 4597.90 | 4648.48 | 4603.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 11:00:00 | 4597.90 | 4648.48 | 4603.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 11:15:00 | 4632.20 | 4645.23 | 4606.43 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:30:00 | 4638.70 | 4661.78 | 4617.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-07 15:15:00 | 4645.80 | 4655.40 | 4656.16 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2026-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 15:15:00 | 4645.80 | 4655.40 | 4656.16 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 4730.00 | 4670.32 | 4662.87 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 14:15:00 | 4818.00 | 4748.99 | 4708.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 14:15:00 | 4797.50 | 4798.74 | 4758.14 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 14:30:00 | 4820.00 | 4798.74 | 4758.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 4809.60 | 4802.07 | 4766.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 4865.60 | 4823.59 | 4798.38 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 14:15:00 | 4894.70 | 4931.67 | 4933.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 70 — SELL (started 2026-04-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 14:15:00 | 4894.70 | 4931.67 | 4933.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 09:15:00 | 4857.60 | 4912.91 | 4924.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 09:15:00 | 4880.00 | 4868.24 | 4890.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 09:15:00 | 4880.00 | 4868.24 | 4890.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 09:15:00 | 4880.00 | 4868.24 | 4890.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 09:30:00 | 4902.60 | 4868.24 | 4890.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 10:15:00 | 4860.50 | 4866.70 | 4887.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 10:30:00 | 4869.70 | 4866.70 | 4887.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-22 09:15:00 | 4805.30 | 4837.35 | 4863.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 10:45:00 | 4783.30 | 4827.70 | 4856.46 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 12:30:00 | 4795.80 | 4812.33 | 4844.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 13:45:00 | 4790.50 | 4807.04 | 4838.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-23 10:15:00 | 4792.10 | 4799.95 | 4827.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 4706.40 | 4692.75 | 4715.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:15:00 | 4728.20 | 4692.75 | 4715.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 4719.50 | 4698.10 | 4715.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:45:00 | 4701.00 | 4695.34 | 4713.04 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 12:45:00 | 4698.10 | 4694.63 | 4711.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 12:00:00 | 4684.00 | 4670.69 | 4687.34 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-30 10:15:00 | 4733.00 | 4695.17 | 4693.14 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2026-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 10:15:00 | 4733.00 | 4695.17 | 4693.14 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-30 11:15:00 | 4759.50 | 4708.03 | 4699.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 14:15:00 | 4729.50 | 4730.28 | 4713.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-30 15:00:00 | 4729.50 | 4730.28 | 4713.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 4798.00 | 4743.82 | 4720.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 4824.00 | 4743.82 | 4720.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 12:15:00 | 4700.50 | 4737.86 | 4726.45 | SL hit (close<static) qty=1.00 sl=4720.30 alert=retest2 |

### Cycle 72 — SELL (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-04 15:15:00 | 4699.00 | 4718.02 | 4719.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-05 09:15:00 | 4682.30 | 4710.88 | 4715.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-06 15:15:00 | 4652.00 | 4645.48 | 4663.10 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-05-07 09:15:00 | 4670.00 | 4645.48 | 4663.10 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 09:15:00 | 4673.00 | 4650.98 | 4664.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 11:15:00 | 4648.80 | 4651.94 | 4663.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 12:15:00 | 4648.00 | 4653.42 | 4662.90 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-07 14:00:00 | 4647.40 | 4650.47 | 4659.80 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-08 09:30:00 | 4634.00 | 4646.69 | 4655.70 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest1 | 2025-05-14 09:15:00 | 4812.00 | 2025-05-16 14:15:00 | 5052.60 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-14 09:15:00 | 4812.00 | 2025-05-22 09:15:00 | 5127.20 | STOP_HIT | 0.50 | 6.55% |
| BUY | retest2 | 2025-05-21 09:30:00 | 5168.00 | 2025-05-22 15:15:00 | 5047.50 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2025-05-21 13:30:00 | 5150.00 | 2025-05-22 15:15:00 | 5047.50 | STOP_HIT | 1.00 | -1.99% |
| BUY | retest2 | 2025-05-22 09:45:00 | 5126.90 | 2025-05-22 15:15:00 | 5047.50 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2025-06-10 13:30:00 | 5438.50 | 2025-06-16 10:15:00 | 5518.00 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-06-11 13:00:00 | 5469.00 | 2025-06-16 10:15:00 | 5518.00 | STOP_HIT | 1.00 | -0.90% |
| SELL | retest2 | 2025-06-13 11:00:00 | 5465.00 | 2025-06-16 10:15:00 | 5518.00 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2025-06-13 11:45:00 | 5460.00 | 2025-06-16 10:15:00 | 5518.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-06-13 15:00:00 | 5439.00 | 2025-06-16 10:15:00 | 5518.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-06-16 09:15:00 | 5439.50 | 2025-06-16 10:15:00 | 5518.00 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest1 | 2025-07-03 10:00:00 | 6429.00 | 2025-07-07 15:15:00 | 6380.50 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest1 | 2025-07-03 12:30:00 | 6432.00 | 2025-07-07 15:15:00 | 6380.50 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest1 | 2025-07-03 14:30:00 | 6430.50 | 2025-07-07 15:15:00 | 6380.50 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest1 | 2025-07-04 09:15:00 | 6436.50 | 2025-07-07 15:15:00 | 6380.50 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2025-07-08 13:30:00 | 6475.50 | 2025-07-10 09:15:00 | 6320.00 | STOP_HIT | 1.00 | -2.40% |
| BUY | retest2 | 2025-07-08 14:45:00 | 6474.50 | 2025-07-10 09:15:00 | 6320.00 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2025-07-09 09:30:00 | 6475.00 | 2025-07-10 09:15:00 | 6320.00 | STOP_HIT | 1.00 | -2.39% |
| BUY | retest2 | 2025-07-09 10:45:00 | 6483.50 | 2025-07-10 09:15:00 | 6320.00 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-07-22 10:15:00 | 6270.00 | 2025-07-23 10:15:00 | 6396.00 | STOP_HIT | 1.00 | -2.01% |
| SELL | retest2 | 2025-07-22 13:45:00 | 6268.00 | 2025-07-23 10:15:00 | 6396.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest2 | 2025-07-23 09:15:00 | 6257.50 | 2025-07-23 10:15:00 | 6396.00 | STOP_HIT | 1.00 | -2.21% |
| SELL | retest2 | 2025-08-06 11:30:00 | 6200.00 | 2025-08-06 12:15:00 | 6333.00 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-08-13 13:45:00 | 5592.00 | 2025-08-18 09:15:00 | 5688.00 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-08-14 11:15:00 | 5592.00 | 2025-08-18 09:15:00 | 5688.00 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-08-18 11:45:00 | 5580.00 | 2025-08-21 09:15:00 | 5614.00 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-08-20 10:15:00 | 5582.50 | 2025-08-21 09:15:00 | 5614.00 | STOP_HIT | 1.00 | -0.56% |
| SELL | retest2 | 2025-08-20 11:15:00 | 5577.00 | 2025-08-21 09:15:00 | 5614.00 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-08-20 15:00:00 | 5567.50 | 2025-08-21 09:15:00 | 5614.00 | STOP_HIT | 1.00 | -0.84% |
| SELL | retest2 | 2025-08-25 14:00:00 | 5499.50 | 2025-08-29 10:15:00 | 5224.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 14:00:00 | 5499.50 | 2025-09-01 09:15:00 | 5243.00 | STOP_HIT | 0.50 | 4.66% |
| BUY | retest2 | 2025-09-12 09:15:00 | 5139.00 | 2025-09-12 12:15:00 | 5103.00 | STOP_HIT | 1.00 | -0.70% |
| BUY | retest2 | 2025-09-18 09:30:00 | 5214.50 | 2025-09-23 10:15:00 | 5135.50 | STOP_HIT | 1.00 | -1.52% |
| BUY | retest2 | 2025-09-19 09:30:00 | 5223.50 | 2025-09-23 10:15:00 | 5135.50 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2025-09-29 09:30:00 | 5005.00 | 2025-10-03 10:15:00 | 4964.50 | STOP_HIT | 1.00 | 0.81% |
| SELL | retest2 | 2025-09-29 10:15:00 | 5006.50 | 2025-10-03 10:15:00 | 4964.50 | STOP_HIT | 1.00 | 0.84% |
| BUY | retest2 | 2025-10-07 10:00:00 | 5007.30 | 2025-10-07 11:15:00 | 4977.00 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2025-10-09 14:30:00 | 5003.10 | 2025-10-13 10:15:00 | 4987.30 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-10-10 13:00:00 | 5009.20 | 2025-10-13 10:15:00 | 4987.30 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-10-23 11:45:00 | 4982.40 | 2025-10-27 09:15:00 | 5053.30 | STOP_HIT | 1.00 | -1.42% |
| SELL | retest2 | 2025-10-23 13:00:00 | 4999.90 | 2025-10-27 09:15:00 | 5053.30 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-10-23 14:00:00 | 4996.80 | 2025-10-27 09:15:00 | 5053.30 | STOP_HIT | 1.00 | -1.13% |
| SELL | retest2 | 2025-10-23 15:00:00 | 4999.70 | 2025-10-27 09:15:00 | 5053.30 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-10-24 11:15:00 | 5000.60 | 2025-10-27 09:15:00 | 5053.30 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-10-24 13:45:00 | 4999.90 | 2025-10-27 09:15:00 | 5053.30 | STOP_HIT | 1.00 | -1.07% |
| SELL | retest2 | 2025-10-30 13:45:00 | 4881.80 | 2025-11-07 10:15:00 | 4637.71 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 11:45:00 | 4879.50 | 2025-11-07 10:15:00 | 4635.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 12:15:00 | 4878.40 | 2025-11-07 10:15:00 | 4634.48 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-31 12:45:00 | 4878.10 | 2025-11-07 10:15:00 | 4634.19 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-10-30 13:45:00 | 4881.80 | 2025-11-10 14:15:00 | 4642.10 | STOP_HIT | 0.50 | 4.91% |
| SELL | retest2 | 2025-10-31 11:45:00 | 4879.50 | 2025-11-10 14:15:00 | 4642.10 | STOP_HIT | 0.50 | 4.87% |
| SELL | retest2 | 2025-10-31 12:15:00 | 4878.40 | 2025-11-10 14:15:00 | 4642.10 | STOP_HIT | 0.50 | 4.84% |
| SELL | retest2 | 2025-10-31 12:45:00 | 4878.10 | 2025-11-10 14:15:00 | 4642.10 | STOP_HIT | 0.50 | 4.84% |
| SELL | retest2 | 2025-11-14 09:15:00 | 4454.00 | 2025-11-17 12:15:00 | 4555.00 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2025-11-14 14:45:00 | 4478.40 | 2025-11-17 12:15:00 | 4555.00 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2025-11-20 12:15:00 | 4580.00 | 2025-11-21 15:15:00 | 4528.30 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-11-20 14:15:00 | 4577.00 | 2025-11-21 15:15:00 | 4528.30 | STOP_HIT | 1.00 | -1.06% |
| BUY | retest2 | 2025-11-21 13:45:00 | 4574.30 | 2025-11-21 15:15:00 | 4528.30 | STOP_HIT | 1.00 | -1.01% |
| BUY | retest2 | 2025-12-02 09:15:00 | 4605.00 | 2025-12-03 09:15:00 | 4571.00 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-12-02 13:45:00 | 4614.60 | 2025-12-03 09:15:00 | 4571.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-12-02 15:00:00 | 4607.40 | 2025-12-03 09:15:00 | 4571.00 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-12-04 13:45:00 | 4511.60 | 2025-12-18 15:15:00 | 4286.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-04 13:45:00 | 4511.60 | 2025-12-19 11:15:00 | 4315.00 | STOP_HIT | 0.50 | 4.36% |
| SELL | retest2 | 2026-01-08 09:15:00 | 4515.70 | 2026-01-20 14:15:00 | 4289.91 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 09:15:00 | 4515.70 | 2026-01-20 14:15:00 | 4398.00 | STOP_HIT | 0.50 | 2.61% |
| SELL | retest2 | 2026-01-08 11:00:00 | 4515.00 | 2026-01-20 14:15:00 | 4289.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:00:00 | 4515.00 | 2026-01-20 14:15:00 | 4398.00 | STOP_HIT | 0.50 | 2.59% |
| SELL | retest2 | 2026-01-08 11:45:00 | 4513.20 | 2026-01-20 14:15:00 | 4287.54 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-08 11:45:00 | 4513.20 | 2026-01-20 14:15:00 | 4398.00 | STOP_HIT | 0.50 | 2.55% |
| SELL | retest2 | 2026-01-29 10:45:00 | 4334.90 | 2026-01-30 12:15:00 | 4430.00 | STOP_HIT | 1.00 | -2.19% |
| SELL | retest2 | 2026-01-29 11:45:00 | 4337.90 | 2026-01-30 12:15:00 | 4430.00 | STOP_HIT | 1.00 | -2.12% |
| BUY | retest2 | 2026-02-04 09:30:00 | 4452.30 | 2026-02-12 09:15:00 | 4897.53 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-04 10:30:00 | 4445.10 | 2026-02-12 09:15:00 | 4889.61 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-04 13:00:00 | 4447.50 | 2026-02-12 09:15:00 | 4892.25 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-05 14:45:00 | 4484.00 | 2026-02-12 09:15:00 | 4932.40 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-02-06 13:00:00 | 4485.30 | 2026-02-12 09:15:00 | 4933.83 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2026-03-05 12:45:00 | 4500.80 | 2026-03-06 14:15:00 | 4600.00 | STOP_HIT | 1.00 | -2.20% |
| SELL | retest2 | 2026-03-16 09:15:00 | 4473.60 | 2026-03-18 12:15:00 | 4499.40 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-03-17 11:15:00 | 4446.80 | 2026-03-18 12:15:00 | 4499.40 | STOP_HIT | 1.00 | -1.18% |
| SELL | retest2 | 2026-03-17 15:15:00 | 4466.40 | 2026-03-18 12:15:00 | 4499.40 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-03-30 09:15:00 | 4422.00 | 2026-03-30 13:15:00 | 4565.00 | STOP_HIT | 1.00 | -3.23% |
| BUY | retest2 | 2026-04-02 12:30:00 | 4638.70 | 2026-04-07 15:15:00 | 4645.80 | STOP_HIT | 1.00 | 0.15% |
| BUY | retest2 | 2026-04-13 10:45:00 | 4865.60 | 2026-04-17 14:15:00 | 4894.70 | STOP_HIT | 1.00 | 0.60% |
| SELL | retest2 | 2026-04-22 10:45:00 | 4783.30 | 2026-04-30 10:15:00 | 4733.00 | STOP_HIT | 1.00 | 1.05% |
| SELL | retest2 | 2026-04-22 12:30:00 | 4795.80 | 2026-04-30 10:15:00 | 4733.00 | STOP_HIT | 1.00 | 1.31% |
| SELL | retest2 | 2026-04-22 13:45:00 | 4790.50 | 2026-04-30 10:15:00 | 4733.00 | STOP_HIT | 1.00 | 1.20% |
| SELL | retest2 | 2026-04-23 10:15:00 | 4792.10 | 2026-04-30 10:15:00 | 4733.00 | STOP_HIT | 1.00 | 1.23% |
| SELL | retest2 | 2026-04-28 11:45:00 | 4701.00 | 2026-04-30 10:15:00 | 4733.00 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2026-04-28 12:45:00 | 4698.10 | 2026-04-30 10:15:00 | 4733.00 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-04-29 12:00:00 | 4684.00 | 2026-04-30 10:15:00 | 4733.00 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2026-05-04 09:15:00 | 4824.00 | 2026-05-04 12:15:00 | 4700.50 | STOP_HIT | 1.00 | -2.56% |
