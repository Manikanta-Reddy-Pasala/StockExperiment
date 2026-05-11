# Blue Dart Express Ltd. (BLUEDART)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 5695.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 8 |
| ALERT1 | 5 |
| ALERT2 | 6 |
| ALERT2_SKIP | 2 |
| ALERT3 | 14 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 10 |
| PARTIAL | 5 |
| TARGET_HIT | 0 |
| STOP_HIT | 10 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 15 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 6 / 9
- **Target hits / Stop hits / Partials:** 0 / 10 / 5
- **Avg / median % per leg:** -0.39% / -1.12%
- **Sum % (uncompounded):** -5.78%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 15 | 6 | 40.0% | 0 | 10 | 5 | -0.39% | -5.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 15 | 6 | 40.0% | 0 | 10 | 5 | -0.39% | -5.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 15 | 6 | 40.0% | 0 | 10 | 5 | -0.39% | -5.8% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-06-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 15:15:00 | 6142.00 | 6499.81 | 6501.38 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-07-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 10:15:00 | 6845.00 | 6488.76 | 6488.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 6901.50 | 6575.34 | 6537.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-28 09:15:00 | 6666.50 | 6696.77 | 6619.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-28 09:15:00 | 6666.50 | 6696.77 | 6619.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 6666.50 | 6696.77 | 6619.17 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 5820.00 | 6551.66 | 6554.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-04 10:15:00 | 5794.00 | 6544.12 | 6551.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 10:15:00 | 5871.00 | 5846.71 | 6031.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-18 11:00:00 | 5871.00 | 5846.71 | 6031.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 5800.50 | 5838.52 | 6009.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-23 10:15:00 | 5771.00 | 5838.52 | 6009.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 10:15:00 | 5779.00 | 5833.07 | 6001.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 13:15:00 | 5782.50 | 5831.41 | 5997.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 13:45:00 | 5770.00 | 5830.76 | 5996.60 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 5952.50 | 5808.63 | 5966.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 09:30:00 | 6119.00 | 5808.63 | 5966.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 5914.00 | 5809.68 | 5965.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:30:00 | 5965.00 | 5809.68 | 5965.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 5948.50 | 5812.21 | 5965.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:00:00 | 5948.50 | 5812.21 | 5965.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 09:15:00 | 5482.45 | 5756.10 | 5908.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 09:15:00 | 5490.05 | 5756.10 | 5908.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 09:15:00 | 5493.38 | 5756.10 | 5908.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-10-09 11:15:00 | 5481.50 | 5750.69 | 5903.97 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 6136.00 | 5638.18 | 5781.97 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-29 09:15:00 | 6136.00 | 5638.18 | 5781.97 | SL hit (close>ema200) qty=0.50 sl=5638.18 alert=retest2 |

### Cycle 4 — BUY (started 2025-11-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 12:15:00 | 6378.00 | 5904.62 | 5903.66 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 5734.00 | 5927.99 | 5928.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-28 14:15:00 | 5720.00 | 5908.78 | 5918.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-30 15:15:00 | 5574.00 | 5553.83 | 5677.11 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-31 09:15:00 | 5534.00 | 5553.83 | 5677.11 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 09:15:00 | 5780.50 | 5554.19 | 5672.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-01 09:30:00 | 5872.00 | 5554.19 | 5672.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-01 10:15:00 | 5799.00 | 5556.62 | 5673.06 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-01 11:15:00 | 5708.00 | 5556.62 | 5673.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-08 10:15:00 | 5422.60 | 5560.43 | 5657.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-27 15:15:00 | 5565.00 | 5444.94 | 5556.33 | SL hit (close>ema200) qty=0.50 sl=5444.94 alert=retest2 |

### Cycle 6 — BUY (started 2026-02-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-13 11:15:00 | 5765.00 | 5608.26 | 5608.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-13 12:15:00 | 5793.50 | 5610.11 | 5609.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 15:15:00 | 5640.00 | 5654.26 | 5633.23 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 15:15:00 | 5640.00 | 5654.26 | 5633.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 5640.00 | 5654.26 | 5633.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-20 09:15:00 | 5631.50 | 5654.26 | 5633.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 09:15:00 | 5630.50 | 5654.03 | 5633.22 | EMA400 retest candle locked (from upside) |

### Cycle 7 — SELL (started 2026-03-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-05 09:15:00 | 5439.00 | 5617.81 | 5618.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-06 10:15:00 | 5420.00 | 5605.11 | 5612.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-10 12:15:00 | 5164.50 | 5143.11 | 5305.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-10 12:45:00 | 5169.40 | 5143.11 | 5305.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 5379.30 | 5150.76 | 5290.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 10:00:00 | 5379.30 | 5150.76 | 5290.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 5395.60 | 5153.20 | 5290.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 11:00:00 | 5395.60 | 5153.20 | 5290.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 10:15:00 | 5299.60 | 5235.26 | 5313.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 12:30:00 | 5294.80 | 5236.72 | 5313.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-24 13:15:00 | 5295.00 | 5236.72 | 5313.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-27 09:15:00 | 5443.80 | 5241.84 | 5314.69 | SL hit (close>static) qty=1.00 sl=5355.20 alert=retest2 |

### Cycle 8 — BUY (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-08 09:15:00 | 5722.00 | 5369.69 | 5368.10 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-09-23 10:15:00 | 5771.00 | 2025-10-09 09:15:00 | 5482.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 10:15:00 | 5779.00 | 2025-10-09 09:15:00 | 5490.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 13:15:00 | 5782.50 | 2025-10-09 09:15:00 | 5493.38 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-24 13:45:00 | 5770.00 | 2025-10-09 11:15:00 | 5481.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-23 10:15:00 | 5771.00 | 2025-10-29 09:15:00 | 6136.00 | STOP_HIT | 0.50 | -6.32% |
| SELL | retest2 | 2025-09-24 10:15:00 | 5779.00 | 2025-10-29 09:15:00 | 6136.00 | STOP_HIT | 0.50 | -6.18% |
| SELL | retest2 | 2025-09-24 13:15:00 | 5782.50 | 2025-10-29 09:15:00 | 6136.00 | STOP_HIT | 0.50 | -6.11% |
| SELL | retest2 | 2025-09-24 13:45:00 | 5770.00 | 2025-10-29 09:15:00 | 6136.00 | STOP_HIT | 0.50 | -6.34% |
| SELL | retest2 | 2026-01-01 11:15:00 | 5708.00 | 2026-01-08 10:15:00 | 5422.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-01 11:15:00 | 5708.00 | 2026-01-27 15:15:00 | 5565.00 | STOP_HIT | 0.50 | 2.51% |
| SELL | retest2 | 2026-02-09 11:45:00 | 5760.00 | 2026-02-09 13:15:00 | 5835.50 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2026-02-09 12:30:00 | 5771.00 | 2026-02-09 13:15:00 | 5835.50 | STOP_HIT | 1.00 | -1.12% |
| SELL | retest2 | 2026-02-13 09:15:00 | 5749.00 | 2026-02-13 11:15:00 | 5765.00 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2026-04-24 12:30:00 | 5294.80 | 2026-04-27 09:15:00 | 5443.80 | STOP_HIT | 1.00 | -2.81% |
| SELL | retest2 | 2026-04-24 13:15:00 | 5295.00 | 2026-04-27 09:15:00 | 5443.80 | STOP_HIT | 1.00 | -2.81% |
