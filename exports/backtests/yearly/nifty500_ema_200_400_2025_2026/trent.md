# Trent Ltd. (TRENT)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 4249.10
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 3 |
| ALERT1 | 3 |
| ALERT2 | 2 |
| ALERT2_SKIP | 1 |
| ALERT3 | 14 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 13 |
| PARTIAL | 1 |
| TARGET_HIT | 1 |
| STOP_HIT | 14 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 14 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 3 / 11
- **Target hits / Stop hits / Partials:** 1 / 12 / 1
- **Avg / median % per leg:** -1.23% / -3.09%
- **Sum % (uncompounded):** -17.22%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL (all) | 14 | 3 | 21.4% | 1 | 12 | 1 | -1.23% | -17.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 14 | 3 | 21.4% | 1 | 12 | 1 | -1.23% | -17.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 14 | 3 | 21.4% | 1 | 12 | 1 | -1.23% | -17.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 09:15:00 | 5661.00 | 5388.31 | 5387.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 11:15:00 | 5796.00 | 5459.52 | 5426.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-13 09:15:00 | 5541.50 | 5551.36 | 5481.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-13 09:15:00 | 5541.50 | 5551.36 | 5481.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-13 09:15:00 | 5541.50 | 5551.36 | 5481.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-13 09:45:00 | 5533.00 | 5551.36 | 5481.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 09:15:00 | 5758.00 | 5843.24 | 5682.85 | EMA400 retest candle locked (from upside) |

### Cycle 2 — SELL (started 2025-07-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-22 14:15:00 | 5363.50 | 5581.54 | 5582.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-23 09:15:00 | 5345.00 | 5576.95 | 5579.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-06 09:15:00 | 5400.50 | 5362.00 | 5451.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-06 10:00:00 | 5400.50 | 5362.00 | 5451.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 10:15:00 | 5436.00 | 5358.15 | 5440.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 10:45:00 | 5440.00 | 5358.15 | 5440.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 11:15:00 | 5415.00 | 5358.72 | 5440.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 10:15:00 | 5396.50 | 5362.62 | 5440.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 14:00:00 | 5403.50 | 5364.21 | 5439.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 15:00:00 | 5395.50 | 5367.21 | 5438.17 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-14 10:30:00 | 5395.00 | 5368.46 | 5437.74 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 5563.50 | 5371.03 | 5437.00 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-08-18 09:15:00 | 5563.50 | 5371.03 | 5437.00 | SL hit (close>static) qty=1.00 sl=5445.00 alert=retest2 |

### Cycle 3 — BUY (started 2026-04-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-23 12:15:00 | 4330.60 | 3908.61 | 3907.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 13:15:00 | 4362.00 | 3913.12 | 3909.38 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-20 12:45:00 | 5465.00 | 2025-05-23 09:15:00 | 5423.50 | STOP_HIT | 1.00 | 0.76% |
| SELL | retest2 | 2025-05-20 14:00:00 | 5457.00 | 2025-05-26 14:15:00 | 5528.50 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-05-22 14:15:00 | 5301.50 | 2025-05-26 14:15:00 | 5528.50 | STOP_HIT | 1.00 | -4.28% |
| SELL | retest2 | 2025-08-12 10:15:00 | 5396.50 | 2025-08-18 09:15:00 | 5563.50 | STOP_HIT | 1.00 | -3.09% |
| SELL | retest2 | 2025-08-12 14:00:00 | 5403.50 | 2025-08-18 09:15:00 | 5563.50 | STOP_HIT | 1.00 | -2.96% |
| SELL | retest2 | 2025-08-13 15:00:00 | 5395.50 | 2025-08-18 09:15:00 | 5563.50 | STOP_HIT | 1.00 | -3.11% |
| SELL | retest2 | 2025-08-14 10:30:00 | 5395.00 | 2025-08-18 09:15:00 | 5563.50 | STOP_HIT | 1.00 | -3.12% |
| SELL | retest2 | 2025-08-22 12:45:00 | 5445.50 | 2025-09-03 14:15:00 | 5481.50 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-08-25 14:30:00 | 5444.50 | 2025-09-04 09:15:00 | 5636.00 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2025-09-01 15:00:00 | 5444.50 | 2025-09-04 09:15:00 | 5636.00 | STOP_HIT | 1.00 | -3.52% |
| SELL | retest2 | 2025-09-02 10:00:00 | 5445.50 | 2025-09-04 09:15:00 | 5636.00 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2025-09-02 13:15:00 | 5424.50 | 2025-09-04 09:15:00 | 5636.00 | STOP_HIT | 1.00 | -3.90% |
| SELL | retest2 | 2025-09-08 13:30:00 | 5404.00 | 2025-09-12 12:15:00 | 5133.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-08 13:30:00 | 5404.00 | 2025-09-23 10:15:00 | 4863.60 | TARGET_HIT | 0.50 | 10.00% |
