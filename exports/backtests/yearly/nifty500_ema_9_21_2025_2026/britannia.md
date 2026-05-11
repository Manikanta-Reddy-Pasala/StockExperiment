# Britannia Industries Ltd. (BRITANNIA)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1528 bars)
- **Last close:** 5516.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 63 |
| ALERT1 | 40 |
| ALERT2 | 39 |
| ALERT2_SKIP | 21 |
| ALERT3 | 105 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 58 |
| PARTIAL | 0 |
| TARGET_HIT | 0 |
| STOP_HIT | 61 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 61 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 10 / 51
- **Target hits / Stop hits / Partials:** 0 / 61 / 0
- **Avg / median % per leg:** -1.01% / -0.80%
- **Sum % (uncompounded):** -61.83%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 33 | 6 | 18.2% | 0 | 33 | 0 | -1.36% | -44.8% |
| BUY @ 2nd Alert (retest1) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.62% | -1.9% |
| BUY @ 3rd Alert (retest2) | 30 | 6 | 20.0% | 0 | 30 | 0 | -1.43% | -43.0% |
| SELL (all) | 28 | 4 | 14.3% | 0 | 28 | 0 | -0.61% | -17.0% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 28 | 4 | 14.3% | 0 | 28 | 0 | -0.61% | -17.0% |
| retest1 (combined) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.62% | -1.9% |
| retest2 (combined) | 58 | 10 | 17.2% | 0 | 58 | 0 | -1.03% | -60.0% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 13:15:00 | 5446.00 | 5483.30 | 5485.48 | EMA200 below EMA400 |

### Cycle 2 — BUY (started 2025-05-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-15 10:15:00 | 5494.50 | 5487.91 | 5487.11 | EMA200 above EMA400 |

### Cycle 3 — SELL (started 2025-05-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-15 11:15:00 | 5475.00 | 5485.33 | 5486.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-15 13:15:00 | 5468.00 | 5481.97 | 5484.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-16 10:15:00 | 5495.50 | 5480.00 | 5482.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-16 10:15:00 | 5495.50 | 5480.00 | 5482.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 10:15:00 | 5495.50 | 5480.00 | 5482.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-16 10:45:00 | 5513.00 | 5480.00 | 5482.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2025-05-16 11:15:00)

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

### Cycle 5 — SELL (started 2025-05-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 13:15:00 | 5460.50 | 5494.34 | 5495.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 5429.00 | 5481.27 | 5489.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 09:15:00 | 5497.00 | 5474.62 | 5484.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 09:15:00 | 5497.00 | 5474.62 | 5484.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 09:15:00 | 5497.00 | 5474.62 | 5484.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 10:00:00 | 5497.00 | 5474.62 | 5484.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 10:15:00 | 5493.50 | 5478.39 | 5485.32 | EMA400 retest candle locked (from downside) |

### Cycle 6 — BUY (started 2025-05-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-21 13:15:00 | 5503.00 | 5490.74 | 5489.73 | EMA200 above EMA400 |

### Cycle 7 — SELL (started 2025-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 09:15:00 | 5464.00 | 5484.41 | 5487.05 | EMA200 below EMA400 |

### Cycle 8 — BUY (started 2025-05-23 15:15:00)

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

### Cycle 9 — SELL (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 09:15:00 | 5446.00 | 5492.50 | 5496.46 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-05-30 09:15:00)

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

### Cycle 11 — SELL (started 2025-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 12:15:00 | 5545.00 | 5559.25 | 5560.03 | EMA200 below EMA400 |

### Cycle 12 — BUY (started 2025-06-05 12:15:00)

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

### Cycle 13 — SELL (started 2025-06-12 11:15:00)

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

### Cycle 14 — BUY (started 2025-06-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-18 11:15:00 | 5568.50 | 5566.82 | 5566.71 | EMA200 above EMA400 |

### Cycle 15 — SELL (started 2025-06-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-18 13:15:00 | 5560.50 | 5565.99 | 5566.37 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2025-06-18 14:15:00)

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

### Cycle 17 — SELL (started 2025-06-19 15:15:00)

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

### Cycle 18 — BUY (started 2025-06-23 15:15:00)

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

### Cycle 19 — SELL (started 2025-07-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 14:15:00 | 5741.00 | 5771.51 | 5774.24 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-07-02 10:15:00)

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

### Cycle 21 — SELL (started 2025-07-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 12:15:00 | 5769.50 | 5801.75 | 5802.69 | EMA200 below EMA400 |

### Cycle 22 — BUY (started 2025-07-07 11:15:00)

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

### Cycle 23 — SELL (started 2025-07-11 12:15:00)

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

### Cycle 24 — BUY (started 2025-07-17 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-17 10:15:00 | 5829.50 | 5797.76 | 5795.33 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2025-07-18 10:15:00)

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

### Cycle 26 — BUY (started 2025-07-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 09:15:00 | 5684.00 | 5634.42 | 5633.75 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 10:15:00 | 5707.00 | 5648.93 | 5640.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 5757.50 | 5766.25 | 5726.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 15:00:00 | 5757.50 | 5766.25 | 5726.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 5771.50 | 5803.49 | 5776.95 | EMA400 retest candle locked (from upside) |

### Cycle 27 — SELL (started 2025-08-05 09:15:00)

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

### Cycle 28 — BUY (started 2025-08-18 09:15:00)

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

### Cycle 29 — SELL (started 2025-08-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 14:15:00 | 5544.50 | 5570.93 | 5571.83 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2025-08-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-25 09:15:00 | 5594.50 | 5575.18 | 5573.57 | EMA200 above EMA400 |

### Cycle 31 — SELL (started 2025-08-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-25 14:15:00 | 5549.50 | 5573.99 | 5574.73 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2025-08-26 09:15:00)

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

### Cycle 33 — SELL (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 12:15:00 | 6214.00 | 6220.31 | 6221.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 13:15:00 | 6199.00 | 6216.05 | 6219.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-16 15:15:00 | 6223.00 | 6215.51 | 6218.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-16 15:15:00 | 6223.00 | 6215.51 | 6218.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-16 15:15:00 | 6223.00 | 6215.51 | 6218.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 6106.00 | 6215.51 | 6218.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-30 11:15:00 | 6009.00 | 5948.08 | 5947.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2025-09-30 11:15:00)

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

### Cycle 35 — SELL (started 2025-10-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-01 11:15:00 | 5931.50 | 5953.23 | 5955.01 | EMA200 below EMA400 |

### Cycle 36 — BUY (started 2025-10-01 15:15:00)

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

### Cycle 37 — SELL (started 2025-10-07 11:15:00)

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

### Cycle 38 — BUY (started 2025-10-16 09:15:00)

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

### Cycle 39 — SELL (started 2025-10-24 10:15:00)

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

### Cycle 40 — BUY (started 2025-11-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-04 13:15:00 | 5873.00 | 5849.25 | 5847.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 09:15:00 | 6076.00 | 5911.46 | 5877.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-10 13:15:00 | 6155.00 | 6158.81 | 6091.89 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-10 13:45:00 | 6162.00 | 6158.81 | 6091.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 5815.00 | 6083.19 | 6073.52 | EMA400 retest candle locked (from upside) |

### Cycle 41 — SELL (started 2025-11-11 10:15:00)

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

### Cycle 42 — BUY (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 11:15:00 | 5872.50 | 5844.40 | 5843.11 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-20 09:15:00 | 5885.00 | 5861.44 | 5852.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-20 11:15:00 | 5808.00 | 5853.00 | 5850.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-20 11:15:00 | 5808.00 | 5853.00 | 5850.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-20 11:15:00 | 5808.00 | 5853.00 | 5850.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-20 12:00:00 | 5808.00 | 5853.00 | 5850.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2025-11-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 12:15:00 | 5773.50 | 5837.10 | 5843.45 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2025-11-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-25 13:15:00 | 5853.50 | 5821.88 | 5819.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 14:15:00 | 5870.50 | 5831.60 | 5824.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-26 15:15:00 | 5868.00 | 5869.36 | 5852.28 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-11-27 09:15:00 | 5894.00 | 5869.36 | 5852.28 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 5844.00 | 5865.99 | 5855.15 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-11-27 11:15:00 | 5844.00 | 5865.99 | 5855.15 | SL hit (close<ema400) qty=1.00 sl=5855.15 alert=retest1 |

### Cycle 45 — SELL (started 2025-11-27 15:15:00)

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

### Cycle 46 — BUY (started 2025-12-02 14:15:00)

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

### Cycle 47 — SELL (started 2025-12-08 14:15:00)

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

### Cycle 48 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 5955.00 | 5885.70 | 5877.79 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2025-12-10 11:15:00)

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

### Cycle 50 — BUY (started 2025-12-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-12 13:15:00 | 5900.00 | 5863.80 | 5859.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-12 14:15:00 | 5914.00 | 5873.84 | 5864.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 15:15:00 | 6050.00 | 6056.06 | 6009.07 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-17 09:15:00 | 6085.50 | 6056.06 | 6009.07 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-12-18 10:00:00 | 6073.00 | 6083.21 | 6052.88 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-18 10:15:00 | 6048.50 | 6076.26 | 6052.48 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-18 10:15:00 | 6048.50 | 6076.26 | 6052.48 | SL hit (close<ema400) qty=1.00 sl=6052.48 alert=retest1 |

### Cycle 51 — SELL (started 2026-03-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 09:15:00 | 5610.50 | 5988.03 | 6024.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 13:15:00 | 5534.00 | 5736.88 | 5879.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 5490.00 | 5487.16 | 5621.32 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-04-01 09:30:00 | 5490.00 | 5487.16 | 5621.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 11:15:00 | 5437.50 | 5443.60 | 5481.09 | EMA400 retest candle locked (from downside) |

### Cycle 52 — BUY (started 2026-04-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-07 12:15:00 | 5500.50 | 5494.26 | 5494.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 14:15:00 | 5538.50 | 5504.43 | 5498.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 11:15:00 | 5569.00 | 5586.66 | 5562.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 12:15:00 | 5453.00 | 5559.93 | 5552.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 12:15:00 | 5453.00 | 5559.93 | 5552.19 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2026-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 13:15:00 | 5479.00 | 5543.74 | 5545.53 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2026-04-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 15:15:00 | 5565.50 | 5530.10 | 5529.94 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2026-04-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 09:15:00 | 5509.00 | 5525.88 | 5528.04 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2026-04-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-13 10:15:00 | 5546.00 | 5529.91 | 5529.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-13 11:15:00 | 5564.00 | 5536.72 | 5532.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 09:15:00 | 5590.50 | 5623.58 | 5596.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 09:15:00 | 5590.50 | 5623.58 | 5596.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 5590.50 | 5623.58 | 5596.58 | EMA400 retest candle locked (from upside) |

### Cycle 57 — SELL (started 2026-04-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 10:15:00 | 5673.50 | 5731.19 | 5731.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 15:15:00 | 5656.50 | 5692.92 | 5710.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-24 11:15:00 | 5730.50 | 5696.65 | 5707.48 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-24 11:15:00 | 5730.50 | 5696.65 | 5707.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-24 11:15:00 | 5730.50 | 5696.65 | 5707.48 | EMA400 retest candle locked (from downside) |

### Cycle 58 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 5750.50 | 5717.79 | 5714.16 | EMA200 above EMA400 |

### Cycle 59 — SELL (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-28 10:15:00 | 5657.50 | 5704.80 | 5710.58 | EMA200 below EMA400 |

### Cycle 60 — BUY (started 2026-04-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 14:15:00 | 5708.00 | 5699.40 | 5698.96 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 5676.50 | 5695.88 | 5697.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 5657.50 | 5688.20 | 5693.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 11:15:00 | 5692.00 | 5688.96 | 5693.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 11:15:00 | 5692.00 | 5688.96 | 5693.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 5692.00 | 5688.96 | 5693.69 | EMA400 retest candle locked (from downside) |

### Cycle 62 — BUY (started 2026-04-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-30 13:15:00 | 5730.00 | 5699.50 | 5697.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-04 09:15:00 | 5765.00 | 5720.90 | 5708.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-04 11:15:00 | 5738.00 | 5739.86 | 5720.42 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 09:15:00 | 5787.50 | 5773.30 | 5746.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 09:15:00 | 5787.50 | 5773.30 | 5746.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 09:15:00 | 5852.50 | 5804.26 | 5774.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-06 14:30:00 | 5816.00 | 5795.16 | 5782.37 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 11:30:00 | 5819.00 | 5795.71 | 5786.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-07 15:15:00 | 5885.50 | 5800.34 | 5791.07 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-07 15:15:00 | 5885.50 | 5817.37 | 5799.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 09:15:00 | 5537.00 | 5817.37 | 5799.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-05-08 09:15:00 | 5615.50 | 5777.00 | 5782.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2026-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 09:15:00 | 5615.50 | 5777.00 | 5782.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-05-08 14:15:00 | 5523.00 | 5612.08 | 5688.41 | Break + close below crossover candle low |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
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
| BUY | retest2 | 2025-12-18 15:00:00 | 6043.00 | 2026-03-27 09:15:00 | 5610.50 | STOP_HIT | 1.00 | -7.16% |
| BUY | retest2 | 2025-12-19 10:00:00 | 6050.50 | 2026-03-27 09:15:00 | 5610.50 | STOP_HIT | 1.00 | -7.27% |
| BUY | retest2 | 2025-12-23 13:45:00 | 6042.50 | 2026-03-27 09:15:00 | 5610.50 | STOP_HIT | 1.00 | -7.15% |
| BUY | retest2 | 2025-12-23 14:45:00 | 6053.00 | 2026-03-27 09:15:00 | 5610.50 | STOP_HIT | 1.00 | -7.31% |
| BUY | retest2 | 2026-05-06 09:15:00 | 5852.50 | 2026-05-08 09:15:00 | 5615.50 | STOP_HIT | 1.00 | -4.05% |
| BUY | retest2 | 2026-05-06 14:30:00 | 5816.00 | 2026-05-08 09:15:00 | 5615.50 | STOP_HIT | 1.00 | -3.45% |
| BUY | retest2 | 2026-05-07 11:30:00 | 5819.00 | 2026-05-08 09:15:00 | 5615.50 | STOP_HIT | 1.00 | -3.50% |
| BUY | retest2 | 2026-05-07 15:15:00 | 5885.50 | 2026-05-08 09:15:00 | 5615.50 | STOP_HIT | 1.00 | -4.59% |
