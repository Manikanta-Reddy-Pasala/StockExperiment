# Choice International Ltd. (CHOICEIN)

## Backtest Summary

- **Window:** 2024-04-08 09:15:00 → 2026-05-08 15:15:00 (3598 bars)
- **Last close:** 686.95
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
| ALERT2 | 4 |
| ALERT2_SKIP | 1 |
| ALERT3 | 19 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 9 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 9 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 10 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 8
- **Target hits / Stop hits / Partials:** 0 / 9 / 1
- **Avg / median % per leg:** -0.37% / -1.06%
- **Sum % (uncompounded):** -3.73%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.85% | -2.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.85% | -2.5% |
| SELL (all) | 7 | 2 | 28.6% | 0 | 6 | 1 | -0.17% | -1.2% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 7 | 2 | 28.6% | 0 | 6 | 1 | -0.17% | -1.2% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 10 | 2 | 20.0% | 0 | 9 | 1 | -0.37% | -3.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 726.70 | 790.97 | 791.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 15:15:00 | 721.80 | 783.51 | 787.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 775.15 | 773.80 | 781.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 10:45:00 | 776.50 | 773.80 | 781.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 782.30 | 773.91 | 781.06 | EMA400 retest candle locked (from downside) |

### Cycle 2 — BUY (started 2025-12-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 10:15:00 | 838.60 | 786.72 | 786.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-30 11:15:00 | 840.00 | 787.25 | 786.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-12 10:15:00 | 808.20 | 809.99 | 800.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-12 11:00:00 | 808.20 | 809.99 | 800.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 14:15:00 | 799.45 | 812.56 | 803.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-20 15:00:00 | 799.45 | 812.56 | 803.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 15:15:00 | 795.00 | 812.39 | 803.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 09:30:00 | 793.20 | 812.14 | 803.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 10:15:00 | 781.60 | 811.84 | 803.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-21 10:45:00 | 779.55 | 811.84 | 803.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 783.80 | 798.27 | 797.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 11:00:00 | 783.80 | 798.27 | 797.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2026-01-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 13:15:00 | 772.60 | 796.27 | 796.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 15:15:00 | 764.00 | 795.64 | 795.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-18 12:15:00 | 793.40 | 778.58 | 785.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-18 12:15:00 | 793.40 | 778.58 | 785.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 12:15:00 | 793.40 | 778.58 | 785.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:00:00 | 793.40 | 778.58 | 785.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-18 13:15:00 | 794.60 | 778.74 | 785.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-18 13:45:00 | 793.85 | 778.74 | 785.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-23 12:15:00 | 778.10 | 782.32 | 786.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-23 12:30:00 | 784.05 | 782.32 | 786.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 09:15:00 | 703.75 | 667.08 | 702.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-10 09:45:00 | 706.00 | 667.08 | 702.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-10 10:15:00 | 703.80 | 667.45 | 702.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 13:15:00 | 700.90 | 668.17 | 702.08 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 14:15:00 | 700.45 | 668.50 | 702.08 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-10 15:15:00 | 701.00 | 668.84 | 702.08 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 14:45:00 | 700.50 | 670.62 | 701.84 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 712.65 | 671.26 | 701.86 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 712.65 | 671.26 | 701.86 | SL hit (close>static) qty=1.00 sl=707.90 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-10-03 14:00:00 | 776.00 | 2025-12-04 10:15:00 | 770.30 | STOP_HIT | 1.00 | -0.73% |
| BUY | retest2 | 2025-12-04 09:45:00 | 776.15 | 2025-12-04 10:15:00 | 770.30 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2025-12-04 12:30:00 | 776.40 | 2025-12-05 11:15:00 | 768.20 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-04-10 13:15:00 | 700.90 | 2026-04-15 09:15:00 | 712.65 | STOP_HIT | 1.00 | -1.68% |
| SELL | retest2 | 2026-04-10 14:15:00 | 700.45 | 2026-04-15 09:15:00 | 712.65 | STOP_HIT | 1.00 | -1.74% |
| SELL | retest2 | 2026-04-10 15:15:00 | 701.00 | 2026-04-15 09:15:00 | 712.65 | STOP_HIT | 1.00 | -1.66% |
| SELL | retest2 | 2026-04-13 14:45:00 | 700.50 | 2026-04-15 09:15:00 | 712.65 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2026-04-24 15:00:00 | 699.35 | 2026-04-27 10:15:00 | 709.00 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2026-04-27 13:45:00 | 700.60 | 2026-04-29 13:15:00 | 665.57 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-04-27 13:45:00 | 700.60 | 2026-05-06 14:15:00 | 686.55 | STOP_HIT | 0.50 | 2.01% |
