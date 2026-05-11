# Choice International Ltd. (CHOICEIN)

## Backtest Summary

- **Window:** 2022-04-08 09:15:00 → 2026-05-08 15:15:00 (7047 bars)
- **Last close:** 686.95
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 5 |
| ALERT1 | 5 |
| ALERT2 | 5 |
| ALERT2_SKIP | 1 |
| ALERT3 | 27 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 18 |
| PARTIAL | 1 |
| TARGET_HIT | 0 |
| STOP_HIT | 18 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 19 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 2 / 17
- **Target hits / Stop hits / Partials:** 0 / 18 / 1
- **Avg / median % per leg:** -1.03% / -1.66%
- **Sum % (uncompounded):** -19.60%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.85% | -2.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 3 | 0 | 0.0% | 0 | 3 | 0 | -0.85% | -2.5% |
| SELL (all) | 16 | 2 | 12.5% | 0 | 15 | 1 | -1.07% | -17.1% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 16 | 2 | 12.5% | 0 | 15 | 1 | -1.07% | -17.1% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 19 | 2 | 10.5% | 0 | 18 | 1 | -1.03% | -19.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-02-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-12 13:15:00 | 492.55 | 518.78 | 518.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-14 09:15:00 | 491.85 | 516.90 | 517.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-19 11:15:00 | 490.60 | 490.34 | 499.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-19 12:00:00 | 490.60 | 490.34 | 499.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 14:15:00 | 502.50 | 490.48 | 499.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-19 15:00:00 | 502.50 | 490.48 | 499.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-19 15:15:00 | 501.00 | 490.59 | 499.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-20 09:15:00 | 505.85 | 490.59 | 499.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 505.10 | 491.46 | 499.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-21 09:45:00 | 505.05 | 491.46 | 499.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-24 09:15:00 | 504.65 | 492.09 | 499.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 11:30:00 | 499.85 | 493.00 | 499.62 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 13:15:00 | 499.90 | 493.09 | 499.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-25 15:15:00 | 499.80 | 493.25 | 499.64 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 11:00:00 | 499.65 | 493.40 | 499.63 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-26 11:15:00 | 498.40 | 493.45 | 499.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-03-26 13:00:00 | 495.95 | 493.48 | 499.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-28 14:15:00 | 503.45 | 493.31 | 499.04 | SL hit (close>static) qty=1.00 sl=500.65 alert=retest2 |

### Cycle 2 — BUY (started 2025-04-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-16 15:15:00 | 538.00 | 502.82 | 502.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-17 11:15:00 | 542.50 | 503.93 | 503.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 15:15:00 | 691.10 | 691.27 | 658.74 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-03 09:15:00 | 690.50 | 691.27 | 658.74 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 10:15:00 | 776.35 | 800.31 | 775.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 10:45:00 | 776.55 | 800.31 | 775.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 12:15:00 | 776.15 | 799.85 | 775.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:00:00 | 776.15 | 799.85 | 775.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 13:15:00 | 769.05 | 799.55 | 775.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 13:45:00 | 769.15 | 799.55 | 775.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 768.70 | 799.24 | 775.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:15:00 | 766.10 | 799.24 | 775.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 770.30 | 798.37 | 775.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-01 11:00:00 | 770.30 | 798.37 | 775.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 773.40 | 796.29 | 775.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-03 10:45:00 | 773.30 | 796.29 | 775.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 12:15:00 | 774.95 | 795.86 | 775.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-03 14:00:00 | 776.00 | 795.66 | 775.15 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-04 09:45:00 | 776.15 | 797.89 | 794.50 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 10:15:00 | 770.30 | 797.62 | 794.38 | SL hit (close<static) qty=1.00 sl=772.00 alert=retest2 |

### Cycle 3 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 726.70 | 790.97 | 791.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-10 15:15:00 | 721.80 | 783.51 | 787.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 10:15:00 | 775.15 | 773.80 | 781.25 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-18 10:45:00 | 776.50 | 773.80 | 781.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 782.30 | 773.91 | 781.06 | EMA400 retest candle locked (from downside) |

### Cycle 4 — BUY (started 2025-12-30 10:15:00)

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

### Cycle 5 — SELL (started 2026-01-30 13:15:00)

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
| SELL | retest2 | 2025-03-25 11:30:00 | 499.85 | 2025-03-28 14:15:00 | 503.45 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2025-03-25 13:15:00 | 499.90 | 2025-04-01 12:15:00 | 504.00 | STOP_HIT | 1.00 | -0.82% |
| SELL | retest2 | 2025-03-25 15:15:00 | 499.80 | 2025-04-01 13:15:00 | 509.00 | STOP_HIT | 1.00 | -1.84% |
| SELL | retest2 | 2025-03-26 11:00:00 | 499.65 | 2025-04-01 13:15:00 | 509.00 | STOP_HIT | 1.00 | -1.87% |
| SELL | retest2 | 2025-03-26 13:00:00 | 495.95 | 2025-04-01 13:15:00 | 509.00 | STOP_HIT | 1.00 | -2.63% |
| SELL | retest2 | 2025-04-01 10:15:00 | 496.50 | 2025-04-01 13:15:00 | 509.00 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-04-07 09:15:00 | 492.75 | 2025-04-08 12:15:00 | 501.55 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-04-08 09:45:00 | 495.30 | 2025-04-08 12:15:00 | 501.55 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-04-09 09:15:00 | 498.00 | 2025-04-11 09:15:00 | 510.05 | STOP_HIT | 1.00 | -2.42% |
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
