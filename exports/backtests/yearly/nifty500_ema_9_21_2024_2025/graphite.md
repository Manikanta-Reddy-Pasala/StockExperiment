# Graphite India Ltd. (GRAPHITE)

## Backtest Summary

- **Window:** 2024-03-12 15:15:00 → 2026-05-08 15:15:00 (3711 bars)
- **Last close:** 752.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 147 |
| ALERT1 | 101 |
| ALERT2 | 100 |
| ALERT2_SKIP | 51 |
| ALERT3 | 247 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 3 |
| ENTRY2 | 126 |
| PARTIAL | 14 |
| TARGET_HIT | 15 |
| STOP_HIT | 114 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 143 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 47 / 96
- **Target hits / Stop hits / Partials:** 15 / 114 / 14
- **Avg / median % per leg:** 0.89% / -0.91%
- **Sum % (uncompounded):** 126.99%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 70 | 19 | 27.1% | 15 | 55 | 0 | 0.91% | 63.7% |
| BUY @ 2nd Alert (retest1) | 3 | 2 | 66.7% | 2 | 1 | 0 | 6.62% | 19.9% |
| BUY @ 3rd Alert (retest2) | 67 | 17 | 25.4% | 13 | 54 | 0 | 0.65% | 43.8% |
| SELL (all) | 73 | 28 | 38.4% | 0 | 59 | 14 | 0.87% | 63.3% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 73 | 28 | 38.4% | 0 | 59 | 14 | 0.87% | 63.3% |
| retest1 (combined) | 3 | 2 | 66.7% | 2 | 1 | 0 | 6.62% | 19.9% |
| retest2 (combined) | 140 | 45 | 32.1% | 13 | 113 | 14 | 0.77% | 107.1% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 13:15:00 | 588.75 | 578.34 | 577.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-14 14:15:00 | 598.05 | 582.28 | 579.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-15 12:15:00 | 585.50 | 586.46 | 582.86 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-15 13:00:00 | 585.50 | 586.46 | 582.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 14:15:00 | 582.25 | 585.45 | 583.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-15 15:00:00 | 582.25 | 585.45 | 583.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-15 15:15:00 | 579.00 | 584.16 | 582.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-16 09:15:00 | 597.10 | 584.16 | 582.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-23 10:15:00 | 594.50 | 604.37 | 604.83 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-23 10:15:00 | 594.50 | 604.37 | 604.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-23 14:15:00 | 590.10 | 597.25 | 600.92 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-24 11:15:00 | 595.35 | 594.86 | 598.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 11:15:00 | 595.35 | 594.86 | 598.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 11:15:00 | 595.35 | 594.86 | 598.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-24 12:00:00 | 595.35 | 594.86 | 598.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-27 09:15:00 | 582.00 | 590.17 | 594.65 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-28 12:00:00 | 573.40 | 580.44 | 585.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-30 09:30:00 | 575.70 | 579.50 | 581.07 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-03 13:15:00 | 581.30 | 574.23 | 573.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 3 — BUY (started 2024-06-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-03 13:15:00 | 581.30 | 574.23 | 573.66 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2024-06-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-04 09:15:00 | 564.50 | 572.96 | 573.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-04 10:15:00 | 534.15 | 565.20 | 569.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-05 12:15:00 | 539.95 | 539.43 | 549.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-05 12:45:00 | 539.25 | 539.43 | 549.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 561.35 | 544.17 | 548.78 | EMA400 retest candle locked (from downside) |

### Cycle 5 — BUY (started 2024-06-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 12:15:00 | 560.20 | 551.99 | 551.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-06 13:15:00 | 564.75 | 554.54 | 552.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-10 15:15:00 | 567.20 | 568.59 | 565.22 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-06-11 09:15:00 | 570.85 | 568.59 | 565.22 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-12 14:15:00 | 570.05 | 575.84 | 573.68 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-12 14:15:00 | 570.05 | 575.84 | 573.68 | SL hit (close<ema400) qty=1.00 sl=573.68 alert=retest1 |

### Cycle 6 — SELL (started 2024-06-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-14 15:15:00 | 571.65 | 574.43 | 574.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-18 09:15:00 | 569.80 | 573.50 | 574.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-18 14:15:00 | 572.70 | 571.66 | 572.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-18 14:15:00 | 572.70 | 571.66 | 572.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 14:15:00 | 572.70 | 571.66 | 572.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-18 14:30:00 | 574.95 | 571.66 | 572.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 15:15:00 | 572.00 | 571.73 | 572.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-19 09:15:00 | 580.20 | 571.73 | 572.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-19 09:15:00 | 579.50 | 573.28 | 573.32 | EMA400 retest candle locked (from downside) |

### Cycle 7 — BUY (started 2024-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 10:15:00 | 580.00 | 574.63 | 573.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-19 12:15:00 | 589.40 | 578.49 | 575.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-20 14:15:00 | 591.65 | 592.11 | 586.24 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-20 15:00:00 | 591.65 | 592.11 | 586.24 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 15:15:00 | 588.80 | 591.45 | 586.47 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 09:15:00 | 595.55 | 591.45 | 586.47 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 10:15:00 | 592.60 | 591.36 | 586.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 12:15:00 | 601.00 | 591.05 | 587.51 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-24 10:30:00 | 591.75 | 589.27 | 588.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 11:15:00 | 589.35 | 589.28 | 588.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 11:45:00 | 589.40 | 589.28 | 588.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 589.75 | 589.38 | 588.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 12:45:00 | 588.20 | 589.38 | 588.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 13:15:00 | 587.85 | 589.07 | 588.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 14:00:00 | 587.85 | 589.07 | 588.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 14:15:00 | 585.80 | 588.42 | 588.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-24 15:00:00 | 585.80 | 588.42 | 588.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-24 15:15:00 | 583.00 | 587.33 | 587.61 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 8 — SELL (started 2024-06-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 15:15:00 | 583.00 | 587.33 | 587.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 12:15:00 | 581.30 | 583.65 | 585.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 09:15:00 | 566.05 | 565.30 | 570.27 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-01 09:45:00 | 568.00 | 565.30 | 570.27 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 11:15:00 | 568.00 | 566.10 | 569.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 11:45:00 | 567.50 | 566.10 | 569.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 12:15:00 | 574.55 | 567.79 | 570.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 12:30:00 | 576.75 | 567.79 | 570.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 13:15:00 | 573.10 | 568.85 | 570.48 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-01 14:15:00 | 571.25 | 568.85 | 570.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-02 09:15:00 | 577.85 | 571.88 | 571.57 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2024-07-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-02 09:15:00 | 577.85 | 571.88 | 571.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-04 09:15:00 | 582.50 | 578.47 | 576.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-04 13:15:00 | 579.50 | 579.59 | 577.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-04 13:45:00 | 579.95 | 579.59 | 577.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 14:15:00 | 578.65 | 579.41 | 577.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-04 14:45:00 | 578.50 | 579.41 | 577.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 15:15:00 | 577.10 | 578.94 | 577.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:15:00 | 579.45 | 578.94 | 577.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 09:15:00 | 575.70 | 578.30 | 577.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 09:45:00 | 575.90 | 578.30 | 577.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-05 10:15:00 | 574.50 | 577.54 | 577.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-05 11:15:00 | 574.20 | 577.54 | 577.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 11:15:00 | 574.50 | 576.93 | 577.10 | EMA200 below EMA400 |

### Cycle 11 — BUY (started 2024-07-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 12:15:00 | 578.50 | 577.24 | 577.23 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2024-07-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-05 13:15:00 | 573.75 | 576.54 | 576.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-05 14:15:00 | 572.35 | 575.71 | 576.50 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-09 09:15:00 | 565.00 | 564.72 | 568.93 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-09 09:30:00 | 565.00 | 564.72 | 568.93 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 09:15:00 | 558.85 | 561.67 | 565.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 10:15:00 | 556.20 | 561.67 | 565.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-07-10 11:00:00 | 555.60 | 560.46 | 564.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-12 09:15:00 | 565.70 | 563.48 | 563.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 13 — BUY (started 2024-07-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-12 09:15:00 | 565.70 | 563.48 | 563.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-12 11:15:00 | 571.70 | 565.37 | 564.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-12 13:15:00 | 564.00 | 565.25 | 564.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-12 13:15:00 | 564.00 | 565.25 | 564.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 13:15:00 | 564.00 | 565.25 | 564.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 14:00:00 | 564.00 | 565.25 | 564.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 14:15:00 | 562.85 | 564.77 | 564.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 15:15:00 | 561.00 | 564.77 | 564.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 14 — SELL (started 2024-07-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 09:15:00 | 556.90 | 562.59 | 563.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-18 10:15:00 | 553.95 | 556.78 | 558.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 523.70 | 518.44 | 524.93 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 523.70 | 518.44 | 524.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 523.70 | 518.44 | 524.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 09:45:00 | 525.95 | 518.44 | 524.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 518.00 | 518.35 | 524.30 | EMA400 retest candle locked (from downside) |

### Cycle 15 — BUY (started 2024-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 12:15:00 | 532.00 | 524.47 | 524.45 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-07-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-26 09:15:00 | 523.00 | 524.44 | 524.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-26 10:15:00 | 521.00 | 523.75 | 524.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-26 12:15:00 | 523.30 | 523.16 | 523.81 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-26 12:15:00 | 523.30 | 523.16 | 523.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-26 12:15:00 | 523.30 | 523.16 | 523.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-26 12:30:00 | 523.00 | 523.16 | 523.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 09:15:00 | 528.00 | 523.52 | 523.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-29 10:15:00 | 530.40 | 523.52 | 523.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 17 — BUY (started 2024-07-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-29 10:15:00 | 528.00 | 524.42 | 524.10 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-07-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-29 13:15:00 | 522.00 | 523.67 | 523.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-29 15:15:00 | 520.90 | 522.82 | 523.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-30 09:15:00 | 524.60 | 523.18 | 523.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 09:15:00 | 524.60 | 523.18 | 523.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 09:15:00 | 524.60 | 523.18 | 523.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-30 09:30:00 | 523.95 | 523.18 | 523.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 19 — BUY (started 2024-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-30 10:15:00 | 541.00 | 526.74 | 525.09 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-07-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-30 15:15:00 | 519.75 | 523.72 | 524.21 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 09:15:00 | 528.80 | 524.74 | 524.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 12:15:00 | 530.75 | 525.90 | 525.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 11:15:00 | 532.80 | 532.90 | 529.48 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-01 12:00:00 | 532.80 | 532.90 | 529.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 14:15:00 | 530.75 | 532.57 | 530.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-01 14:30:00 | 529.95 | 532.57 | 530.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 15:15:00 | 528.40 | 531.74 | 530.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-02 09:15:00 | 524.00 | 531.74 | 530.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2024-08-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-02 09:15:00 | 510.90 | 527.57 | 528.30 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-05 09:15:00 | 505.45 | 515.09 | 520.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 12:15:00 | 524.40 | 501.04 | 505.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-06 12:15:00 | 524.40 | 501.04 | 505.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 12:15:00 | 524.40 | 501.04 | 505.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-06 13:00:00 | 524.40 | 501.04 | 505.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 13:15:00 | 505.40 | 501.91 | 505.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 14:30:00 | 500.10 | 501.35 | 505.23 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 15:00:00 | 499.10 | 501.35 | 505.23 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-07 11:15:00 | 518.55 | 508.16 | 507.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — BUY (started 2024-08-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-07 11:15:00 | 518.55 | 508.16 | 507.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-07 14:15:00 | 519.75 | 513.63 | 510.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-08 14:15:00 | 521.10 | 521.69 | 516.92 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-08 15:00:00 | 521.10 | 521.69 | 516.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-09 15:15:00 | 520.00 | 522.17 | 519.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-12 09:15:00 | 539.00 | 522.17 | 519.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-14 09:45:00 | 531.05 | 539.76 | 536.46 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-14 13:15:00 | 525.50 | 534.73 | 534.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 24 — SELL (started 2024-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 13:15:00 | 525.50 | 534.73 | 534.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 15:15:00 | 524.00 | 531.07 | 533.16 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-16 10:15:00 | 530.95 | 530.39 | 532.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-16 10:15:00 | 530.95 | 530.39 | 532.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 10:15:00 | 530.95 | 530.39 | 532.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 10:45:00 | 532.65 | 530.39 | 532.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 11:15:00 | 532.00 | 530.71 | 532.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-16 11:45:00 | 533.00 | 530.71 | 532.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 12:15:00 | 530.10 | 530.59 | 532.20 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 15:00:00 | 529.00 | 530.15 | 531.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 09:15:00 | 539.80 | 531.74 | 532.14 | SL hit (close>static) qty=1.00 sl=532.30 alert=retest2 |

### Cycle 25 — BUY (started 2024-08-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 12:15:00 | 534.50 | 532.79 | 532.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 535.95 | 533.48 | 532.95 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 10:15:00 | 533.10 | 533.40 | 532.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 10:15:00 | 533.10 | 533.40 | 532.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 10:15:00 | 533.10 | 533.40 | 532.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 11:00:00 | 533.10 | 533.40 | 532.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 540.55 | 534.83 | 533.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-21 09:15:00 | 543.90 | 537.45 | 535.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-22 11:45:00 | 547.70 | 540.84 | 538.59 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-23 11:15:00 | 533.50 | 538.56 | 538.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2024-08-23 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-23 11:15:00 | 533.50 | 538.56 | 538.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-23 13:15:00 | 531.35 | 536.37 | 537.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-26 09:15:00 | 537.50 | 535.20 | 536.72 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-26 09:15:00 | 537.50 | 535.20 | 536.72 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 537.50 | 535.20 | 536.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-26 09:30:00 | 540.75 | 535.20 | 536.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 10:15:00 | 532.80 | 534.72 | 536.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-26 12:00:00 | 532.00 | 534.18 | 535.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 09:30:00 | 531.35 | 529.59 | 531.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 11:15:00 | 505.40 | 511.21 | 513.98 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-04 15:15:00 | 509.75 | 509.22 | 511.97 | SL hit (close>ema200) qty=0.50 sl=509.22 alert=retest2 |

### Cycle 27 — BUY (started 2024-09-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-12 12:15:00 | 513.20 | 502.94 | 502.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-13 09:15:00 | 518.90 | 510.07 | 506.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-17 09:15:00 | 534.15 | 536.60 | 527.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-17 09:30:00 | 533.90 | 536.60 | 527.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 595.60 | 601.52 | 596.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 09:45:00 | 596.40 | 601.52 | 596.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 597.50 | 600.72 | 596.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-26 11:15:00 | 594.70 | 600.72 | 596.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 11:15:00 | 596.25 | 599.82 | 596.70 | EMA400 retest candle locked (from upside) |

### Cycle 28 — SELL (started 2024-09-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 09:15:00 | 593.15 | 595.07 | 595.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 12:15:00 | 586.00 | 589.73 | 591.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 14:15:00 | 591.30 | 589.51 | 591.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 14:15:00 | 591.30 | 589.51 | 591.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 591.30 | 589.51 | 591.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-30 15:00:00 | 591.30 | 589.51 | 591.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 591.25 | 589.86 | 591.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:15:00 | 587.95 | 589.86 | 591.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 586.10 | 589.11 | 590.86 | EMA400 retest candle locked (from downside) |

### Cycle 29 — BUY (started 2024-10-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-01 12:15:00 | 604.25 | 592.35 | 591.90 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-03 11:15:00 | 583.75 | 592.08 | 592.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-03 12:15:00 | 582.95 | 590.26 | 591.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 10:15:00 | 550.85 | 548.15 | 560.38 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-08 11:00:00 | 550.85 | 548.15 | 560.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 15:15:00 | 556.00 | 552.28 | 557.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:15:00 | 569.00 | 552.28 | 557.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-09 09:15:00 | 579.90 | 557.81 | 559.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-09 09:30:00 | 579.80 | 557.81 | 559.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 577.70 | 561.79 | 561.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-10 09:15:00 | 586.45 | 574.04 | 568.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-11 10:15:00 | 577.45 | 577.76 | 573.82 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-11 11:00:00 | 577.45 | 577.76 | 573.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 15:15:00 | 577.10 | 578.94 | 576.10 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-10-14 09:15:00 | 581.25 | 578.94 | 576.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-17 14:15:00 | 583.20 | 585.16 | 585.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2024-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-17 14:15:00 | 583.20 | 585.16 | 585.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-17 15:15:00 | 581.50 | 584.43 | 585.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-28 10:15:00 | 506.05 | 501.16 | 510.48 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-28 11:00:00 | 506.05 | 501.16 | 510.48 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-29 09:15:00 | 505.55 | 503.47 | 507.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 10:30:00 | 501.00 | 503.09 | 507.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-29 13:30:00 | 501.80 | 503.69 | 506.53 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-30 09:15:00 | 519.85 | 507.97 | 507.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 09:15:00 | 519.85 | 507.97 | 507.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 10:15:00 | 521.50 | 510.68 | 509.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-04 09:15:00 | 515.25 | 524.38 | 520.86 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-04 09:15:00 | 515.25 | 524.38 | 520.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 515.25 | 524.38 | 520.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 515.25 | 524.38 | 520.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 10:15:00 | 515.00 | 522.51 | 520.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 11:00:00 | 515.00 | 522.51 | 520.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-11-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 15:15:00 | 514.80 | 518.54 | 518.99 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-11-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 13:15:00 | 523.45 | 519.42 | 519.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 527.15 | 521.50 | 520.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 14:15:00 | 534.75 | 536.15 | 531.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 15:00:00 | 534.75 | 536.15 | 531.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 15:15:00 | 532.45 | 535.41 | 531.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-08 09:15:00 | 532.55 | 535.41 | 531.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-08 09:15:00 | 534.00 | 535.13 | 531.83 | EMA400 retest candle locked (from upside) |

### Cycle 36 — SELL (started 2024-11-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-08 13:15:00 | 522.05 | 528.76 | 529.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 15:15:00 | 519.20 | 525.87 | 528.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-11 13:15:00 | 529.85 | 523.65 | 525.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-11 13:15:00 | 529.85 | 523.65 | 525.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 13:15:00 | 529.85 | 523.65 | 525.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-11-11 14:00:00 | 529.85 | 523.65 | 525.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-11 14:15:00 | 512.40 | 521.40 | 524.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-12 10:00:00 | 510.00 | 517.69 | 522.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-13 14:15:00 | 484.50 | 495.23 | 504.19 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-11-14 09:15:00 | 496.85 | 494.38 | 502.17 | SL hit (close>ema200) qty=0.50 sl=494.38 alert=retest2 |

### Cycle 37 — BUY (started 2024-11-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-25 10:15:00 | 484.30 | 475.28 | 474.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-27 09:15:00 | 488.30 | 481.90 | 479.73 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 12:15:00 | 482.00 | 483.32 | 481.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-27 12:15:00 | 482.00 | 483.32 | 481.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 12:15:00 | 482.00 | 483.32 | 481.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 13:00:00 | 482.00 | 483.32 | 481.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-27 13:15:00 | 482.90 | 483.24 | 481.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-27 13:30:00 | 481.85 | 483.24 | 481.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 09:15:00 | 486.70 | 483.97 | 482.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-28 11:30:00 | 493.95 | 486.64 | 483.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-03 09:15:00 | 543.35 | 525.54 | 514.46 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 38 — SELL (started 2024-12-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-06 12:15:00 | 558.20 | 566.65 | 567.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-06 13:15:00 | 557.15 | 564.75 | 566.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-09 09:15:00 | 565.65 | 563.22 | 565.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-09 09:15:00 | 565.65 | 563.22 | 565.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 09:15:00 | 565.65 | 563.22 | 565.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 09:45:00 | 569.50 | 563.22 | 565.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 10:15:00 | 571.60 | 564.89 | 565.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-09 10:45:00 | 569.15 | 564.89 | 565.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-09 11:15:00 | 567.30 | 565.37 | 566.01 | EMA400 retest candle locked (from downside) |

### Cycle 39 — BUY (started 2024-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-09 14:15:00 | 568.20 | 566.45 | 566.40 | EMA200 above EMA400 |

### Cycle 40 — SELL (started 2024-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-10 09:15:00 | 561.50 | 565.43 | 565.95 | EMA200 below EMA400 |

### Cycle 41 — BUY (started 2024-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 09:15:00 | 573.25 | 566.63 | 565.98 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-12-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-12 09:15:00 | 559.95 | 566.14 | 566.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-12 10:15:00 | 558.50 | 564.61 | 565.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-13 10:15:00 | 563.50 | 560.50 | 562.50 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-13 10:15:00 | 563.50 | 560.50 | 562.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 10:15:00 | 563.50 | 560.50 | 562.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 10:45:00 | 564.00 | 560.50 | 562.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 11:15:00 | 564.20 | 561.24 | 562.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-13 11:45:00 | 563.70 | 561.24 | 562.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 12:15:00 | 561.45 | 561.28 | 562.54 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2024-12-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-16 09:15:00 | 564.45 | 563.12 | 563.10 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2024-12-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 09:15:00 | 560.80 | 563.32 | 563.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-17 13:15:00 | 557.65 | 560.73 | 562.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-18 09:15:00 | 560.75 | 558.93 | 560.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-18 09:15:00 | 560.75 | 558.93 | 560.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 09:15:00 | 560.75 | 558.93 | 560.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 09:30:00 | 568.85 | 558.93 | 560.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 10:15:00 | 561.00 | 559.34 | 560.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-18 10:30:00 | 562.45 | 559.34 | 560.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-18 11:15:00 | 557.65 | 559.00 | 560.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-18 12:30:00 | 556.10 | 558.77 | 560.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-19 10:15:00 | 564.40 | 561.06 | 560.90 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2024-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-19 10:15:00 | 564.40 | 561.06 | 560.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-19 12:15:00 | 566.15 | 562.50 | 561.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-20 09:15:00 | 559.70 | 562.76 | 562.09 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 09:15:00 | 559.70 | 562.76 | 562.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 09:15:00 | 559.70 | 562.76 | 562.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-20 10:00:00 | 559.70 | 562.76 | 562.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 570.00 | 564.21 | 562.81 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2024-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-23 09:15:00 | 554.05 | 561.77 | 562.33 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2024-12-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 09:15:00 | 566.90 | 562.07 | 562.07 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2024-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 14:15:00 | 561.20 | 562.02 | 562.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-24 15:15:00 | 556.50 | 560.92 | 561.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-26 10:15:00 | 562.40 | 560.49 | 561.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-26 10:15:00 | 562.40 | 560.49 | 561.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 10:15:00 | 562.40 | 560.49 | 561.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-26 11:00:00 | 562.40 | 560.49 | 561.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-26 11:15:00 | 558.80 | 560.15 | 561.01 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 13:45:00 | 557.00 | 559.56 | 560.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-26 15:15:00 | 558.10 | 559.31 | 560.38 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 09:45:00 | 557.90 | 558.86 | 559.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-27 12:00:00 | 558.05 | 558.00 | 559.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 14:15:00 | 560.10 | 558.45 | 559.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-27 15:00:00 | 560.10 | 558.45 | 559.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-27 15:15:00 | 563.00 | 559.36 | 559.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-30 09:15:00 | 568.75 | 559.36 | 559.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-30 09:15:00 | 576.50 | 562.79 | 561.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 09:15:00 | 576.50 | 562.79 | 561.11 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2024-12-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-31 09:15:00 | 556.00 | 560.79 | 561.42 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2024-12-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-31 15:15:00 | 563.95 | 561.80 | 561.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-01 11:15:00 | 572.00 | 564.75 | 563.02 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 14:15:00 | 571.20 | 572.88 | 570.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-03 14:15:00 | 571.20 | 572.88 | 570.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 14:15:00 | 571.20 | 572.88 | 570.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-03 15:00:00 | 571.20 | 572.88 | 570.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-03 15:15:00 | 569.00 | 572.10 | 570.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 09:15:00 | 555.45 | 572.10 | 570.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 52 — SELL (started 2025-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 09:15:00 | 556.75 | 569.03 | 569.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 10:15:00 | 550.65 | 565.36 | 567.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 12:15:00 | 557.55 | 557.50 | 560.84 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 12:45:00 | 556.85 | 557.50 | 560.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 557.00 | 557.40 | 560.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 13:30:00 | 557.20 | 557.40 | 560.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 09:15:00 | 510.90 | 503.63 | 511.36 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2025-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-20 10:15:00 | 515.30 | 511.29 | 510.98 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2025-01-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-21 10:15:00 | 505.95 | 510.96 | 511.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-21 12:15:00 | 502.40 | 508.35 | 510.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-22 14:15:00 | 491.55 | 490.75 | 497.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-22 15:00:00 | 491.55 | 490.75 | 497.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 09:15:00 | 491.70 | 491.42 | 494.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 09:30:00 | 495.40 | 491.42 | 494.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 495.50 | 492.24 | 494.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-24 11:00:00 | 495.50 | 492.24 | 494.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 11:15:00 | 497.55 | 493.30 | 494.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:00:00 | 494.65 | 494.55 | 495.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-24 14:15:00 | 501.30 | 495.90 | 495.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — BUY (started 2025-01-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-24 14:15:00 | 501.30 | 495.90 | 495.70 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 478.45 | 493.23 | 494.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-27 10:15:00 | 476.00 | 489.79 | 492.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-28 12:15:00 | 479.00 | 477.85 | 483.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-28 13:00:00 | 479.00 | 477.85 | 483.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 09:15:00 | 476.40 | 473.07 | 475.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-30 10:00:00 | 476.40 | 473.07 | 475.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 10:15:00 | 473.75 | 473.20 | 475.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-30 13:15:00 | 470.85 | 473.13 | 475.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-31 15:15:00 | 476.00 | 473.68 | 473.63 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 57 — BUY (started 2025-01-31 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-31 15:15:00 | 476.00 | 473.68 | 473.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 09:15:00 | 482.10 | 475.36 | 474.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-01 11:15:00 | 476.85 | 476.91 | 475.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-01 11:15:00 | 476.85 | 476.91 | 475.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 11:15:00 | 476.85 | 476.91 | 475.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 11:45:00 | 478.00 | 476.91 | 475.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 12:15:00 | 472.70 | 476.07 | 475.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-01 13:00:00 | 472.70 | 476.07 | 475.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-01 13:15:00 | 478.55 | 476.57 | 475.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-01 15:00:00 | 481.00 | 477.45 | 475.93 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-03 10:15:00 | 471.05 | 475.71 | 475.52 | SL hit (close<static) qty=1.00 sl=471.20 alert=retest2 |

### Cycle 58 — SELL (started 2025-02-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 11:15:00 | 468.50 | 474.27 | 474.88 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-02-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-04 11:15:00 | 484.95 | 475.24 | 474.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-04 13:15:00 | 486.15 | 478.84 | 476.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-06 13:15:00 | 501.30 | 505.21 | 497.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-02-06 14:00:00 | 501.30 | 505.21 | 497.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 09:15:00 | 500.15 | 503.36 | 498.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-07 10:15:00 | 496.60 | 503.36 | 498.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 10:15:00 | 500.80 | 502.85 | 498.81 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:15:00 | 502.55 | 502.85 | 498.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-07 11:45:00 | 503.65 | 502.54 | 499.04 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-07 13:15:00 | 493.20 | 500.26 | 498.58 | SL hit (close<static) qty=1.00 sl=495.50 alert=retest2 |

### Cycle 60 — SELL (started 2025-02-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-10 09:15:00 | 485.35 | 496.02 | 496.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-10 14:15:00 | 484.40 | 489.04 | 492.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-13 12:15:00 | 445.70 | 444.94 | 455.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-13 13:00:00 | 445.70 | 444.94 | 455.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 09:15:00 | 404.35 | 394.55 | 403.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 09:45:00 | 407.65 | 394.55 | 403.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 10:15:00 | 401.75 | 395.99 | 403.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 10:30:00 | 404.40 | 395.99 | 403.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 11:15:00 | 400.80 | 396.95 | 402.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 11:30:00 | 400.90 | 396.95 | 402.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 14:15:00 | 401.70 | 398.18 | 401.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-19 15:00:00 | 401.70 | 398.18 | 401.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-19 15:15:00 | 399.35 | 398.41 | 401.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-20 09:15:00 | 398.60 | 398.41 | 401.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-20 09:15:00 | 408.75 | 400.48 | 402.35 | SL hit (close>static) qty=1.00 sl=403.15 alert=retest2 |

### Cycle 61 — BUY (started 2025-02-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 11:15:00 | 419.00 | 406.03 | 404.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 13:15:00 | 421.85 | 411.40 | 407.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 408.65 | 414.30 | 410.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 408.65 | 414.30 | 410.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 408.65 | 414.30 | 410.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 408.65 | 414.30 | 410.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 409.70 | 413.38 | 410.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-21 14:45:00 | 417.00 | 413.84 | 411.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-24 09:45:00 | 412.45 | 414.77 | 412.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-02-25 09:15:00 | 412.55 | 413.54 | 412.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-25 10:15:00 | 408.80 | 412.27 | 412.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 62 — SELL (started 2025-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-25 10:15:00 | 408.80 | 412.27 | 412.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-25 11:15:00 | 407.50 | 411.31 | 411.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-28 13:15:00 | 395.90 | 393.24 | 397.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-28 13:15:00 | 395.90 | 393.24 | 397.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 13:15:00 | 395.90 | 393.24 | 397.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 13:45:00 | 395.80 | 393.24 | 397.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-28 14:15:00 | 392.25 | 393.05 | 397.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-28 14:45:00 | 396.50 | 393.05 | 397.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-03 09:15:00 | 370.65 | 387.60 | 394.00 | EMA400 retest candle locked (from downside) |

### Cycle 63 — BUY (started 2025-03-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-05 12:15:00 | 389.95 | 384.45 | 384.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 09:15:00 | 408.15 | 391.55 | 387.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 10:15:00 | 411.05 | 412.58 | 407.08 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 11:00:00 | 411.05 | 412.58 | 407.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 400.60 | 409.08 | 407.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 15:00:00 | 400.60 | 409.08 | 407.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 400.00 | 407.27 | 406.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 405.15 | 407.27 | 406.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 09:15:00 | 431.55 | 430.62 | 424.42 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 12:00:00 | 439.05 | 432.82 | 426.53 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-13 13:45:00 | 440.15 | 434.81 | 428.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 12:00:00 | 439.50 | 438.52 | 433.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-17 13:30:00 | 439.00 | 438.15 | 433.81 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 455.00 | 459.23 | 453.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:00:00 | 455.00 | 459.23 | 453.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 10:15:00 | 455.85 | 458.56 | 453.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 11:45:00 | 456.05 | 457.88 | 453.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-20 12:15:00 | 456.20 | 457.88 | 453.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 09:15:00 | 464.05 | 457.79 | 455.17 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-25 09:15:00 | 482.96 | 473.64 | 466.35 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 11:15:00 | 486.65 | 491.22 | 491.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 13:15:00 | 481.20 | 488.72 | 490.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 14:15:00 | 483.45 | 481.68 | 484.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-01 15:00:00 | 483.45 | 481.68 | 484.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 15:15:00 | 488.00 | 482.94 | 484.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 09:15:00 | 479.70 | 482.94 | 484.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-02 11:15:00 | 482.95 | 483.09 | 484.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-02 11:15:00 | 491.90 | 484.85 | 485.34 | SL hit (close>static) qty=1.00 sl=489.95 alert=retest2 |

### Cycle 65 — BUY (started 2025-04-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 12:15:00 | 497.65 | 487.41 | 486.46 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2025-04-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 11:15:00 | 482.65 | 490.30 | 490.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 480.90 | 487.06 | 489.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-07 15:15:00 | 458.80 | 455.04 | 467.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-08 09:15:00 | 465.80 | 455.04 | 467.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-08 09:15:00 | 458.50 | 455.73 | 466.38 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-09 09:15:00 | 456.00 | 461.04 | 464.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-15 10:15:00 | 466.10 | 456.39 | 456.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — BUY (started 2025-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 10:15:00 | 466.10 | 456.39 | 456.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 11:15:00 | 472.95 | 459.70 | 457.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-16 15:15:00 | 471.25 | 472.03 | 467.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 09:15:00 | 470.65 | 472.03 | 467.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 09:15:00 | 468.25 | 471.28 | 467.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-17 10:00:00 | 468.25 | 471.28 | 467.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-17 10:15:00 | 483.55 | 473.73 | 469.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 11:30:00 | 487.55 | 483.18 | 477.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 09:15:00 | 491.60 | 484.45 | 479.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 10:30:00 | 487.85 | 485.54 | 481.21 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-22 12:00:00 | 487.50 | 485.93 | 481.78 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 14:15:00 | 484.00 | 486.04 | 482.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-22 15:00:00 | 484.00 | 486.04 | 482.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 15:15:00 | 481.75 | 485.18 | 482.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:00:00 | 476.30 | 483.40 | 482.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 10:15:00 | 479.05 | 482.53 | 481.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 10:45:00 | 477.35 | 482.53 | 481.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 13:15:00 | 480.65 | 481.67 | 481.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 14:00:00 | 480.65 | 481.67 | 481.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 14:15:00 | 482.65 | 481.87 | 481.72 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-24 09:15:00 | 479.45 | 481.41 | 481.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 68 — SELL (started 2025-04-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 09:15:00 | 479.45 | 481.41 | 481.54 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2025-04-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-24 10:15:00 | 483.25 | 481.77 | 481.69 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-24 12:15:00 | 480.25 | 481.55 | 481.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-24 13:15:00 | 479.40 | 481.12 | 481.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 09:15:00 | 470.15 | 467.15 | 471.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 10:00:00 | 470.15 | 467.15 | 471.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 10:15:00 | 470.25 | 467.77 | 471.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-28 10:45:00 | 471.70 | 467.77 | 471.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 11:15:00 | 472.15 | 468.64 | 471.78 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-28 14:30:00 | 468.40 | 469.89 | 471.68 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 09:30:00 | 467.15 | 469.56 | 471.22 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 12:15:00 | 469.85 | 458.72 | 458.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — BUY (started 2025-05-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 12:15:00 | 469.85 | 458.72 | 458.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-05 13:15:00 | 471.40 | 461.26 | 459.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-06 11:15:00 | 460.60 | 464.49 | 462.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-06 11:15:00 | 460.60 | 464.49 | 462.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 11:15:00 | 460.60 | 464.49 | 462.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 12:00:00 | 460.60 | 464.49 | 462.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 12:15:00 | 460.00 | 463.59 | 462.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 12:45:00 | 460.50 | 463.59 | 462.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 13:15:00 | 462.00 | 463.27 | 462.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:15:00 | 458.45 | 463.27 | 462.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-06 14:15:00 | 461.35 | 462.89 | 462.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-06 14:45:00 | 457.45 | 462.89 | 462.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 72 — SELL (started 2025-05-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 15:15:00 | 455.40 | 461.39 | 461.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-07 09:15:00 | 453.30 | 459.77 | 460.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 11:15:00 | 462.25 | 460.06 | 460.64 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-07 11:15:00 | 462.25 | 460.06 | 460.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 462.25 | 460.06 | 460.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:00:00 | 462.25 | 460.06 | 460.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 12:15:00 | 457.45 | 459.54 | 460.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 12:30:00 | 461.55 | 459.54 | 460.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 13:15:00 | 459.30 | 459.49 | 460.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 14:00:00 | 459.30 | 459.49 | 460.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 14:15:00 | 459.85 | 459.56 | 460.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 15:00:00 | 459.85 | 459.56 | 460.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 460.35 | 459.72 | 460.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 462.15 | 459.72 | 460.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 73 — BUY (started 2025-05-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 09:15:00 | 464.15 | 460.61 | 460.59 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2025-05-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 12:15:00 | 457.45 | 460.58 | 460.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 13:15:00 | 453.00 | 459.06 | 459.94 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 462.15 | 445.31 | 449.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 462.15 | 445.31 | 449.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 462.15 | 445.31 | 449.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 462.15 | 445.31 | 449.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 10:15:00 | 465.05 | 449.26 | 450.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 11:00:00 | 465.05 | 449.26 | 450.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 75 — BUY (started 2025-05-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 11:15:00 | 465.85 | 452.58 | 451.99 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 470.00 | 456.06 | 453.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 13:15:00 | 463.45 | 464.39 | 460.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 13:45:00 | 464.00 | 464.39 | 460.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 555.70 | 498.95 | 487.61 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 11:45:00 | 560.80 | 521.61 | 500.45 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-19 14:15:00 | 566.50 | 536.11 | 511.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-22 14:15:00 | 530.60 | 535.37 | 535.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2025-05-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-22 14:15:00 | 530.60 | 535.37 | 535.78 | EMA200 below EMA400 |

### Cycle 77 — BUY (started 2025-05-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 09:15:00 | 542.15 | 536.03 | 535.97 | EMA200 above EMA400 |

### Cycle 78 — SELL (started 2025-05-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 12:15:00 | 535.70 | 537.66 | 537.75 | EMA200 below EMA400 |

### Cycle 79 — BUY (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 13:15:00 | 539.95 | 538.12 | 537.95 | EMA200 above EMA400 |

### Cycle 80 — SELL (started 2025-05-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-26 14:15:00 | 535.55 | 537.61 | 537.73 | EMA200 below EMA400 |

### Cycle 81 — BUY (started 2025-05-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-27 09:15:00 | 548.90 | 539.97 | 538.79 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-05-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-27 15:15:00 | 534.60 | 538.52 | 538.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 13:15:00 | 530.65 | 536.02 | 537.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 09:15:00 | 535.65 | 534.58 | 536.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 09:15:00 | 535.65 | 534.58 | 536.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 535.65 | 534.58 | 536.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:15:00 | 541.20 | 534.58 | 536.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 540.95 | 535.86 | 536.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:30:00 | 543.35 | 535.86 | 536.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-29 11:15:00 | 553.95 | 539.47 | 538.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-30 09:15:00 | 560.80 | 550.19 | 544.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-30 14:15:00 | 550.95 | 552.48 | 548.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-30 14:45:00 | 552.60 | 552.48 | 548.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 15:15:00 | 541.70 | 550.32 | 547.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-02 11:15:00 | 551.20 | 549.48 | 547.65 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 09:15:00 | 561.70 | 548.00 | 547.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-03 14:00:00 | 551.25 | 550.62 | 549.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-04 10:00:00 | 552.40 | 552.34 | 550.58 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 13:15:00 | 557.85 | 564.27 | 560.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 13:45:00 | 557.40 | 564.27 | 560.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 14:15:00 | 554.50 | 562.31 | 560.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-05 14:45:00 | 554.90 | 562.31 | 560.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-06-06 09:15:00 | 548.80 | 558.62 | 558.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — SELL (started 2025-06-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-06 09:15:00 | 548.80 | 558.62 | 558.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 14:15:00 | 541.05 | 549.64 | 553.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 09:15:00 | 552.00 | 549.01 | 552.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-09 09:15:00 | 552.00 | 549.01 | 552.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 552.00 | 549.01 | 552.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 10:00:00 | 552.00 | 549.01 | 552.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 10:15:00 | 550.55 | 549.32 | 552.49 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 10:15:00 | 560.05 | 554.46 | 553.78 | EMA200 above EMA400 |

### Cycle 86 — SELL (started 2025-06-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 13:15:00 | 548.10 | 553.34 | 553.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 09:15:00 | 544.20 | 550.64 | 552.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 530.50 | 525.80 | 532.03 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:00:00 | 530.50 | 525.80 | 532.03 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 532.20 | 527.08 | 532.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:45:00 | 532.60 | 527.08 | 532.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 13:15:00 | 533.30 | 528.33 | 532.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 13:30:00 | 532.85 | 528.33 | 532.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 537.80 | 530.22 | 532.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 14:45:00 | 537.15 | 530.22 | 532.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2025-06-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 09:15:00 | 541.70 | 534.18 | 534.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-18 14:15:00 | 553.50 | 541.28 | 538.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-19 11:15:00 | 544.35 | 545.38 | 541.61 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-19 12:00:00 | 544.35 | 545.38 | 541.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 12:15:00 | 538.50 | 544.01 | 541.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:00:00 | 538.50 | 544.01 | 541.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 13:15:00 | 540.45 | 543.30 | 541.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 13:30:00 | 537.35 | 543.30 | 541.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 14:15:00 | 539.15 | 542.47 | 541.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-19 14:30:00 | 540.00 | 542.47 | 541.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 15:15:00 | 534.70 | 540.91 | 540.48 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-20 09:15:00 | 544.90 | 540.91 | 540.48 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-02 11:15:00 | 562.60 | 566.96 | 567.12 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-07-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 11:15:00 | 562.60 | 566.96 | 567.12 | EMA200 below EMA400 |

### Cycle 89 — BUY (started 2025-07-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-03 14:15:00 | 567.20 | 566.12 | 566.10 | EMA200 above EMA400 |

### Cycle 90 — SELL (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-04 09:15:00 | 565.25 | 566.03 | 566.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 562.05 | 565.09 | 565.63 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 15:15:00 | 565.00 | 564.38 | 565.04 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 15:15:00 | 565.00 | 564.38 | 565.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 565.00 | 564.38 | 565.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 560.60 | 564.38 | 565.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 561.40 | 563.78 | 564.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 10:30:00 | 559.50 | 562.53 | 564.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 09:30:00 | 559.45 | 557.61 | 560.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 12:45:00 | 559.60 | 558.62 | 558.82 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-09 14:15:00 | 558.70 | 558.84 | 558.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 14:15:00 | 560.00 | 559.08 | 559.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 91 — BUY (started 2025-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 14:15:00 | 560.00 | 559.08 | 559.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-10 09:15:00 | 566.25 | 560.50 | 559.66 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 13:15:00 | 564.50 | 564.53 | 563.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 13:30:00 | 564.00 | 564.53 | 563.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 563.00 | 564.23 | 563.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:30:00 | 563.05 | 564.23 | 563.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 560.95 | 563.57 | 562.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-14 09:15:00 | 563.95 | 563.57 | 562.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-17 15:15:00 | 578.00 | 582.82 | 582.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2025-07-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-17 15:15:00 | 578.00 | 582.82 | 582.86 | EMA200 below EMA400 |

### Cycle 93 — BUY (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 12:15:00 | 583.45 | 582.92 | 582.87 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 13:15:00 | 581.05 | 582.55 | 582.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 14:15:00 | 577.90 | 581.62 | 582.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 581.75 | 580.90 | 581.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-21 09:15:00 | 581.75 | 580.90 | 581.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 581.75 | 580.90 | 581.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:45:00 | 581.85 | 580.90 | 581.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 10:15:00 | 580.70 | 580.86 | 581.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 10:30:00 | 581.30 | 580.86 | 581.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 11:15:00 | 580.50 | 580.79 | 581.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-21 11:45:00 | 581.75 | 580.79 | 581.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 578.80 | 578.65 | 580.05 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 10:15:00 | 576.40 | 578.65 | 580.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 11:00:00 | 575.95 | 578.11 | 579.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 10:30:00 | 574.85 | 574.26 | 574.41 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 09:15:00 | 547.58 | 564.89 | 569.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 09:15:00 | 547.15 | 564.89 | 569.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-25 09:15:00 | 546.11 | 564.89 | 569.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-28 09:15:00 | 551.55 | 549.25 | 557.51 | SL hit (close>ema200) qty=0.50 sl=549.25 alert=retest2 |

### Cycle 95 — BUY (started 2025-07-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 10:15:00 | 557.60 | 551.24 | 550.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 09:15:00 | 582.00 | 559.76 | 555.41 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-31 14:15:00 | 571.65 | 572.45 | 564.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-31 15:00:00 | 571.65 | 572.45 | 564.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |

### Cycle 96 — SELL (started 2025-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-01 14:15:00 | 527.10 | 564.92 | 565.36 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-08-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 14:15:00 | 536.10 | 527.28 | 527.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-13 10:15:00 | 539.70 | 532.73 | 529.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 09:15:00 | 536.35 | 536.76 | 533.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 09:45:00 | 537.05 | 536.76 | 533.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 11:15:00 | 535.45 | 536.46 | 534.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 11:45:00 | 534.35 | 536.46 | 534.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 13:15:00 | 534.05 | 535.88 | 534.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 13:45:00 | 534.00 | 535.88 | 534.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 14:15:00 | 534.90 | 535.69 | 534.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-14 14:30:00 | 535.75 | 535.69 | 534.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 533.40 | 535.23 | 534.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 539.15 | 535.23 | 534.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 13:15:00 | 541.65 | 545.44 | 545.94 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 98 — SELL (started 2025-08-21 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 13:15:00 | 541.65 | 545.44 | 545.94 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 14:15:00 | 541.15 | 544.58 | 545.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 14:15:00 | 521.15 | 520.56 | 526.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-26 15:00:00 | 521.15 | 520.56 | 526.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 09:15:00 | 517.45 | 512.76 | 515.90 | EMA400 retest candle locked (from downside) |

### Cycle 99 — BUY (started 2025-09-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 15:15:00 | 519.90 | 516.89 | 516.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 09:15:00 | 524.70 | 518.46 | 517.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 12:15:00 | 518.95 | 520.55 | 518.92 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 12:15:00 | 518.95 | 520.55 | 518.92 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 518.95 | 520.55 | 518.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 12:45:00 | 518.70 | 520.55 | 518.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 518.50 | 520.14 | 518.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 09:45:00 | 527.70 | 521.09 | 519.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-03 13:00:00 | 527.30 | 523.12 | 520.97 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:45:00 | 531.10 | 525.29 | 522.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 13:45:00 | 527.15 | 526.75 | 524.39 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 14:15:00 | 521.45 | 525.69 | 524.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 15:00:00 | 521.45 | 525.69 | 524.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 15:15:00 | 519.65 | 524.48 | 523.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 09:15:00 | 520.05 | 524.48 | 523.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-05 09:15:00 | 517.00 | 522.98 | 523.10 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — SELL (started 2025-09-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 09:15:00 | 517.00 | 522.98 | 523.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-05 10:15:00 | 516.60 | 521.71 | 522.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-08 09:15:00 | 522.55 | 519.00 | 520.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-08 09:15:00 | 522.55 | 519.00 | 520.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 522.55 | 519.00 | 520.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:45:00 | 522.80 | 519.00 | 520.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 10:15:00 | 521.55 | 519.51 | 520.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-08 12:00:00 | 518.00 | 519.21 | 520.25 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 09:30:00 | 521.00 | 518.06 | 519.15 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-09 11:30:00 | 520.55 | 518.65 | 519.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-10 09:15:00 | 524.40 | 520.37 | 519.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 101 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 524.40 | 520.37 | 519.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-12 12:15:00 | 533.65 | 525.12 | 523.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 13:15:00 | 571.30 | 571.66 | 566.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-22 14:00:00 | 571.30 | 571.66 | 566.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 573.80 | 571.24 | 567.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 10:15:00 | 582.55 | 571.24 | 567.67 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 09:45:00 | 577.95 | 573.63 | 570.90 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 13:45:00 | 576.45 | 573.48 | 571.66 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-25 09:45:00 | 578.40 | 573.97 | 572.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 572.25 | 574.28 | 573.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:00:00 | 572.25 | 574.28 | 573.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 14:15:00 | 571.40 | 573.70 | 572.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:45:00 | 572.00 | 573.70 | 572.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 571.80 | 573.32 | 572.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-26 09:15:00 | 565.00 | 573.32 | 572.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-26 09:15:00 | 560.85 | 570.83 | 571.74 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 102 — SELL (started 2025-09-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 09:15:00 | 560.85 | 570.83 | 571.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 10:15:00 | 558.65 | 568.39 | 570.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 09:15:00 | 550.00 | 549.68 | 555.75 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 10:00:00 | 550.00 | 549.68 | 555.75 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 14:15:00 | 555.30 | 549.39 | 553.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-30 15:00:00 | 555.30 | 549.39 | 553.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 558.50 | 551.21 | 553.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:15:00 | 560.70 | 551.21 | 553.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 09:15:00 | 551.55 | 549.74 | 551.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 09:30:00 | 553.80 | 549.74 | 551.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 10:15:00 | 547.30 | 549.25 | 551.11 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-03 11:15:00 | 546.00 | 549.25 | 551.11 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-03 14:15:00 | 553.90 | 549.01 | 550.19 | SL hit (close>static) qty=1.00 sl=552.70 alert=retest2 |

### Cycle 103 — BUY (started 2025-10-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 09:15:00 | 566.05 | 553.86 | 552.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-08 10:15:00 | 588.00 | 575.53 | 568.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-09 09:15:00 | 587.10 | 588.45 | 579.58 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-09 09:30:00 | 583.40 | 588.45 | 579.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 582.00 | 586.13 | 580.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 12:45:00 | 581.85 | 586.13 | 580.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 13:15:00 | 576.30 | 584.16 | 580.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-09 14:00:00 | 576.30 | 584.16 | 580.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 14:15:00 | 587.50 | 584.83 | 580.90 | EMA400 retest candle locked (from upside) |

### Cycle 104 — SELL (started 2025-10-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-10 11:15:00 | 571.70 | 579.01 | 579.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-10 12:15:00 | 569.00 | 577.01 | 578.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 09:15:00 | 552.60 | 551.98 | 557.36 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-15 09:30:00 | 552.50 | 551.98 | 557.36 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 12:15:00 | 554.80 | 553.03 | 556.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 12:45:00 | 556.85 | 553.03 | 556.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 557.15 | 554.23 | 556.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 557.15 | 554.23 | 556.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 553.10 | 554.01 | 556.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 560.00 | 554.01 | 556.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 558.45 | 554.89 | 556.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:30:00 | 561.05 | 554.89 | 556.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 11:15:00 | 558.25 | 555.97 | 556.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:00:00 | 558.25 | 555.97 | 556.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 12:15:00 | 560.75 | 556.92 | 557.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 12:30:00 | 560.80 | 556.92 | 557.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 105 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 559.45 | 557.43 | 557.25 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-10-16 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 15:15:00 | 554.75 | 556.89 | 557.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-17 10:15:00 | 552.70 | 555.96 | 556.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-17 15:15:00 | 553.90 | 553.64 | 555.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-20 09:15:00 | 555.10 | 553.64 | 555.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 550.00 | 552.91 | 554.60 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 10:45:00 | 549.70 | 552.23 | 554.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 11:45:00 | 549.15 | 551.63 | 553.69 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 12:45:00 | 549.40 | 551.24 | 553.32 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-20 13:30:00 | 549.65 | 550.73 | 552.90 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 13:15:00 | 556.50 | 551.26 | 552.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:00:00 | 556.50 | 551.26 | 552.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 557.00 | 552.41 | 552.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 557.00 | 552.41 | 552.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-10-23 10:15:00 | 556.80 | 553.74 | 553.49 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 107 — BUY (started 2025-10-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 10:15:00 | 556.80 | 553.74 | 553.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-23 11:15:00 | 561.20 | 555.23 | 554.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 14:15:00 | 556.45 | 557.06 | 555.41 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 15:00:00 | 556.45 | 557.06 | 555.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 556.00 | 556.85 | 555.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 559.25 | 556.85 | 555.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 560.70 | 557.62 | 555.94 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-27 09:45:00 | 566.60 | 558.30 | 557.10 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 567.75 | 560.71 | 558.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-10-29 10:15:00 | 623.26 | 594.01 | 579.33 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-11-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 14:15:00 | 614.00 | 620.07 | 620.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 601.45 | 615.34 | 618.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-11 09:15:00 | 569.70 | 559.42 | 574.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-11 10:00:00 | 569.70 | 559.42 | 574.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 10:15:00 | 571.15 | 561.77 | 574.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 11:00:00 | 571.15 | 561.77 | 574.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 11:15:00 | 578.10 | 565.03 | 574.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 12:00:00 | 578.10 | 565.03 | 574.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 12:15:00 | 571.55 | 566.34 | 574.24 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 13:30:00 | 568.95 | 566.72 | 573.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 15:00:00 | 568.65 | 567.11 | 573.24 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 09:15:00 | 565.20 | 567.68 | 572.94 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 14:15:00 | 568.25 | 565.97 | 569.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 14:15:00 | 568.90 | 566.55 | 569.75 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-13 14:15:00 | 571.65 | 571.05 | 571.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 109 — BUY (started 2025-11-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-13 14:15:00 | 571.65 | 571.05 | 571.01 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-14 10:15:00 | 574.70 | 571.80 | 571.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 12:15:00 | 570.45 | 571.90 | 571.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 12:15:00 | 570.45 | 571.90 | 571.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 12:15:00 | 570.45 | 571.90 | 571.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:00:00 | 570.45 | 571.90 | 571.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 569.50 | 571.42 | 571.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:30:00 | 568.00 | 571.42 | 571.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 575.00 | 572.44 | 571.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 574.30 | 572.44 | 571.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 574.00 | 572.75 | 572.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-18 11:15:00 | 580.15 | 573.67 | 572.96 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-19 14:15:00 | 569.55 | 579.14 | 579.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 110 — SELL (started 2025-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 14:15:00 | 569.55 | 579.14 | 579.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 09:15:00 | 564.50 | 574.66 | 577.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 14:15:00 | 564.80 | 564.72 | 570.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 14:30:00 | 563.25 | 564.72 | 570.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 554.30 | 551.09 | 556.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 554.30 | 551.09 | 556.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 562.80 | 553.74 | 556.55 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2025-11-26 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 12:15:00 | 559.40 | 557.01 | 556.82 | EMA200 above EMA400 |

### Cycle 112 — SELL (started 2025-11-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 11:15:00 | 556.80 | 556.96 | 556.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 14:15:00 | 554.90 | 556.41 | 556.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-28 13:15:00 | 555.30 | 552.74 | 554.31 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 13:15:00 | 555.30 | 552.74 | 554.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 13:15:00 | 555.30 | 552.74 | 554.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 13:45:00 | 556.00 | 552.74 | 554.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 14:15:00 | 555.00 | 553.19 | 554.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-28 15:00:00 | 555.00 | 553.19 | 554.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 10:15:00 | 556.25 | 554.77 | 554.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-01 11:45:00 | 552.10 | 554.25 | 554.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-08 13:15:00 | 524.50 | 534.00 | 538.14 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 10:15:00 | 530.45 | 529.43 | 534.33 | SL hit (close>ema200) qty=0.50 sl=529.43 alert=retest2 |

### Cycle 113 — BUY (started 2025-12-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 15:15:00 | 540.00 | 536.41 | 536.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-10 09:15:00 | 548.20 | 538.77 | 537.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-10 15:15:00 | 539.95 | 542.75 | 540.59 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-10 15:15:00 | 539.95 | 542.75 | 540.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-10 15:15:00 | 539.95 | 542.75 | 540.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 09:15:00 | 545.55 | 542.75 | 540.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 10:00:00 | 546.15 | 543.43 | 541.09 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-11 12:15:00 | 545.50 | 543.93 | 541.74 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-12 12:15:00 | 539.50 | 541.43 | 541.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — SELL (started 2025-12-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 12:15:00 | 539.50 | 541.43 | 541.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 13:15:00 | 538.60 | 540.86 | 541.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 14:15:00 | 541.15 | 540.92 | 541.18 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 14:15:00 | 541.15 | 540.92 | 541.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 541.15 | 540.92 | 541.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-12 15:15:00 | 542.50 | 540.92 | 541.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 15:15:00 | 542.50 | 541.24 | 541.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:15:00 | 545.90 | 541.24 | 541.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 115 — BUY (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 09:15:00 | 545.55 | 542.10 | 541.69 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-12-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-16 11:15:00 | 538.35 | 542.26 | 542.55 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 12:15:00 | 536.85 | 541.18 | 542.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-18 15:15:00 | 528.70 | 527.66 | 531.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-19 09:15:00 | 531.85 | 527.66 | 531.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 535.50 | 529.23 | 531.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:45:00 | 536.60 | 529.23 | 531.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 544.70 | 532.32 | 533.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-19 10:30:00 | 544.50 | 532.32 | 533.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 117 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 541.50 | 534.16 | 533.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 550.15 | 540.10 | 536.88 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-24 14:15:00 | 571.40 | 572.25 | 566.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-24 15:00:00 | 571.40 | 572.25 | 566.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 09:15:00 | 578.00 | 572.86 | 567.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 10:30:00 | 582.35 | 574.77 | 568.80 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 12:00:00 | 582.20 | 576.26 | 570.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-26 12:45:00 | 581.00 | 577.28 | 571.05 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-12-31 10:15:00 | 640.59 | 603.73 | 595.47 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 118 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 652.05 | 661.06 | 661.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 15:15:00 | 646.50 | 653.95 | 657.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-14 09:15:00 | 632.00 | 609.66 | 618.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-14 09:15:00 | 632.00 | 609.66 | 618.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 632.00 | 609.66 | 618.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:00:00 | 632.00 | 609.66 | 618.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 644.15 | 616.56 | 620.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 644.15 | 616.56 | 620.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — BUY (started 2026-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 12:15:00 | 638.00 | 624.72 | 623.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 09:15:00 | 657.30 | 634.90 | 629.27 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 14:15:00 | 639.75 | 641.02 | 635.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 15:00:00 | 639.75 | 641.02 | 635.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 642.85 | 641.07 | 636.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 11:30:00 | 647.35 | 642.29 | 637.52 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 12:15:00 | 647.25 | 642.29 | 637.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 11:15:00 | 632.25 | 638.70 | 638.10 | SL hit (close<static) qty=1.00 sl=633.15 alert=retest2 |

### Cycle 120 — SELL (started 2026-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 12:15:00 | 629.00 | 636.76 | 637.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 13:15:00 | 623.35 | 634.08 | 636.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 629.15 | 626.78 | 630.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-21 12:15:00 | 629.15 | 626.78 | 630.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 12:15:00 | 629.15 | 626.78 | 630.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 12:45:00 | 632.95 | 626.78 | 630.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 13:15:00 | 617.50 | 624.92 | 629.38 | EMA400 retest candle locked (from downside) |

### Cycle 121 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 635.95 | 630.50 | 630.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-22 14:15:00 | 636.50 | 631.70 | 630.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 10:15:00 | 631.05 | 632.20 | 631.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 10:15:00 | 631.05 | 632.20 | 631.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 631.05 | 632.20 | 631.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:30:00 | 631.60 | 632.20 | 631.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 11:15:00 | 625.10 | 630.78 | 630.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 12:00:00 | 625.10 | 630.78 | 630.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — SELL (started 2026-01-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 12:15:00 | 621.05 | 628.83 | 629.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 13:15:00 | 613.40 | 625.75 | 628.28 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 09:15:00 | 627.90 | 621.83 | 625.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-27 09:15:00 | 627.90 | 621.83 | 625.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 627.90 | 621.83 | 625.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:45:00 | 627.10 | 621.83 | 625.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 10:15:00 | 623.05 | 622.07 | 625.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-27 14:00:00 | 620.00 | 621.38 | 624.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-27 15:15:00 | 633.00 | 624.37 | 625.03 | SL hit (close>static) qty=1.00 sl=628.15 alert=retest2 |

### Cycle 123 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 640.95 | 628.51 | 626.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 11:15:00 | 645.80 | 631.97 | 628.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-30 09:15:00 | 637.80 | 656.49 | 649.27 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 637.80 | 656.49 | 649.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 637.80 | 656.49 | 649.27 | EMA400 retest candle locked (from upside) |

### Cycle 124 — SELL (started 2026-01-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-30 12:15:00 | 621.10 | 643.08 | 644.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-30 15:15:00 | 609.90 | 631.46 | 638.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 14:15:00 | 600.90 | 595.00 | 606.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 15:00:00 | 600.90 | 595.00 | 606.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 612.60 | 599.96 | 606.92 | EMA400 retest candle locked (from downside) |

### Cycle 125 — BUY (started 2026-02-03 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 12:15:00 | 622.15 | 611.30 | 610.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 640.10 | 622.54 | 616.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 09:15:00 | 623.30 | 630.69 | 625.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-05 09:15:00 | 623.30 | 630.69 | 625.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 09:15:00 | 623.30 | 630.69 | 625.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 10:00:00 | 623.30 | 630.69 | 625.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 10:15:00 | 620.10 | 628.57 | 624.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-05 11:00:00 | 620.10 | 628.57 | 624.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-05 11:15:00 | 627.25 | 628.31 | 624.88 | EMA400 retest candle locked (from upside) |

### Cycle 126 — SELL (started 2026-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 09:15:00 | 610.70 | 622.48 | 623.23 | EMA200 below EMA400 |

### Cycle 127 — BUY (started 2026-02-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-09 09:15:00 | 643.75 | 624.98 | 623.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-10 09:15:00 | 656.35 | 638.14 | 631.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-11 09:15:00 | 653.30 | 657.06 | 646.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-11 09:15:00 | 653.30 | 657.06 | 646.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-11 09:15:00 | 653.30 | 657.06 | 646.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-11 10:00:00 | 653.30 | 657.06 | 646.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 655.10 | 663.96 | 659.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:00:00 | 655.10 | 663.96 | 659.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 661.80 | 663.52 | 659.80 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2026-02-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 15:15:00 | 642.40 | 656.52 | 657.80 | EMA200 below EMA400 |

### Cycle 129 — BUY (started 2026-02-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 14:15:00 | 672.60 | 656.79 | 656.18 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 686.55 | 665.09 | 660.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 15:15:00 | 671.70 | 672.05 | 666.50 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 15:15:00 | 671.70 | 672.05 | 666.50 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 15:15:00 | 671.70 | 672.05 | 666.50 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-18 09:15:00 | 685.90 | 672.05 | 666.50 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-24 12:15:00 | 680.85 | 687.27 | 687.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 130 — SELL (started 2026-02-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-24 12:15:00 | 680.85 | 687.27 | 687.85 | EMA200 below EMA400 |

### Cycle 131 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 710.00 | 691.97 | 689.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-25 11:15:00 | 724.20 | 698.42 | 692.75 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 14:15:00 | 718.00 | 730.30 | 722.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 14:15:00 | 718.00 | 730.30 | 722.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 718.00 | 730.30 | 722.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 718.00 | 730.30 | 722.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 717.20 | 727.68 | 722.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 707.65 | 727.68 | 722.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-02 10:15:00 | 702.15 | 720.24 | 719.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 11:00:00 | 702.15 | 720.24 | 719.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — SELL (started 2026-03-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 11:15:00 | 694.30 | 715.05 | 717.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 688.30 | 706.21 | 711.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 09:15:00 | 684.50 | 682.21 | 694.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 10:15:00 | 691.60 | 682.21 | 694.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 10:15:00 | 691.15 | 684.00 | 693.88 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 11:45:00 | 680.25 | 683.95 | 692.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 12:30:00 | 683.40 | 683.87 | 692.10 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:00:00 | 683.55 | 683.87 | 692.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-05 13:30:00 | 682.35 | 682.49 | 690.73 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 679.85 | 682.62 | 688.28 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 11:45:00 | 678.70 | 681.44 | 687.22 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:00:00 | 678.90 | 680.93 | 686.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 13:30:00 | 678.05 | 680.44 | 685.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 646.24 | 668.86 | 678.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 649.23 | 668.86 | 678.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 649.37 | 668.86 | 678.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 648.23 | 668.86 | 678.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 644.76 | 668.86 | 678.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 644.95 | 668.86 | 678.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-09 09:15:00 | 644.15 | 668.86 | 678.84 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-10 14:15:00 | 636.00 | 634.63 | 646.66 | SL hit (close>ema200) qty=0.50 sl=634.63 alert=retest2 |

### Cycle 133 — BUY (started 2026-03-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-12 12:15:00 | 656.50 | 650.15 | 649.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 13:15:00 | 657.35 | 651.59 | 650.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-12 14:15:00 | 646.30 | 650.53 | 649.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-12 14:15:00 | 646.30 | 650.53 | 649.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 14:15:00 | 646.30 | 650.53 | 649.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-12 15:00:00 | 646.30 | 650.53 | 649.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-12 15:15:00 | 647.95 | 650.02 | 649.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 09:15:00 | 634.35 | 650.02 | 649.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 134 — SELL (started 2026-03-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 09:15:00 | 628.00 | 645.61 | 647.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 10:15:00 | 614.40 | 639.37 | 644.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 612.60 | 608.16 | 620.42 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:45:00 | 611.00 | 608.16 | 620.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 616.85 | 610.97 | 619.64 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:15:00 | 610.85 | 611.82 | 618.56 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 12:45:00 | 611.55 | 611.85 | 617.96 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 13:15:00 | 611.20 | 611.85 | 617.96 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-18 09:15:00 | 625.20 | 617.75 | 619.11 | SL hit (close>static) qty=1.00 sl=623.75 alert=retest2 |

### Cycle 135 — BUY (started 2026-03-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 11:15:00 | 625.85 | 620.99 | 620.44 | EMA200 above EMA400 |

### Cycle 136 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 613.05 | 619.95 | 620.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 13:15:00 | 605.20 | 615.01 | 618.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-20 09:15:00 | 621.50 | 612.29 | 615.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-20 09:15:00 | 621.50 | 612.29 | 615.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-20 09:15:00 | 621.50 | 612.29 | 615.68 | EMA400 retest candle locked (from downside) |

### Cycle 137 — BUY (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-20 13:15:00 | 621.30 | 617.87 | 617.60 | EMA200 above EMA400 |

### Cycle 138 — SELL (started 2026-03-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 14:15:00 | 606.30 | 615.55 | 616.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 575.50 | 606.29 | 612.11 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 12:15:00 | 572.50 | 570.71 | 583.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 12:45:00 | 571.80 | 570.71 | 583.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 09:15:00 | 589.20 | 575.34 | 581.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:00:00 | 589.20 | 575.34 | 581.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 10:15:00 | 589.00 | 578.07 | 582.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-25 10:45:00 | 589.20 | 578.07 | 582.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — BUY (started 2026-03-25 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 13:15:00 | 596.65 | 586.29 | 585.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-27 09:15:00 | 642.40 | 600.34 | 592.43 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 11:15:00 | 628.20 | 629.30 | 617.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-30 11:45:00 | 626.25 | 629.30 | 617.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 618.25 | 626.43 | 618.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 15:00:00 | 618.25 | 626.43 | 618.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 620.00 | 625.14 | 618.85 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 655.60 | 625.14 | 618.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 09:45:00 | 629.75 | 638.27 | 631.79 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:30:00 | 632.60 | 633.79 | 630.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 09:30:00 | 641.00 | 636.09 | 633.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 10:15:00 | 630.25 | 634.92 | 632.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 11:00:00 | 630.25 | 634.92 | 632.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 11:15:00 | 629.75 | 633.89 | 632.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 12:00:00 | 629.75 | 633.89 | 632.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 12:15:00 | 631.45 | 633.40 | 632.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 12:30:00 | 628.90 | 633.40 | 632.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 14:15:00 | 635.70 | 633.51 | 632.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-06 14:45:00 | 632.65 | 633.51 | 632.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 09:15:00 | 640.15 | 636.18 | 634.09 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2026-04-07 15:15:00 | 629.00 | 634.05 | 634.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 140 — SELL (started 2026-04-07 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 15:15:00 | 629.00 | 634.05 | 634.18 | EMA200 below EMA400 |

### Cycle 141 — BUY (started 2026-04-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 09:15:00 | 650.65 | 637.37 | 635.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-08 10:15:00 | 655.80 | 641.05 | 637.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-08 15:15:00 | 643.50 | 643.72 | 640.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-08 15:15:00 | 643.50 | 643.72 | 640.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-08 15:15:00 | 643.50 | 643.72 | 640.51 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-09 09:15:00 | 653.80 | 643.72 | 640.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-09 12:15:00 | 638.35 | 645.08 | 642.48 | SL hit (close<static) qty=1.00 sl=640.10 alert=retest2 |

### Cycle 142 — SELL (started 2026-04-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-10 13:15:00 | 638.55 | 641.08 | 641.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 09:15:00 | 626.30 | 637.81 | 639.75 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-13 10:15:00 | 640.80 | 638.41 | 639.84 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 10:15:00 | 640.80 | 638.41 | 639.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 10:15:00 | 640.80 | 638.41 | 639.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-13 11:00:00 | 640.80 | 638.41 | 639.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 11:15:00 | 641.95 | 639.12 | 640.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 13:00:00 | 637.00 | 638.69 | 639.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-13 15:15:00 | 637.90 | 638.87 | 639.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-15 09:15:00 | 663.40 | 643.62 | 641.67 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 663.40 | 643.62 | 641.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 10:15:00 | 665.00 | 647.90 | 643.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-17 15:15:00 | 678.35 | 678.87 | 671.89 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-20 09:15:00 | 686.00 | 678.87 | 671.89 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2026-04-20 09:45:00 | 684.80 | 680.41 | 673.23 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-20 10:15:00 | 754.60 | 694.51 | 680.29 | Target hit (10%) qty=1.00 alert=retest1 |

### Cycle 144 — SELL (started 2026-04-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 12:15:00 | 722.45 | 730.54 | 731.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-24 13:15:00 | 720.00 | 728.43 | 730.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 740.80 | 729.67 | 730.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 740.80 | 729.67 | 730.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 740.80 | 729.67 | 730.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:30:00 | 746.15 | 729.67 | 730.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 145 — BUY (started 2026-04-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 10:15:00 | 738.85 | 731.51 | 730.90 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-29 11:15:00 | 745.65 | 739.89 | 737.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-29 14:15:00 | 740.00 | 741.11 | 738.51 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-29 14:15:00 | 740.00 | 741.11 | 738.51 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 14:15:00 | 740.00 | 741.11 | 738.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 15:00:00 | 740.00 | 741.11 | 738.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 15:15:00 | 740.10 | 740.91 | 738.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 09:15:00 | 713.80 | 740.91 | 738.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 146 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 702.60 | 733.24 | 735.38 | EMA200 below EMA400 |

### Cycle 147 — BUY (started 2026-05-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 09:15:00 | 716.25 | 712.93 | 712.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-07 10:15:00 | 726.00 | 715.54 | 713.85 | Break + close above crossover candle high |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-16 09:15:00 | 597.10 | 2024-05-23 10:15:00 | 594.50 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2024-05-28 12:00:00 | 573.40 | 2024-06-03 13:15:00 | 581.30 | STOP_HIT | 1.00 | -1.38% |
| SELL | retest2 | 2024-05-30 09:30:00 | 575.70 | 2024-06-03 13:15:00 | 581.30 | STOP_HIT | 1.00 | -0.97% |
| BUY | retest1 | 2024-06-11 09:15:00 | 570.85 | 2024-06-12 14:15:00 | 570.05 | STOP_HIT | 1.00 | -0.14% |
| BUY | retest2 | 2024-06-13 09:15:00 | 572.50 | 2024-06-14 15:15:00 | 571.65 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2024-06-13 11:15:00 | 572.65 | 2024-06-14 15:15:00 | 571.65 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2024-06-14 14:00:00 | 572.95 | 2024-06-14 15:15:00 | 571.65 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2024-06-21 09:15:00 | 595.55 | 2024-06-24 15:15:00 | 583.00 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2024-06-21 10:15:00 | 592.60 | 2024-06-24 15:15:00 | 583.00 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2024-06-21 12:15:00 | 601.00 | 2024-06-24 15:15:00 | 583.00 | STOP_HIT | 1.00 | -3.00% |
| BUY | retest2 | 2024-06-24 10:30:00 | 591.75 | 2024-06-24 15:15:00 | 583.00 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2024-07-01 14:15:00 | 571.25 | 2024-07-02 09:15:00 | 577.85 | STOP_HIT | 1.00 | -1.16% |
| SELL | retest2 | 2024-07-10 10:15:00 | 556.20 | 2024-07-12 09:15:00 | 565.70 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2024-07-10 11:00:00 | 555.60 | 2024-07-12 09:15:00 | 565.70 | STOP_HIT | 1.00 | -1.82% |
| SELL | retest2 | 2024-08-06 14:30:00 | 500.10 | 2024-08-07 11:15:00 | 518.55 | STOP_HIT | 1.00 | -3.69% |
| SELL | retest2 | 2024-08-06 15:00:00 | 499.10 | 2024-08-07 11:15:00 | 518.55 | STOP_HIT | 1.00 | -3.90% |
| BUY | retest2 | 2024-08-12 09:15:00 | 539.00 | 2024-08-14 13:15:00 | 525.50 | STOP_HIT | 1.00 | -2.50% |
| BUY | retest2 | 2024-08-14 09:45:00 | 531.05 | 2024-08-14 13:15:00 | 525.50 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2024-08-16 15:00:00 | 529.00 | 2024-08-19 09:15:00 | 539.80 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest2 | 2024-08-21 09:15:00 | 543.90 | 2024-08-23 11:15:00 | 533.50 | STOP_HIT | 1.00 | -1.91% |
| BUY | retest2 | 2024-08-22 11:45:00 | 547.70 | 2024-08-23 11:15:00 | 533.50 | STOP_HIT | 1.00 | -2.59% |
| SELL | retest2 | 2024-08-26 12:00:00 | 532.00 | 2024-09-04 11:15:00 | 505.40 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-26 12:00:00 | 532.00 | 2024-09-04 15:15:00 | 509.75 | STOP_HIT | 0.50 | 4.18% |
| SELL | retest2 | 2024-08-28 09:30:00 | 531.35 | 2024-09-06 11:15:00 | 504.78 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-28 09:30:00 | 531.35 | 2024-09-10 09:15:00 | 503.75 | STOP_HIT | 0.50 | 5.19% |
| BUY | retest2 | 2024-10-14 09:15:00 | 581.25 | 2024-10-17 14:15:00 | 583.20 | STOP_HIT | 1.00 | 0.34% |
| SELL | retest2 | 2024-10-29 10:30:00 | 501.00 | 2024-10-30 09:15:00 | 519.85 | STOP_HIT | 1.00 | -3.76% |
| SELL | retest2 | 2024-10-29 13:30:00 | 501.80 | 2024-10-30 09:15:00 | 519.85 | STOP_HIT | 1.00 | -3.60% |
| SELL | retest2 | 2024-11-12 10:00:00 | 510.00 | 2024-11-13 14:15:00 | 484.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-12 10:00:00 | 510.00 | 2024-11-14 09:15:00 | 496.85 | STOP_HIT | 0.50 | 2.58% |
| BUY | retest2 | 2024-11-28 11:30:00 | 493.95 | 2024-12-03 09:15:00 | 543.35 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-18 12:30:00 | 556.10 | 2024-12-19 10:15:00 | 564.40 | STOP_HIT | 1.00 | -1.49% |
| SELL | retest2 | 2024-12-26 13:45:00 | 557.00 | 2024-12-30 09:15:00 | 576.50 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2024-12-26 15:15:00 | 558.10 | 2024-12-30 09:15:00 | 576.50 | STOP_HIT | 1.00 | -3.30% |
| SELL | retest2 | 2024-12-27 09:45:00 | 557.90 | 2024-12-30 09:15:00 | 576.50 | STOP_HIT | 1.00 | -3.33% |
| SELL | retest2 | 2024-12-27 12:00:00 | 558.05 | 2024-12-30 09:15:00 | 576.50 | STOP_HIT | 1.00 | -3.31% |
| SELL | retest2 | 2025-01-24 14:00:00 | 494.65 | 2025-01-24 14:15:00 | 501.30 | STOP_HIT | 1.00 | -1.34% |
| SELL | retest2 | 2025-01-30 13:15:00 | 470.85 | 2025-01-31 15:15:00 | 476.00 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2025-02-01 15:00:00 | 481.00 | 2025-02-03 10:15:00 | 471.05 | STOP_HIT | 1.00 | -2.07% |
| BUY | retest2 | 2025-02-07 11:15:00 | 502.55 | 2025-02-07 13:15:00 | 493.20 | STOP_HIT | 1.00 | -1.86% |
| BUY | retest2 | 2025-02-07 11:45:00 | 503.65 | 2025-02-07 13:15:00 | 493.20 | STOP_HIT | 1.00 | -2.07% |
| SELL | retest2 | 2025-02-20 09:15:00 | 398.60 | 2025-02-20 09:15:00 | 408.75 | STOP_HIT | 1.00 | -2.55% |
| BUY | retest2 | 2025-02-21 14:45:00 | 417.00 | 2025-02-25 10:15:00 | 408.80 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-02-24 09:45:00 | 412.45 | 2025-02-25 10:15:00 | 408.80 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-02-25 09:15:00 | 412.55 | 2025-02-25 10:15:00 | 408.80 | STOP_HIT | 1.00 | -0.91% |
| BUY | retest2 | 2025-03-13 12:00:00 | 439.05 | 2025-03-25 09:15:00 | 482.96 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-13 13:45:00 | 440.15 | 2025-03-25 09:15:00 | 484.17 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-17 12:00:00 | 439.50 | 2025-03-25 09:15:00 | 483.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-17 13:30:00 | 439.00 | 2025-03-25 09:15:00 | 482.90 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-20 11:45:00 | 456.05 | 2025-03-25 09:15:00 | 501.66 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-20 12:15:00 | 456.20 | 2025-03-25 09:15:00 | 501.82 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-21 09:15:00 | 464.05 | 2025-03-25 09:15:00 | 510.46 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-04-02 09:15:00 | 479.70 | 2025-04-02 11:15:00 | 491.90 | STOP_HIT | 1.00 | -2.54% |
| SELL | retest2 | 2025-04-02 11:15:00 | 482.95 | 2025-04-02 11:15:00 | 491.90 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2025-04-09 09:15:00 | 456.00 | 2025-04-15 10:15:00 | 466.10 | STOP_HIT | 1.00 | -2.21% |
| BUY | retest2 | 2025-04-21 11:30:00 | 487.55 | 2025-04-24 09:15:00 | 479.45 | STOP_HIT | 1.00 | -1.66% |
| BUY | retest2 | 2025-04-22 09:15:00 | 491.60 | 2025-04-24 09:15:00 | 479.45 | STOP_HIT | 1.00 | -2.47% |
| BUY | retest2 | 2025-04-22 10:30:00 | 487.85 | 2025-04-24 09:15:00 | 479.45 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-04-22 12:00:00 | 487.50 | 2025-04-24 09:15:00 | 479.45 | STOP_HIT | 1.00 | -1.65% |
| SELL | retest2 | 2025-04-28 14:30:00 | 468.40 | 2025-05-05 12:15:00 | 469.85 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-04-29 09:30:00 | 467.15 | 2025-05-05 12:15:00 | 469.85 | STOP_HIT | 1.00 | -0.58% |
| BUY | retest2 | 2025-05-19 11:45:00 | 560.80 | 2025-05-22 14:15:00 | 530.60 | STOP_HIT | 1.00 | -5.39% |
| BUY | retest2 | 2025-05-19 14:15:00 | 566.50 | 2025-05-22 14:15:00 | 530.60 | STOP_HIT | 1.00 | -6.34% |
| BUY | retest2 | 2025-06-02 11:15:00 | 551.20 | 2025-06-06 09:15:00 | 548.80 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-06-03 09:15:00 | 561.70 | 2025-06-06 09:15:00 | 548.80 | STOP_HIT | 1.00 | -2.30% |
| BUY | retest2 | 2025-06-03 14:00:00 | 551.25 | 2025-06-06 09:15:00 | 548.80 | STOP_HIT | 1.00 | -0.44% |
| BUY | retest2 | 2025-06-04 10:00:00 | 552.40 | 2025-06-06 09:15:00 | 548.80 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-06-20 09:15:00 | 544.90 | 2025-07-02 11:15:00 | 562.60 | STOP_HIT | 1.00 | 3.25% |
| SELL | retest2 | 2025-07-07 10:30:00 | 559.50 | 2025-07-09 14:15:00 | 560.00 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-07-08 09:30:00 | 559.45 | 2025-07-09 14:15:00 | 560.00 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-07-09 12:45:00 | 559.60 | 2025-07-09 14:15:00 | 560.00 | STOP_HIT | 1.00 | -0.07% |
| SELL | retest2 | 2025-07-09 14:15:00 | 558.70 | 2025-07-09 14:15:00 | 560.00 | STOP_HIT | 1.00 | -0.23% |
| BUY | retest2 | 2025-07-14 09:15:00 | 563.95 | 2025-07-17 15:15:00 | 578.00 | STOP_HIT | 1.00 | 2.49% |
| SELL | retest2 | 2025-07-22 10:15:00 | 576.40 | 2025-07-25 09:15:00 | 547.58 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 11:00:00 | 575.95 | 2025-07-25 09:15:00 | 547.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 10:30:00 | 574.85 | 2025-07-25 09:15:00 | 546.11 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-22 10:15:00 | 576.40 | 2025-07-28 09:15:00 | 551.55 | STOP_HIT | 0.50 | 4.31% |
| SELL | retest2 | 2025-07-22 11:00:00 | 575.95 | 2025-07-28 09:15:00 | 551.55 | STOP_HIT | 0.50 | 4.24% |
| SELL | retest2 | 2025-07-24 10:30:00 | 574.85 | 2025-07-28 09:15:00 | 551.55 | STOP_HIT | 0.50 | 4.05% |
| BUY | retest2 | 2025-08-18 09:15:00 | 539.15 | 2025-08-21 13:15:00 | 541.65 | STOP_HIT | 1.00 | 0.46% |
| BUY | retest2 | 2025-09-03 09:45:00 | 527.70 | 2025-09-05 09:15:00 | 517.00 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-09-03 13:00:00 | 527.30 | 2025-09-05 09:15:00 | 517.00 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-09-04 09:45:00 | 531.10 | 2025-09-05 09:15:00 | 517.00 | STOP_HIT | 1.00 | -2.65% |
| BUY | retest2 | 2025-09-04 13:45:00 | 527.15 | 2025-09-05 09:15:00 | 517.00 | STOP_HIT | 1.00 | -1.93% |
| SELL | retest2 | 2025-09-08 12:00:00 | 518.00 | 2025-09-10 09:15:00 | 524.40 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2025-09-09 09:30:00 | 521.00 | 2025-09-10 09:15:00 | 524.40 | STOP_HIT | 1.00 | -0.65% |
| SELL | retest2 | 2025-09-09 11:30:00 | 520.55 | 2025-09-10 09:15:00 | 524.40 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-09-23 10:15:00 | 582.55 | 2025-09-26 09:15:00 | 560.85 | STOP_HIT | 1.00 | -3.73% |
| BUY | retest2 | 2025-09-24 09:45:00 | 577.95 | 2025-09-26 09:15:00 | 560.85 | STOP_HIT | 1.00 | -2.96% |
| BUY | retest2 | 2025-09-24 13:45:00 | 576.45 | 2025-09-26 09:15:00 | 560.85 | STOP_HIT | 1.00 | -2.71% |
| BUY | retest2 | 2025-09-25 09:45:00 | 578.40 | 2025-09-26 09:15:00 | 560.85 | STOP_HIT | 1.00 | -3.03% |
| SELL | retest2 | 2025-10-03 11:15:00 | 546.00 | 2025-10-03 14:15:00 | 553.90 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-10-20 10:45:00 | 549.70 | 2025-10-23 10:15:00 | 556.80 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-10-20 11:45:00 | 549.15 | 2025-10-23 10:15:00 | 556.80 | STOP_HIT | 1.00 | -1.39% |
| SELL | retest2 | 2025-10-20 12:45:00 | 549.40 | 2025-10-23 10:15:00 | 556.80 | STOP_HIT | 1.00 | -1.35% |
| SELL | retest2 | 2025-10-20 13:30:00 | 549.65 | 2025-10-23 10:15:00 | 556.80 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-10-27 09:45:00 | 566.60 | 2025-10-29 10:15:00 | 623.26 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-10-28 09:15:00 | 567.75 | 2025-10-29 10:15:00 | 624.53 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-11-11 13:30:00 | 568.95 | 2025-11-13 14:15:00 | 571.65 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-11-11 15:00:00 | 568.65 | 2025-11-13 14:15:00 | 571.65 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-11-12 09:15:00 | 565.20 | 2025-11-13 14:15:00 | 571.65 | STOP_HIT | 1.00 | -1.14% |
| SELL | retest2 | 2025-11-12 14:15:00 | 568.25 | 2025-11-13 14:15:00 | 571.65 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-11-18 11:15:00 | 580.15 | 2025-11-19 14:15:00 | 569.55 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2025-12-01 11:45:00 | 552.10 | 2025-12-08 13:15:00 | 524.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-01 11:45:00 | 552.10 | 2025-12-09 10:15:00 | 530.45 | STOP_HIT | 0.50 | 3.92% |
| BUY | retest2 | 2025-12-11 09:15:00 | 545.55 | 2025-12-12 12:15:00 | 539.50 | STOP_HIT | 1.00 | -1.11% |
| BUY | retest2 | 2025-12-11 10:00:00 | 546.15 | 2025-12-12 12:15:00 | 539.50 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2025-12-11 12:15:00 | 545.50 | 2025-12-12 12:15:00 | 539.50 | STOP_HIT | 1.00 | -1.10% |
| BUY | retest2 | 2025-12-26 10:30:00 | 582.35 | 2025-12-31 10:15:00 | 640.59 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-26 12:00:00 | 582.20 | 2025-12-31 10:15:00 | 640.42 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-12-26 12:45:00 | 581.00 | 2025-12-31 10:15:00 | 639.10 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-01-19 11:30:00 | 647.35 | 2026-01-20 11:15:00 | 632.25 | STOP_HIT | 1.00 | -2.33% |
| BUY | retest2 | 2026-01-19 12:15:00 | 647.25 | 2026-01-20 11:15:00 | 632.25 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2026-01-27 14:00:00 | 620.00 | 2026-01-27 15:15:00 | 633.00 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2026-02-18 09:15:00 | 685.90 | 2026-02-24 12:15:00 | 680.85 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2026-03-05 11:45:00 | 680.25 | 2026-03-09 09:15:00 | 646.24 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 12:30:00 | 683.40 | 2026-03-09 09:15:00 | 649.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 13:00:00 | 683.55 | 2026-03-09 09:15:00 | 649.37 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 13:30:00 | 682.35 | 2026-03-09 09:15:00 | 648.23 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 11:45:00 | 678.70 | 2026-03-09 09:15:00 | 644.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 13:00:00 | 678.90 | 2026-03-09 09:15:00 | 644.95 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-06 13:30:00 | 678.05 | 2026-03-09 09:15:00 | 644.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-05 11:45:00 | 680.25 | 2026-03-10 14:15:00 | 636.00 | STOP_HIT | 0.50 | 6.50% |
| SELL | retest2 | 2026-03-05 12:30:00 | 683.40 | 2026-03-10 14:15:00 | 636.00 | STOP_HIT | 0.50 | 6.94% |
| SELL | retest2 | 2026-03-05 13:00:00 | 683.55 | 2026-03-10 14:15:00 | 636.00 | STOP_HIT | 0.50 | 6.96% |
| SELL | retest2 | 2026-03-05 13:30:00 | 682.35 | 2026-03-10 14:15:00 | 636.00 | STOP_HIT | 0.50 | 6.79% |
| SELL | retest2 | 2026-03-06 11:45:00 | 678.70 | 2026-03-10 14:15:00 | 636.00 | STOP_HIT | 0.50 | 6.29% |
| SELL | retest2 | 2026-03-06 13:00:00 | 678.90 | 2026-03-10 14:15:00 | 636.00 | STOP_HIT | 0.50 | 6.32% |
| SELL | retest2 | 2026-03-06 13:30:00 | 678.05 | 2026-03-10 14:15:00 | 636.00 | STOP_HIT | 0.50 | 6.20% |
| SELL | retest2 | 2026-03-17 12:15:00 | 610.85 | 2026-03-18 09:15:00 | 625.20 | STOP_HIT | 1.00 | -2.35% |
| SELL | retest2 | 2026-03-17 12:45:00 | 611.55 | 2026-03-18 09:15:00 | 625.20 | STOP_HIT | 1.00 | -2.23% |
| SELL | retest2 | 2026-03-17 13:15:00 | 611.20 | 2026-03-18 09:15:00 | 625.20 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2026-04-01 09:15:00 | 655.60 | 2026-04-07 15:15:00 | 629.00 | STOP_HIT | 1.00 | -4.06% |
| BUY | retest2 | 2026-04-02 09:45:00 | 629.75 | 2026-04-07 15:15:00 | 629.00 | STOP_HIT | 1.00 | -0.12% |
| BUY | retest2 | 2026-04-02 12:30:00 | 632.60 | 2026-04-07 15:15:00 | 629.00 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2026-04-06 09:30:00 | 641.00 | 2026-04-07 15:15:00 | 629.00 | STOP_HIT | 1.00 | -1.87% |
| BUY | retest2 | 2026-04-09 09:15:00 | 653.80 | 2026-04-09 12:15:00 | 638.35 | STOP_HIT | 1.00 | -2.36% |
| BUY | retest2 | 2026-04-10 09:30:00 | 646.30 | 2026-04-10 12:15:00 | 639.00 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2026-04-10 11:00:00 | 644.85 | 2026-04-10 12:15:00 | 639.00 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2026-04-13 13:00:00 | 637.00 | 2026-04-15 09:15:00 | 663.40 | STOP_HIT | 1.00 | -4.14% |
| SELL | retest2 | 2026-04-13 15:15:00 | 637.90 | 2026-04-15 09:15:00 | 663.40 | STOP_HIT | 1.00 | -4.00% |
| BUY | retest1 | 2026-04-20 09:15:00 | 686.00 | 2026-04-20 10:15:00 | 754.60 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest1 | 2026-04-20 09:45:00 | 684.80 | 2026-04-20 10:15:00 | 753.28 | TARGET_HIT | 1.00 | 10.00% |
