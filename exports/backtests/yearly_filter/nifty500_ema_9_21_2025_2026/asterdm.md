# Aster DM Healthcare Ltd. (ASTERDM)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 742.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 78 |
| ALERT1 | 47 |
| ALERT2 | 46 |
| ALERT2_SKIP | 26 |
| ALERT3 | 120 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 1 |
| ENTRY2 | 66 |
| PARTIAL | 4 |
| TARGET_HIT | 1 |
| STOP_HIT | 67 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 71 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 17 / 54
- **Target hits / Stop hits / Partials:** 0 / 67 / 4
- **Avg / median % per leg:** -0.27% / -0.66%
- **Sum % (uncompounded):** -19.20%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 31 | 6 | 19.4% | 0 | 31 | 0 | -0.30% | -9.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 31 | 6 | 19.4% | 0 | 31 | 0 | -0.30% | -9.4% |
| SELL (all) | 40 | 11 | 27.5% | 0 | 36 | 4 | -0.25% | -9.8% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.62% | -1.6% |
| SELL @ 3rd Alert (retest2) | 39 | 11 | 28.2% | 0 | 35 | 4 | -0.21% | -8.2% |
| retest1 (combined) | 1 | 0 | 0.0% | 0 | 1 | 0 | -1.62% | -1.6% |
| retest2 (combined) | 70 | 17 | 24.3% | 0 | 66 | 4 | -0.25% | -17.6% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2025-05-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 15:15:00 | 577.45 | 580.33 | 580.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-21 09:15:00 | 567.00 | 577.66 | 579.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 10:15:00 | 569.00 | 561.77 | 567.69 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-22 10:15:00 | 569.00 | 561.77 | 567.69 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 10:15:00 | 569.00 | 561.77 | 567.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 11:00:00 | 569.00 | 561.77 | 567.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 11:15:00 | 567.70 | 562.96 | 567.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 11:00:00 | 562.70 | 566.52 | 567.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 11:30:00 | 560.65 | 564.81 | 566.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-30 10:15:00 | 563.95 | 548.27 | 547.60 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — BUY (started 2025-05-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 10:15:00 | 563.95 | 548.27 | 547.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 10:15:00 | 580.20 | 559.84 | 555.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 10:15:00 | 582.35 | 583.11 | 576.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 11:00:00 | 582.35 | 583.11 | 576.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 577.50 | 581.14 | 577.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 577.50 | 581.14 | 577.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 577.50 | 580.41 | 577.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 573.40 | 580.41 | 577.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 572.00 | 578.73 | 577.05 | EMA400 retest candle locked (from upside) |

### Cycle 3 — SELL (started 2025-06-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-09 11:15:00 | 569.00 | 575.67 | 575.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-11 09:15:00 | 565.80 | 569.34 | 571.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 10:15:00 | 571.00 | 569.67 | 571.35 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 10:15:00 | 571.00 | 569.67 | 571.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 10:15:00 | 571.00 | 569.67 | 571.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 11:00:00 | 571.00 | 569.67 | 571.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 11:15:00 | 571.55 | 570.05 | 571.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 12:00:00 | 571.55 | 570.05 | 571.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 12:15:00 | 574.10 | 570.86 | 571.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 14:00:00 | 570.30 | 570.75 | 571.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 15:00:00 | 569.80 | 570.56 | 571.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 09:15:00 | 578.05 | 571.93 | 571.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — BUY (started 2025-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-12 09:15:00 | 578.05 | 571.93 | 571.82 | EMA200 above EMA400 |

### Cycle 5 — SELL (started 2025-06-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 13:15:00 | 561.50 | 570.41 | 571.30 | EMA200 below EMA400 |

### Cycle 6 — BUY (started 2025-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-16 09:15:00 | 580.00 | 570.24 | 569.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-16 10:15:00 | 589.30 | 574.05 | 571.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-17 09:15:00 | 572.95 | 581.03 | 577.19 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-17 09:15:00 | 572.95 | 581.03 | 577.19 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 572.95 | 581.03 | 577.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 10:00:00 | 572.95 | 581.03 | 577.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 571.20 | 579.07 | 576.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-17 11:00:00 | 571.20 | 579.07 | 576.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — SELL (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 12:15:00 | 566.35 | 574.35 | 574.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-20 09:15:00 | 563.60 | 567.92 | 569.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-23 09:15:00 | 562.40 | 562.28 | 565.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-23 10:15:00 | 563.70 | 562.28 | 565.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 10:15:00 | 569.85 | 563.79 | 565.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 10:45:00 | 570.65 | 563.79 | 565.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 11:15:00 | 569.90 | 565.01 | 565.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-23 11:30:00 | 571.20 | 565.01 | 565.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 8 — BUY (started 2025-06-23 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 13:15:00 | 570.25 | 566.78 | 566.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 14:15:00 | 573.25 | 568.07 | 567.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 09:15:00 | 567.50 | 568.91 | 567.78 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-24 09:15:00 | 567.50 | 568.91 | 567.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-24 09:15:00 | 567.50 | 568.91 | 567.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-25 10:30:00 | 579.90 | 574.47 | 571.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 15:15:00 | 588.00 | 590.07 | 590.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — SELL (started 2025-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 15:15:00 | 588.00 | 590.07 | 590.08 | EMA200 below EMA400 |

### Cycle 10 — BUY (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-02 10:15:00 | 592.60 | 590.48 | 590.26 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-02 11:15:00 | 600.20 | 592.43 | 591.16 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-02 14:15:00 | 592.40 | 594.23 | 592.46 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 14:15:00 | 592.40 | 594.23 | 592.46 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 14:15:00 | 592.40 | 594.23 | 592.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-02 15:00:00 | 592.40 | 594.23 | 592.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 15:15:00 | 594.00 | 594.18 | 592.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-03 09:15:00 | 620.15 | 594.18 | 592.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-09 10:15:00 | 615.55 | 622.41 | 623.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — SELL (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-09 10:15:00 | 615.55 | 622.41 | 623.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-09 13:15:00 | 612.50 | 618.35 | 620.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-11 11:15:00 | 605.40 | 603.83 | 608.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-11 12:00:00 | 605.40 | 603.83 | 608.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 13:15:00 | 609.20 | 605.39 | 608.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 13:45:00 | 608.50 | 605.39 | 608.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 14:15:00 | 608.55 | 606.03 | 608.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-11 14:45:00 | 610.00 | 606.03 | 608.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 15:15:00 | 605.50 | 605.92 | 608.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-14 09:15:00 | 601.90 | 605.92 | 608.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-14 09:15:00 | 604.25 | 605.59 | 607.93 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 13:30:00 | 597.80 | 602.10 | 604.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-15 15:15:00 | 598.50 | 601.63 | 603.72 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 598.85 | 600.69 | 602.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-17 13:00:00 | 597.30 | 599.88 | 601.20 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 09:15:00 | 593.75 | 596.16 | 598.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 10:15:00 | 592.10 | 596.16 | 598.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-18 13:30:00 | 593.65 | 595.37 | 597.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-21 09:15:00 | 587.25 | 595.00 | 596.98 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-22 12:15:00 | 598.55 | 594.47 | 594.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 12 — BUY (started 2025-07-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-22 12:15:00 | 598.55 | 594.47 | 594.06 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-23 10:15:00 | 600.60 | 597.97 | 596.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-23 12:15:00 | 594.95 | 598.94 | 596.99 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-23 12:15:00 | 594.95 | 598.94 | 596.99 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 594.95 | 598.94 | 596.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:00:00 | 594.95 | 598.94 | 596.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 595.65 | 598.28 | 596.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:15:00 | 595.30 | 598.28 | 596.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 598.40 | 598.30 | 597.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 14:30:00 | 595.95 | 598.30 | 597.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 593.95 | 597.46 | 596.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:30:00 | 594.60 | 597.46 | 596.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 10:15:00 | 596.90 | 597.35 | 596.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-24 10:45:00 | 598.00 | 597.35 | 596.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 11:15:00 | 598.30 | 597.54 | 596.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 12:15:00 | 599.45 | 597.54 | 596.99 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 12:45:00 | 599.40 | 598.03 | 597.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 14:00:00 | 599.55 | 598.34 | 597.47 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 14:15:00 | 595.40 | 597.75 | 597.28 | SL hit (close<static) qty=1.00 sl=596.05 alert=retest2 |

### Cycle 13 — SELL (started 2025-07-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-24 15:15:00 | 593.45 | 596.89 | 596.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 09:15:00 | 590.40 | 595.59 | 596.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 583.70 | 580.39 | 584.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-29 09:15:00 | 583.70 | 580.39 | 584.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 09:15:00 | 583.70 | 580.39 | 584.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 10:00:00 | 583.70 | 580.39 | 584.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 10:15:00 | 581.60 | 580.63 | 584.36 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2025-07-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-29 15:15:00 | 591.80 | 585.97 | 585.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-30 09:15:00 | 594.15 | 587.60 | 586.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 09:15:00 | 596.40 | 601.82 | 597.63 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-01 09:15:00 | 596.40 | 601.82 | 597.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 09:15:00 | 596.40 | 601.82 | 597.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 10:00:00 | 596.40 | 601.82 | 597.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 10:15:00 | 597.05 | 600.87 | 597.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 11:15:00 | 596.40 | 600.87 | 597.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 11:15:00 | 593.00 | 599.30 | 597.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-01 12:00:00 | 593.00 | 599.30 | 597.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 12:15:00 | 596.15 | 598.67 | 597.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-01 14:30:00 | 598.15 | 597.11 | 596.61 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-04 09:15:00 | 586.00 | 594.55 | 595.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2025-08-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-04 09:15:00 | 586.00 | 594.55 | 595.51 | EMA200 below EMA400 |

### Cycle 16 — BUY (started 2025-08-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-05 12:15:00 | 599.60 | 594.72 | 594.11 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 10:15:00 | 592.70 | 594.72 | 594.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-07 12:15:00 | 590.75 | 593.65 | 594.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 594.20 | 593.53 | 594.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 594.20 | 593.53 | 594.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 594.20 | 593.53 | 594.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 14:45:00 | 594.50 | 593.53 | 594.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 596.00 | 594.02 | 594.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 590.00 | 594.02 | 594.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 590.00 | 593.22 | 593.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 11:45:00 | 588.00 | 591.75 | 593.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 15:15:00 | 587.00 | 585.29 | 587.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-12 12:15:00 | 598.45 | 589.97 | 588.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — BUY (started 2025-08-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-12 12:15:00 | 598.45 | 589.97 | 588.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 13:15:00 | 600.55 | 592.09 | 589.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-14 12:15:00 | 608.20 | 608.57 | 603.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-14 13:00:00 | 608.20 | 608.57 | 603.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-14 15:15:00 | 606.40 | 607.59 | 604.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 612.40 | 607.59 | 604.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-21 09:15:00 | 614.10 | 615.25 | 615.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — SELL (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 09:15:00 | 614.10 | 615.25 | 615.34 | EMA200 below EMA400 |

### Cycle 20 — BUY (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 10:15:00 | 616.30 | 615.46 | 615.42 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2025-08-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 11:15:00 | 610.25 | 614.42 | 614.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 12:15:00 | 606.05 | 612.75 | 614.14 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 09:15:00 | 608.90 | 605.79 | 608.21 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 09:15:00 | 608.90 | 605.79 | 608.21 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 09:15:00 | 608.90 | 605.79 | 608.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 09:30:00 | 605.70 | 605.79 | 608.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 608.40 | 606.31 | 608.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-25 12:45:00 | 604.95 | 606.53 | 608.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 10:30:00 | 601.15 | 604.79 | 606.45 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 12:30:00 | 605.15 | 605.16 | 606.35 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-29 09:15:00 | 574.70 | 597.43 | 600.49 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-29 09:15:00 | 597.55 | 597.43 | 600.49 | SL hit (close>static) qty=0.50 sl=597.43 alert=retest2 |

### Cycle 22 — BUY (started 2025-09-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 09:15:00 | 611.60 | 603.28 | 602.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-02 11:15:00 | 618.75 | 609.70 | 606.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 617.30 | 618.23 | 612.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 10:30:00 | 617.00 | 618.23 | 612.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 635.30 | 637.68 | 630.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-05 12:30:00 | 630.90 | 637.68 | 630.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 637.95 | 638.09 | 632.90 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 14:15:00 | 645.00 | 638.90 | 634.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-08 14:45:00 | 642.55 | 639.14 | 635.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 642.55 | 639.26 | 637.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 09:15:00 | 642.25 | 641.70 | 640.05 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 09:15:00 | 640.35 | 641.43 | 640.08 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-09-11 14:15:00 | 636.60 | 639.78 | 639.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 23 — SELL (started 2025-09-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-11 14:15:00 | 636.60 | 639.78 | 639.80 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2025-09-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-11 15:15:00 | 640.00 | 639.83 | 639.82 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2025-09-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-12 09:15:00 | 632.00 | 638.26 | 639.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-12 11:15:00 | 627.60 | 635.14 | 637.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-12 14:15:00 | 632.35 | 631.91 | 635.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-12 15:00:00 | 632.35 | 631.91 | 635.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 613.90 | 613.83 | 617.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 11:30:00 | 613.00 | 613.83 | 617.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 15:15:00 | 617.20 | 614.05 | 616.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:15:00 | 621.45 | 614.05 | 616.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 09:15:00 | 629.00 | 617.04 | 617.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-19 09:45:00 | 627.70 | 617.04 | 617.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — BUY (started 2025-09-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-19 10:15:00 | 631.40 | 619.91 | 618.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-19 13:15:00 | 637.80 | 626.88 | 622.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 15:15:00 | 646.95 | 647.08 | 638.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-23 09:15:00 | 647.15 | 647.08 | 638.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 639.15 | 644.49 | 638.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 639.15 | 644.49 | 638.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 11:15:00 | 636.30 | 642.86 | 638.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:30:00 | 635.95 | 642.86 | 638.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 12:15:00 | 637.35 | 641.75 | 638.46 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-23 15:15:00 | 639.60 | 639.99 | 638.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-24 10:00:00 | 641.80 | 640.29 | 638.63 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-24 11:15:00 | 635.00 | 638.87 | 638.25 | SL hit (close<static) qty=1.00 sl=636.10 alert=retest2 |

### Cycle 27 — SELL (started 2025-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 12:15:00 | 633.70 | 637.83 | 637.84 | EMA200 below EMA400 |

### Cycle 28 — BUY (started 2025-09-25 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-25 11:15:00 | 646.65 | 637.59 | 637.30 | EMA200 above EMA400 |

### Cycle 29 — SELL (started 2025-09-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-26 10:15:00 | 629.30 | 636.47 | 637.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 623.65 | 633.91 | 636.12 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 639.00 | 631.63 | 633.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-29 09:15:00 | 639.00 | 631.63 | 633.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 09:15:00 | 639.00 | 631.63 | 633.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-29 09:45:00 | 640.80 | 631.63 | 633.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-29 10:15:00 | 629.00 | 631.11 | 633.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 11:15:00 | 626.70 | 631.11 | 633.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 13:15:00 | 627.70 | 629.98 | 632.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 14:30:00 | 626.65 | 628.82 | 631.50 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 14:45:00 | 627.40 | 627.03 | 629.08 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 15:15:00 | 626.00 | 626.83 | 628.80 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-03 09:15:00 | 630.45 | 629.23 | 629.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 30 — BUY (started 2025-10-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 09:15:00 | 630.45 | 629.23 | 629.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 12:15:00 | 639.70 | 632.14 | 630.51 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 13:15:00 | 663.55 | 667.57 | 657.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 14:00:00 | 663.55 | 667.57 | 657.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 15:15:00 | 658.80 | 665.10 | 658.36 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 09:15:00 | 667.25 | 665.10 | 658.36 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-14 15:15:00 | 685.00 | 691.61 | 692.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 31 — SELL (started 2025-10-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 15:15:00 | 685.00 | 691.61 | 692.27 | EMA200 below EMA400 |

### Cycle 32 — BUY (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 13:15:00 | 695.35 | 692.55 | 692.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-15 14:15:00 | 699.00 | 693.84 | 692.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-16 13:15:00 | 696.45 | 697.05 | 695.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-16 13:45:00 | 696.95 | 697.05 | 695.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 700.70 | 717.61 | 715.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:00:00 | 700.70 | 717.61 | 715.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 10:15:00 | 701.15 | 714.32 | 714.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 10:45:00 | 700.25 | 714.32 | 714.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 33 — SELL (started 2025-10-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 11:15:00 | 703.60 | 712.17 | 713.14 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-27 14:15:00 | 696.25 | 703.24 | 707.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-28 14:15:00 | 699.90 | 699.37 | 702.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-28 15:00:00 | 699.90 | 699.37 | 702.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 696.15 | 698.31 | 701.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 10:15:00 | 693.95 | 698.72 | 700.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-30 13:45:00 | 694.75 | 697.58 | 699.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-31 09:15:00 | 678.85 | 697.75 | 699.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 10:45:00 | 691.20 | 686.36 | 690.23 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 11:15:00 | 692.00 | 687.49 | 690.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-03 11:45:00 | 692.50 | 687.49 | 690.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 12:15:00 | 683.55 | 686.70 | 689.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-03 13:15:00 | 681.85 | 686.70 | 689.77 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-04 09:30:00 | 680.10 | 682.79 | 686.71 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-06 10:15:00 | 691.80 | 687.58 | 687.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 34 — BUY (started 2025-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-06 10:15:00 | 691.80 | 687.58 | 687.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-06 12:15:00 | 697.50 | 690.20 | 688.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-06 14:15:00 | 686.65 | 690.39 | 688.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 14:15:00 | 686.65 | 690.39 | 688.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 14:15:00 | 686.65 | 690.39 | 688.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-06 15:00:00 | 686.65 | 690.39 | 688.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 15:15:00 | 689.00 | 690.11 | 688.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-07 09:15:00 | 679.45 | 690.11 | 688.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 09:15:00 | 681.45 | 688.38 | 688.26 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-07 10:45:00 | 694.15 | 689.39 | 688.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 09:15:00 | 680.00 | 689.25 | 689.31 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — SELL (started 2025-11-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 09:15:00 | 680.00 | 689.25 | 689.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-10 12:15:00 | 673.25 | 683.40 | 686.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-10 15:15:00 | 682.80 | 682.21 | 684.98 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-11-11 09:15:00 | 672.35 | 682.21 | 684.98 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 14:15:00 | 679.55 | 679.01 | 681.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-11 15:00:00 | 679.55 | 679.01 | 681.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 09:15:00 | 683.25 | 679.54 | 681.54 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-12 09:15:00 | 683.25 | 679.54 | 681.54 | SL hit (close>ema400) qty=1.00 sl=681.54 alert=retest1 |

### Cycle 36 — BUY (started 2025-11-12 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 13:15:00 | 685.00 | 682.66 | 682.58 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2025-11-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-12 15:15:00 | 678.00 | 681.69 | 682.15 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 09:15:00 | 675.20 | 680.39 | 681.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-13 10:15:00 | 681.90 | 680.69 | 681.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 10:15:00 | 681.90 | 680.69 | 681.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 681.90 | 680.69 | 681.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 681.90 | 680.69 | 681.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 678.40 | 680.24 | 681.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 14:45:00 | 675.30 | 678.24 | 680.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 09:45:00 | 676.80 | 677.66 | 679.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 10:30:00 | 676.05 | 677.87 | 679.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 12:45:00 | 676.40 | 677.89 | 679.12 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 677.40 | 677.79 | 678.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:30:00 | 678.20 | 677.79 | 678.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 678.35 | 677.90 | 678.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 14:45:00 | 678.80 | 677.90 | 678.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 678.10 | 677.94 | 678.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 09:15:00 | 673.35 | 677.94 | 678.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 11:15:00 | 674.65 | 677.00 | 678.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 12:45:00 | 675.90 | 676.19 | 677.60 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 13:15:00 | 680.50 | 677.05 | 677.87 | SL hit (close>static) qty=1.00 sl=679.00 alert=retest2 |

### Cycle 38 — BUY (started 2025-11-18 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 09:15:00 | 697.25 | 681.93 | 679.94 | EMA200 above EMA400 |

### Cycle 39 — SELL (started 2025-11-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-19 11:15:00 | 672.70 | 680.74 | 681.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 13:15:00 | 669.00 | 676.98 | 679.59 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-20 09:15:00 | 671.80 | 671.58 | 676.08 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-20 10:00:00 | 671.80 | 671.58 | 676.08 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 669.10 | 659.83 | 664.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 10:00:00 | 669.10 | 659.83 | 664.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 10:15:00 | 669.50 | 661.77 | 665.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 11:00:00 | 669.50 | 661.77 | 665.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 656.80 | 664.17 | 665.40 | EMA400 retest candle locked (from downside) |

### Cycle 40 — BUY (started 2025-11-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 09:15:00 | 665.00 | 662.76 | 662.69 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2025-11-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 10:15:00 | 661.85 | 662.58 | 662.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-27 11:15:00 | 660.60 | 662.18 | 662.43 | Break + close below crossover candle low |

### Cycle 42 — BUY (started 2025-11-27 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-27 12:15:00 | 664.40 | 662.62 | 662.61 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 13:15:00 | 668.85 | 663.87 | 663.18 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-28 09:15:00 | 662.25 | 663.95 | 663.43 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-28 09:15:00 | 662.25 | 663.95 | 663.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-28 09:15:00 | 662.25 | 663.95 | 663.43 | EMA400 retest candle locked (from upside) |

### Cycle 43 — SELL (started 2025-11-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-28 11:15:00 | 660.35 | 662.66 | 662.90 | EMA200 below EMA400 |

### Cycle 44 — BUY (started 2025-11-28 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-28 14:15:00 | 666.10 | 662.98 | 662.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-01 09:15:00 | 669.80 | 664.68 | 663.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 11:15:00 | 663.05 | 664.96 | 664.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 11:15:00 | 663.05 | 664.96 | 664.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 11:15:00 | 663.05 | 664.96 | 664.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 12:00:00 | 663.05 | 664.96 | 664.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 12:15:00 | 665.15 | 665.00 | 664.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-01 13:30:00 | 666.85 | 666.00 | 664.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-02 09:15:00 | 658.10 | 665.20 | 664.75 | SL hit (close<static) qty=1.00 sl=662.00 alert=retest2 |

### Cycle 45 — SELL (started 2025-12-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 10:15:00 | 660.10 | 664.18 | 664.33 | EMA200 below EMA400 |

### Cycle 46 — BUY (started 2025-12-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 15:15:00 | 667.50 | 664.03 | 664.02 | EMA200 above EMA400 |

### Cycle 47 — SELL (started 2025-12-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 09:15:00 | 660.65 | 663.35 | 663.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 10:15:00 | 659.95 | 662.67 | 663.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 11:15:00 | 657.45 | 656.01 | 658.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-04 12:00:00 | 657.45 | 656.01 | 658.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 12:15:00 | 650.75 | 654.96 | 657.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-05 09:30:00 | 648.80 | 653.03 | 656.05 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-12-09 09:15:00 | 616.36 | 623.00 | 633.53 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-09 14:15:00 | 622.00 | 621.27 | 628.47 | SL hit (close>ema200) qty=0.50 sl=621.27 alert=retest2 |

### Cycle 48 — BUY (started 2025-12-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 09:15:00 | 610.70 | 599.73 | 598.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 11:15:00 | 616.05 | 604.89 | 601.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-29 09:15:00 | 607.10 | 612.59 | 607.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-29 09:15:00 | 607.10 | 612.59 | 607.30 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 607.10 | 612.59 | 607.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 607.10 | 612.59 | 607.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 608.25 | 611.72 | 607.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 13:00:00 | 613.30 | 611.95 | 608.24 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-29 13:45:00 | 614.00 | 612.48 | 608.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-30 09:15:00 | 605.85 | 610.73 | 608.87 | SL hit (close<static) qty=1.00 sl=606.55 alert=retest2 |

### Cycle 49 — SELL (started 2025-12-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 11:15:00 | 600.00 | 607.71 | 607.77 | EMA200 below EMA400 |

### Cycle 50 — BUY (started 2025-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-30 13:15:00 | 615.85 | 608.10 | 607.86 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2026-01-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 12:15:00 | 604.55 | 610.02 | 610.19 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2026-01-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 13:15:00 | 612.00 | 609.51 | 609.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 14:15:00 | 614.60 | 610.53 | 609.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 10:15:00 | 618.00 | 620.44 | 617.52 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-07 11:00:00 | 618.00 | 620.44 | 617.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 618.05 | 619.96 | 617.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:45:00 | 616.40 | 619.96 | 617.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 620.60 | 620.09 | 617.85 | EMA400 retest candle locked (from upside) |

### Cycle 53 — SELL (started 2026-01-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 12:15:00 | 615.35 | 617.36 | 617.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 09:15:00 | 613.30 | 615.89 | 616.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 09:15:00 | 605.10 | 603.59 | 607.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-13 09:15:00 | 605.10 | 603.59 | 607.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 605.10 | 603.59 | 607.82 | EMA400 retest candle locked (from downside) |

### Cycle 54 — BUY (started 2026-01-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 13:15:00 | 609.65 | 607.60 | 607.60 | EMA200 above EMA400 |

### Cycle 55 — SELL (started 2026-01-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-14 15:15:00 | 606.90 | 607.61 | 607.62 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-16 10:15:00 | 600.60 | 605.83 | 606.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 560.30 | 555.84 | 566.85 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 12:45:00 | 559.25 | 555.84 | 566.85 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 14:15:00 | 568.25 | 558.99 | 566.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-21 15:00:00 | 568.25 | 558.99 | 566.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-21 15:15:00 | 567.20 | 560.63 | 566.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 09:15:00 | 571.75 | 560.63 | 566.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 10:15:00 | 572.60 | 564.52 | 567.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-22 11:00:00 | 572.60 | 564.52 | 567.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 11:15:00 | 570.35 | 565.68 | 567.60 | EMA400 retest candle locked (from downside) |

### Cycle 56 — BUY (started 2026-01-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 15:15:00 | 574.90 | 569.69 | 569.07 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-23 10:15:00 | 565.05 | 568.33 | 568.53 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-23 11:15:00 | 561.30 | 566.93 | 567.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-28 09:15:00 | 557.75 | 556.48 | 559.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 10:00:00 | 557.75 | 556.48 | 559.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 10:15:00 | 548.30 | 542.91 | 547.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 11:00:00 | 548.30 | 542.91 | 547.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 11:15:00 | 545.55 | 543.44 | 546.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:15:00 | 550.25 | 543.44 | 546.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 12:15:00 | 550.00 | 544.75 | 547.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 12:45:00 | 550.95 | 544.75 | 547.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 13:15:00 | 551.95 | 546.19 | 547.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 14:00:00 | 551.95 | 546.19 | 547.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2026-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-01 09:15:00 | 558.50 | 550.08 | 549.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-01 10:15:00 | 566.55 | 553.38 | 550.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-02 09:15:00 | 553.00 | 561.58 | 557.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-02 09:15:00 | 553.00 | 561.58 | 557.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 09:15:00 | 553.00 | 561.58 | 557.15 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-03 10:45:00 | 563.50 | 557.51 | 556.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 14:15:00 | 546.30 | 554.51 | 555.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — SELL (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-03 14:15:00 | 546.30 | 554.51 | 555.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-03 15:15:00 | 544.55 | 552.51 | 554.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-05 14:15:00 | 538.30 | 536.81 | 541.33 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-05 15:00:00 | 538.30 | 536.81 | 541.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 549.80 | 539.36 | 541.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 09:45:00 | 552.95 | 539.36 | 541.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 558.00 | 543.09 | 543.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:00:00 | 558.00 | 543.09 | 543.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — BUY (started 2026-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 11:15:00 | 560.00 | 546.47 | 544.71 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-06 13:15:00 | 562.00 | 551.75 | 547.55 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-13 09:15:00 | 592.45 | 599.52 | 593.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-13 09:15:00 | 592.45 | 599.52 | 593.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 09:15:00 | 592.45 | 599.52 | 593.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-13 10:00:00 | 592.45 | 599.52 | 593.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 10:15:00 | 597.10 | 599.04 | 593.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 10:00:00 | 607.85 | 601.64 | 597.19 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-16 10:30:00 | 608.00 | 604.28 | 598.80 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 12:15:00 | 640.15 | 645.10 | 645.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 61 — SELL (started 2026-03-02 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 12:15:00 | 640.15 | 645.10 | 645.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 636.40 | 643.11 | 644.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 11:15:00 | 645.30 | 643.18 | 644.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-04 11:15:00 | 645.30 | 643.18 | 644.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 11:15:00 | 645.30 | 643.18 | 644.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 11:45:00 | 645.45 | 643.18 | 644.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-04 12:15:00 | 649.80 | 644.50 | 644.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-04 12:45:00 | 648.35 | 644.50 | 644.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 62 — BUY (started 2026-03-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-04 13:15:00 | 646.00 | 644.80 | 644.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 09:15:00 | 656.05 | 648.49 | 646.59 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-09 09:15:00 | 643.45 | 665.56 | 661.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-09 09:15:00 | 643.45 | 665.56 | 661.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 09:15:00 | 643.45 | 665.56 | 661.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:00:00 | 643.45 | 665.56 | 661.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 10:15:00 | 643.70 | 661.19 | 659.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 10:30:00 | 637.35 | 661.19 | 659.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-09 12:15:00 | 660.95 | 659.83 | 659.27 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 09:15:00 | 665.05 | 660.07 | 659.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-10 11:00:00 | 667.60 | 662.57 | 660.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-13 12:15:00 | 666.50 | 675.20 | 676.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2026-03-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 12:15:00 | 666.50 | 675.20 | 676.17 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-16 09:15:00 | 651.00 | 667.48 | 672.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-17 09:15:00 | 670.10 | 655.14 | 661.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-17 09:15:00 | 670.10 | 655.14 | 661.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 09:15:00 | 670.10 | 655.14 | 661.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 10:00:00 | 670.10 | 655.14 | 661.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 10:15:00 | 672.25 | 658.57 | 662.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 11:00:00 | 672.25 | 658.57 | 662.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 12:15:00 | 664.65 | 660.64 | 662.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 12:45:00 | 665.45 | 660.64 | 662.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 671.05 | 662.72 | 663.28 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:00:00 | 671.05 | 662.72 | 663.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 64 — BUY (started 2026-03-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-17 14:15:00 | 668.90 | 663.96 | 663.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-17 15:15:00 | 672.70 | 665.70 | 664.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-18 13:15:00 | 671.85 | 672.21 | 668.88 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-18 13:30:00 | 674.85 | 672.21 | 668.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 14:15:00 | 664.00 | 670.57 | 668.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-18 15:00:00 | 664.00 | 670.57 | 668.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 15:15:00 | 659.00 | 668.26 | 667.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 09:15:00 | 645.95 | 668.26 | 667.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 65 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 644.70 | 663.54 | 665.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 10:15:00 | 639.45 | 658.73 | 663.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-23 15:15:00 | 623.20 | 620.95 | 630.00 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 09:15:00 | 635.05 | 620.95 | 630.00 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 630.25 | 622.81 | 630.03 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 624.80 | 624.65 | 630.21 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 11:15:00 | 638.95 | 627.51 | 631.00 | SL hit (close>static) qty=1.00 sl=636.90 alert=retest2 |

### Cycle 66 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 641.60 | 634.11 | 633.38 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 650.80 | 637.45 | 634.96 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 09:15:00 | 651.00 | 656.84 | 650.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-30 09:15:00 | 651.00 | 656.84 | 650.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 651.00 | 656.84 | 650.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:00:00 | 651.00 | 656.84 | 650.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 10:15:00 | 650.80 | 655.63 | 650.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 10:30:00 | 645.65 | 655.63 | 650.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 11:15:00 | 652.55 | 655.01 | 650.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-30 12:15:00 | 653.00 | 655.01 | 650.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-02 15:15:00 | 652.00 | 663.44 | 664.71 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 67 — SELL (started 2026-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 15:15:00 | 652.00 | 663.44 | 664.71 | EMA200 below EMA400 |

### Cycle 68 — BUY (started 2026-04-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 13:15:00 | 670.50 | 665.95 | 665.37 | EMA200 above EMA400 |

### Cycle 69 — SELL (started 2026-04-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-07 11:15:00 | 657.10 | 664.70 | 665.21 | EMA200 below EMA400 |

### Cycle 70 — BUY (started 2026-04-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-08 11:15:00 | 670.00 | 663.71 | 663.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-09 09:15:00 | 673.55 | 667.23 | 665.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 14:15:00 | 668.00 | 669.77 | 667.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-09 14:15:00 | 668.00 | 669.77 | 667.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 14:15:00 | 668.00 | 669.77 | 667.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-09 15:00:00 | 668.00 | 669.77 | 667.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-09 15:15:00 | 671.00 | 670.01 | 668.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 09:30:00 | 671.65 | 670.22 | 668.33 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 10:30:00 | 671.80 | 670.55 | 668.65 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 11:00:00 | 671.85 | 670.55 | 668.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-10 12:30:00 | 672.10 | 672.38 | 669.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 672.85 | 676.17 | 672.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 679.45 | 676.45 | 673.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 14:15:00 | 667.55 | 671.49 | 671.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2026-04-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 14:15:00 | 667.55 | 671.49 | 671.62 | EMA200 below EMA400 |

### Cycle 72 — BUY (started 2026-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 09:15:00 | 678.65 | 672.20 | 671.88 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 10:15:00 | 681.60 | 674.08 | 672.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-16 09:15:00 | 677.30 | 679.51 | 676.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-16 09:15:00 | 677.30 | 679.51 | 676.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 09:15:00 | 677.30 | 679.51 | 676.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 09:30:00 | 676.00 | 679.51 | 676.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 678.40 | 679.29 | 676.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 10:45:00 | 678.15 | 679.29 | 676.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 13:15:00 | 679.85 | 679.10 | 677.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 14:15:00 | 672.90 | 679.10 | 677.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 14:15:00 | 676.00 | 678.48 | 677.20 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-16 14:30:00 | 673.55 | 678.48 | 677.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 15:15:00 | 675.70 | 677.92 | 677.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 677.85 | 677.92 | 677.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-20 13:45:00 | 678.30 | 683.60 | 682.45 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-20 15:15:00 | 677.00 | 681.65 | 681.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2026-04-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-20 15:15:00 | 677.00 | 681.65 | 681.72 | EMA200 below EMA400 |

### Cycle 74 — BUY (started 2026-04-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-21 09:15:00 | 687.35 | 682.79 | 682.23 | EMA200 above EMA400 |

### Cycle 75 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 677.60 | 681.88 | 682.35 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2026-04-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 15:15:00 | 684.00 | 681.96 | 681.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-23 09:15:00 | 685.65 | 682.69 | 682.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 11:15:00 | 714.90 | 714.95 | 707.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-28 11:30:00 | 715.00 | 714.95 | 707.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 706.20 | 713.20 | 707.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 13:00:00 | 706.20 | 713.20 | 707.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 13:15:00 | 705.35 | 711.63 | 707.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 14:00:00 | 705.35 | 711.63 | 707.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 14:15:00 | 705.65 | 710.43 | 707.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 15:00:00 | 705.65 | 710.43 | 707.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-29 09:15:00 | 698.55 | 707.35 | 706.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-29 09:45:00 | 698.60 | 707.35 | 706.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — SELL (started 2026-04-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 10:15:00 | 691.15 | 704.11 | 704.90 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 685.45 | 696.16 | 699.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 14:15:00 | 700.05 | 695.08 | 698.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 14:15:00 | 700.05 | 695.08 | 698.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 700.05 | 695.08 | 698.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:45:00 | 703.00 | 695.08 | 698.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 707.00 | 697.47 | 698.87 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 750.20 | 697.47 | 698.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — BUY (started 2026-05-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 09:15:00 | 754.80 | 708.93 | 703.96 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-23 11:00:00 | 562.70 | 2025-05-30 10:15:00 | 563.95 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-05-23 11:30:00 | 560.65 | 2025-05-30 10:15:00 | 563.95 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-06-11 14:00:00 | 570.30 | 2025-06-12 09:15:00 | 578.05 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2025-06-11 15:00:00 | 569.80 | 2025-06-12 09:15:00 | 578.05 | STOP_HIT | 1.00 | -1.45% |
| BUY | retest2 | 2025-06-25 10:30:00 | 579.90 | 2025-07-01 15:15:00 | 588.00 | STOP_HIT | 1.00 | 1.40% |
| BUY | retest2 | 2025-07-03 09:15:00 | 620.15 | 2025-07-09 10:15:00 | 615.55 | STOP_HIT | 1.00 | -0.74% |
| SELL | retest2 | 2025-07-15 13:30:00 | 597.80 | 2025-07-22 12:15:00 | 598.55 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2025-07-15 15:15:00 | 598.50 | 2025-07-22 12:15:00 | 598.55 | STOP_HIT | 1.00 | -0.01% |
| SELL | retest2 | 2025-07-17 09:15:00 | 598.85 | 2025-07-22 12:15:00 | 598.55 | STOP_HIT | 1.00 | 0.05% |
| SELL | retest2 | 2025-07-17 13:00:00 | 597.30 | 2025-07-22 12:15:00 | 598.55 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-07-18 10:15:00 | 592.10 | 2025-07-22 12:15:00 | 598.55 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-07-18 13:30:00 | 593.65 | 2025-07-22 12:15:00 | 598.55 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2025-07-21 09:15:00 | 587.25 | 2025-07-22 12:15:00 | 598.55 | STOP_HIT | 1.00 | -1.92% |
| BUY | retest2 | 2025-07-24 12:15:00 | 599.45 | 2025-07-24 14:15:00 | 595.40 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2025-07-24 12:45:00 | 599.40 | 2025-07-24 14:15:00 | 595.40 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-07-24 14:00:00 | 599.55 | 2025-07-24 14:15:00 | 595.40 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2025-08-01 14:30:00 | 598.15 | 2025-08-04 09:15:00 | 586.00 | STOP_HIT | 1.00 | -2.03% |
| SELL | retest2 | 2025-08-08 11:45:00 | 588.00 | 2025-08-12 12:15:00 | 598.45 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-08-11 15:15:00 | 587.00 | 2025-08-12 12:15:00 | 598.45 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2025-08-18 09:15:00 | 612.40 | 2025-08-21 09:15:00 | 614.10 | STOP_HIT | 1.00 | 0.28% |
| SELL | retest2 | 2025-08-25 12:45:00 | 604.95 | 2025-08-29 09:15:00 | 574.70 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-25 12:45:00 | 604.95 | 2025-08-29 09:15:00 | 597.55 | STOP_HIT | 0.50 | 1.22% |
| SELL | retest2 | 2025-08-26 10:30:00 | 601.15 | 2025-08-29 09:15:00 | 571.09 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 10:30:00 | 601.15 | 2025-08-29 09:15:00 | 597.55 | STOP_HIT | 0.50 | 0.60% |
| SELL | retest2 | 2025-08-26 12:30:00 | 605.15 | 2025-08-29 09:15:00 | 574.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-08-26 12:30:00 | 605.15 | 2025-08-29 09:15:00 | 597.55 | STOP_HIT | 0.50 | 1.26% |
| BUY | retest2 | 2025-09-08 14:15:00 | 645.00 | 2025-09-11 14:15:00 | 636.60 | STOP_HIT | 1.00 | -1.30% |
| BUY | retest2 | 2025-09-08 14:45:00 | 642.55 | 2025-09-11 14:15:00 | 636.60 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-09-10 09:15:00 | 642.55 | 2025-09-11 14:15:00 | 636.60 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-09-11 09:15:00 | 642.25 | 2025-09-11 14:15:00 | 636.60 | STOP_HIT | 1.00 | -0.88% |
| BUY | retest2 | 2025-09-23 15:15:00 | 639.60 | 2025-09-24 11:15:00 | 635.00 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-09-24 10:00:00 | 641.80 | 2025-09-24 11:15:00 | 635.00 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2025-09-29 11:15:00 | 626.70 | 2025-10-03 09:15:00 | 630.45 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-09-29 13:15:00 | 627.70 | 2025-10-03 09:15:00 | 630.45 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-09-29 14:30:00 | 626.65 | 2025-10-03 09:15:00 | 630.45 | STOP_HIT | 1.00 | -0.61% |
| SELL | retest2 | 2025-09-30 14:45:00 | 627.40 | 2025-10-03 09:15:00 | 630.45 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-10-08 09:15:00 | 667.25 | 2025-10-14 15:15:00 | 685.00 | STOP_HIT | 1.00 | 2.66% |
| SELL | retest2 | 2025-10-30 10:15:00 | 693.95 | 2025-11-06 10:15:00 | 691.80 | STOP_HIT | 1.00 | 0.31% |
| SELL | retest2 | 2025-10-30 13:45:00 | 694.75 | 2025-11-06 10:15:00 | 691.80 | STOP_HIT | 1.00 | 0.42% |
| SELL | retest2 | 2025-10-31 09:15:00 | 678.85 | 2025-11-06 10:15:00 | 691.80 | STOP_HIT | 1.00 | -1.91% |
| SELL | retest2 | 2025-11-03 10:45:00 | 691.20 | 2025-11-06 10:15:00 | 691.80 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-11-03 13:15:00 | 681.85 | 2025-11-06 10:15:00 | 691.80 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2025-11-04 09:30:00 | 680.10 | 2025-11-06 10:15:00 | 691.80 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2025-11-07 10:45:00 | 694.15 | 2025-11-10 09:15:00 | 680.00 | STOP_HIT | 1.00 | -2.04% |
| SELL | retest1 | 2025-11-11 09:15:00 | 672.35 | 2025-11-12 09:15:00 | 683.25 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-11-13 14:45:00 | 675.30 | 2025-11-17 13:15:00 | 680.50 | STOP_HIT | 1.00 | -0.77% |
| SELL | retest2 | 2025-11-14 09:45:00 | 676.80 | 2025-11-17 13:15:00 | 680.50 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-11-14 10:30:00 | 676.05 | 2025-11-17 13:15:00 | 680.50 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-11-14 12:45:00 | 676.40 | 2025-11-18 09:15:00 | 697.25 | STOP_HIT | 1.00 | -3.08% |
| SELL | retest2 | 2025-11-17 09:15:00 | 673.35 | 2025-11-18 09:15:00 | 697.25 | STOP_HIT | 1.00 | -3.55% |
| SELL | retest2 | 2025-11-17 11:15:00 | 674.65 | 2025-11-18 09:15:00 | 697.25 | STOP_HIT | 1.00 | -3.35% |
| SELL | retest2 | 2025-11-17 12:45:00 | 675.90 | 2025-11-18 09:15:00 | 697.25 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2025-12-01 13:30:00 | 666.85 | 2025-12-02 09:15:00 | 658.10 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-12-05 09:30:00 | 648.80 | 2025-12-09 09:15:00 | 616.36 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-12-05 09:30:00 | 648.80 | 2025-12-09 14:15:00 | 622.00 | STOP_HIT | 0.50 | 4.13% |
| BUY | retest2 | 2025-12-29 13:00:00 | 613.30 | 2025-12-30 09:15:00 | 605.85 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-12-29 13:45:00 | 614.00 | 2025-12-30 09:15:00 | 605.85 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2026-02-03 10:45:00 | 563.50 | 2026-02-03 14:15:00 | 546.30 | STOP_HIT | 1.00 | -3.05% |
| BUY | retest2 | 2026-02-16 10:00:00 | 607.85 | 2026-03-02 12:15:00 | 640.15 | STOP_HIT | 1.00 | 5.31% |
| BUY | retest2 | 2026-02-16 10:30:00 | 608.00 | 2026-03-02 12:15:00 | 640.15 | STOP_HIT | 1.00 | 5.29% |
| BUY | retest2 | 2026-03-10 09:15:00 | 665.05 | 2026-03-13 12:15:00 | 666.50 | STOP_HIT | 1.00 | 0.22% |
| BUY | retest2 | 2026-03-10 11:00:00 | 667.60 | 2026-03-13 12:15:00 | 666.50 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2026-03-24 10:30:00 | 624.80 | 2026-03-24 11:15:00 | 638.95 | STOP_HIT | 1.00 | -2.26% |
| BUY | retest2 | 2026-03-30 12:15:00 | 653.00 | 2026-04-02 15:15:00 | 652.00 | STOP_HIT | 1.00 | -0.15% |
| BUY | retest2 | 2026-04-10 09:30:00 | 671.65 | 2026-04-13 14:15:00 | 667.55 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2026-04-10 10:30:00 | 671.80 | 2026-04-13 14:15:00 | 667.55 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2026-04-10 11:00:00 | 671.85 | 2026-04-13 14:15:00 | 667.55 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2026-04-10 12:30:00 | 672.10 | 2026-04-13 14:15:00 | 667.55 | STOP_HIT | 1.00 | -0.68% |
| BUY | retest2 | 2026-04-13 10:45:00 | 679.45 | 2026-04-13 14:15:00 | 667.55 | STOP_HIT | 1.00 | -1.75% |
| BUY | retest2 | 2026-04-17 09:15:00 | 677.85 | 2026-04-20 15:15:00 | 677.00 | STOP_HIT | 1.00 | -0.13% |
| BUY | retest2 | 2026-04-20 13:45:00 | 678.30 | 2026-04-20 15:15:00 | 677.00 | STOP_HIT | 1.00 | -0.19% |
