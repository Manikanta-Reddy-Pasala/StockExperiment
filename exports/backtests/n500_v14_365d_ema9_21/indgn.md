# Indegene Ltd. (INDGN)

## Backtest Summary

- **Window:** 2025-04-11 09:15:00 → 2026-05-08 15:15:00 (1850 bars)
- **Last close:** 530.00
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 92 |
| ALERT1 | 48 |
| ALERT2 | 47 |
| ALERT2_SKIP | 26 |
| ALERT3 | 159 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 91 |
| PARTIAL | 4 |
| TARGET_HIT | 2 |
| STOP_HIT | 89 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 95 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 25 / 70
- **Target hits / Stop hits / Partials:** 2 / 89 / 4
- **Avg / median % per leg:** 0.27% / -0.63%
- **Sum % (uncompounded):** 25.18%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 43 | 16 | 37.2% | 2 | 41 | 0 | 0.86% | 37.0% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 43 | 16 | 37.2% | 2 | 41 | 0 | 0.86% | 37.0% |
| SELL (all) | 52 | 9 | 17.3% | 0 | 48 | 4 | -0.23% | -11.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 52 | 9 | 17.3% | 0 | 48 | 4 | -0.23% | -11.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 95 | 25 | 26.3% | 2 | 89 | 4 | 0.27% | 25.2% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 561.00 | 552.48 | 551.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 568.10 | 561.22 | 556.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 10:15:00 | 567.20 | 569.20 | 564.95 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-14 11:00:00 | 567.20 | 569.20 | 564.95 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 14:15:00 | 598.45 | 605.30 | 600.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 15:00:00 | 598.45 | 605.30 | 600.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 15:15:00 | 597.10 | 603.66 | 599.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-20 09:15:00 | 603.35 | 603.66 | 599.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-20 12:15:00 | 595.65 | 601.92 | 600.38 | SL hit (close<static) qty=1.00 sl=596.15 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 09:30:00 | 603.80 | 600.40 | 599.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 10:45:00 | 601.65 | 600.65 | 600.07 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-21 11:45:00 | 602.00 | 600.41 | 600.01 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 12:15:00 | 600.55 | 600.44 | 600.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-22 09:15:00 | 603.40 | 600.47 | 600.17 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-23 13:45:00 | 601.55 | 602.84 | 602.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-23 14:15:00 | 600.20 | 602.31 | 602.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-23 14:15:00 | 600.20 | 602.31 | 602.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-23 14:15:00 | 600.20 | 602.31 | 602.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-23 14:15:00 | 600.20 | 602.31 | 602.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-05-23 14:15:00 | 600.20 | 602.31 | 602.34 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-23 14:15:00 | 600.20 | 602.31 | 602.34 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 09:15:00 | 607.00 | 602.88 | 602.57 | EMA200 above EMA400 |

### Cycle 4 — SELL (started 2025-05-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 09:15:00 | 601.50 | 605.41 | 605.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 12:15:00 | 595.65 | 601.48 | 603.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-30 09:15:00 | 592.55 | 590.93 | 594.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-30 09:15:00 | 592.55 | 590.93 | 594.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 592.55 | 590.93 | 594.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:45:00 | 593.25 | 590.93 | 594.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 11:15:00 | 601.00 | 593.07 | 594.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 11:45:00 | 600.65 | 593.07 | 594.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 12:15:00 | 605.30 | 595.51 | 595.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 13:00:00 | 605.30 | 595.51 | 595.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 5 — BUY (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 13:15:00 | 601.75 | 596.76 | 596.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 15:15:00 | 606.40 | 600.17 | 598.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 592.15 | 611.17 | 607.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 592.15 | 611.17 | 607.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 592.15 | 611.17 | 607.66 | EMA400 retest candle locked (from upside) |

### Cycle 6 — SELL (started 2025-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-04 11:15:00 | 592.55 | 604.43 | 605.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-04 14:15:00 | 590.55 | 598.19 | 601.71 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-05 09:15:00 | 607.00 | 599.36 | 601.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-05 09:15:00 | 607.00 | 599.36 | 601.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 09:15:00 | 607.00 | 599.36 | 601.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 09:30:00 | 605.90 | 599.36 | 601.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-05 10:15:00 | 610.15 | 601.52 | 602.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-05 10:45:00 | 613.60 | 601.52 | 602.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2025-06-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-05 11:15:00 | 612.00 | 603.61 | 603.25 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 15:15:00 | 591.50 | 601.05 | 602.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-06 09:15:00 | 581.65 | 597.17 | 600.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-09 10:15:00 | 590.00 | 586.95 | 591.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-09 11:00:00 | 590.00 | 586.95 | 591.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 11:15:00 | 592.10 | 587.98 | 591.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 11:30:00 | 593.40 | 587.98 | 591.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 12:15:00 | 592.35 | 588.85 | 591.90 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 12:45:00 | 592.85 | 588.85 | 591.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 13:15:00 | 591.60 | 589.40 | 591.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 14:15:00 | 590.35 | 589.40 | 591.87 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 15:15:00 | 590.00 | 589.80 | 591.83 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-11 09:30:00 | 586.50 | 583.99 | 587.09 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-11 11:15:00 | 599.70 | 587.60 | 588.22 | SL hit (close>static) qty=1.00 sl=596.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-11 11:15:00 | 599.70 | 587.60 | 588.22 | SL hit (close>static) qty=1.00 sl=596.70 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-11 11:15:00 | 599.70 | 587.60 | 588.22 | SL hit (close>static) qty=1.00 sl=596.70 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-11 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 12:15:00 | 596.55 | 589.39 | 588.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-11 15:15:00 | 612.90 | 596.08 | 592.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 13:15:00 | 596.05 | 600.34 | 596.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-12 13:15:00 | 596.05 | 600.34 | 596.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 13:15:00 | 596.05 | 600.34 | 596.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 14:00:00 | 596.05 | 600.34 | 596.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 14:15:00 | 599.10 | 600.09 | 596.65 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 09:15:00 | 601.05 | 599.87 | 596.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-13 14:30:00 | 602.50 | 598.86 | 597.61 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 586.30 | 596.61 | 596.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-06-16 09:15:00 | 586.30 | 596.61 | 596.82 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-06-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-16 09:15:00 | 586.30 | 596.61 | 596.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 14:15:00 | 580.65 | 587.75 | 590.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 596.30 | 588.36 | 590.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 596.30 | 588.36 | 590.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 596.30 | 588.36 | 590.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:45:00 | 595.35 | 588.36 | 590.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 589.65 | 588.62 | 590.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 13:45:00 | 587.45 | 588.63 | 590.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 13:15:00 | 577.15 | 576.45 | 576.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-06-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 13:15:00 | 577.15 | 576.45 | 576.44 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-06-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-24 14:15:00 | 575.00 | 576.16 | 576.31 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-24 15:15:00 | 574.00 | 575.73 | 576.10 | Break + close below crossover candle low |

### Cycle 13 — BUY (started 2025-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-25 09:15:00 | 581.95 | 576.97 | 576.63 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-06-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-26 11:15:00 | 574.25 | 577.08 | 577.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-26 13:15:00 | 572.60 | 575.82 | 576.72 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-30 11:15:00 | 569.95 | 569.78 | 572.06 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-30 12:00:00 | 569.95 | 569.78 | 572.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 14:15:00 | 574.20 | 570.46 | 571.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-30 15:00:00 | 574.20 | 570.46 | 571.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-30 15:15:00 | 575.05 | 571.38 | 572.08 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-01 09:15:00 | 579.25 | 571.38 | 572.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 15 — BUY (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-01 10:15:00 | 578.65 | 573.37 | 572.89 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-02 10:15:00 | 569.05 | 572.36 | 572.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-04 11:15:00 | 563.70 | 567.77 | 569.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-04 14:15:00 | 569.30 | 566.71 | 568.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-04 14:15:00 | 569.30 | 566.71 | 568.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 14:15:00 | 569.30 | 566.71 | 568.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-04 15:00:00 | 569.30 | 566.71 | 568.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-04 15:15:00 | 567.20 | 566.81 | 568.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-07 09:15:00 | 577.40 | 566.81 | 568.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-07 09:15:00 | 570.05 | 567.46 | 568.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 11:15:00 | 567.50 | 567.67 | 568.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 12:00:00 | 567.55 | 567.64 | 568.43 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-07 15:15:00 | 565.85 | 568.11 | 568.46 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-08 10:00:00 | 566.85 | 567.50 | 568.10 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-08 10:15:00 | 566.90 | 567.38 | 567.99 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-08 10:30:00 | 567.00 | 567.38 | 567.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 564.50 | 564.00 | 565.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:45:00 | 568.35 | 564.00 | 565.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 10:15:00 | 566.85 | 564.57 | 565.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 10:30:00 | 566.40 | 564.57 | 565.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 11:15:00 | 567.90 | 565.23 | 565.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 11:30:00 | 567.80 | 565.23 | 565.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-07-09 14:15:00 | 575.00 | 567.70 | 566.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-09 14:15:00 | 575.00 | 567.70 | 566.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-09 14:15:00 | 575.00 | 567.70 | 566.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-09 14:15:00 | 575.00 | 567.70 | 566.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-07-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 14:15:00 | 575.00 | 567.70 | 566.95 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2025-07-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 13:15:00 | 564.95 | 566.94 | 567.04 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-10 14:15:00 | 570.00 | 567.56 | 567.31 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2025-07-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 10:15:00 | 563.00 | 566.90 | 567.11 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 567.75 | 565.53 | 565.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 573.10 | 567.04 | 566.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 12:15:00 | 575.20 | 575.30 | 571.68 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 12:15:00 | 575.20 | 575.30 | 571.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 575.20 | 575.30 | 571.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 13:00:00 | 575.20 | 575.30 | 571.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 13:15:00 | 575.05 | 577.71 | 575.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-17 14:00:00 | 575.05 | 577.71 | 575.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-17 14:15:00 | 578.10 | 577.79 | 575.67 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-07-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 12:15:00 | 572.70 | 574.80 | 574.86 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2025-07-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-18 13:15:00 | 576.05 | 575.05 | 574.97 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-07-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 09:15:00 | 566.45 | 573.34 | 574.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-21 11:15:00 | 561.15 | 569.70 | 572.33 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 11:15:00 | 557.05 | 554.23 | 558.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 11:45:00 | 556.45 | 554.23 | 558.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 12:15:00 | 559.70 | 555.33 | 558.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-23 13:00:00 | 559.70 | 555.33 | 558.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 13:15:00 | 558.45 | 555.95 | 558.90 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:45:00 | 556.55 | 557.28 | 558.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 10:15:00 | 562.95 | 558.42 | 559.23 | SL hit (close>static) qty=1.00 sl=561.65 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 15:00:00 | 556.10 | 558.48 | 559.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:45:00 | 556.15 | 557.01 | 558.24 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 09:15:00 | 556.50 | 555.33 | 556.45 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 09:15:00 | 556.85 | 555.63 | 556.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 09:30:00 | 558.65 | 555.63 | 556.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 10:15:00 | 556.45 | 555.80 | 556.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:00:00 | 556.45 | 555.80 | 556.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 11:15:00 | 556.35 | 555.91 | 556.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-28 11:30:00 | 555.15 | 555.91 | 556.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-28 12:15:00 | 551.75 | 555.08 | 556.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-28 14:15:00 | 548.75 | 554.43 | 555.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 572.25 | 549.32 | 548.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 572.25 | 549.32 | 548.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 572.25 | 549.32 | 548.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-01 09:15:00 | 572.25 | 549.32 | 548.04 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 25 — BUY (started 2025-08-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-01 09:15:00 | 572.25 | 549.32 | 548.04 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-01 11:15:00 | 582.30 | 559.50 | 553.11 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-05 10:15:00 | 575.00 | 579.86 | 572.64 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-05 11:00:00 | 575.00 | 579.86 | 572.64 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 11:15:00 | 566.05 | 577.10 | 572.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-05 12:00:00 | 566.05 | 577.10 | 572.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-05 12:15:00 | 568.20 | 575.32 | 571.69 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-05 14:45:00 | 569.30 | 572.80 | 571.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-06 10:15:00 | 565.20 | 569.45 | 569.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 26 — SELL (started 2025-08-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-06 10:15:00 | 565.20 | 569.45 | 569.84 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-08-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-06 12:15:00 | 572.70 | 570.16 | 570.10 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-08-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 09:15:00 | 567.75 | 569.69 | 569.95 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-08-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 10:15:00 | 575.40 | 570.83 | 570.45 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-08-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-11 09:15:00 | 566.50 | 570.80 | 571.18 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 572.55 | 571.18 | 571.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 14:15:00 | 575.00 | 572.79 | 572.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-13 09:15:00 | 571.75 | 572.77 | 572.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-13 09:15:00 | 571.75 | 572.77 | 572.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 09:15:00 | 571.75 | 572.77 | 572.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-14 09:15:00 | 577.90 | 572.60 | 572.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:15:00 | 579.40 | 572.44 | 572.38 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 10:15:00 | 577.35 | 585.08 | 583.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-20 10:45:00 | 578.10 | 583.74 | 582.67 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-20 12:15:00 | 580.50 | 582.56 | 582.29 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-20 13:00:00 | 580.50 | 582.56 | 582.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-20 13:15:00 | 579.25 | 581.89 | 582.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-20 13:15:00 | 579.25 | 581.89 | 582.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-20 13:15:00 | 579.25 | 581.89 | 582.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-08-20 13:15:00 | 579.25 | 581.89 | 582.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-20 13:15:00 | 579.25 | 581.89 | 582.01 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-21 09:15:00 | 587.60 | 582.49 | 582.20 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-08-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-21 10:15:00 | 579.75 | 581.94 | 581.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-21 11:15:00 | 578.35 | 581.22 | 581.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-22 09:15:00 | 578.05 | 577.78 | 579.58 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-22 10:00:00 | 578.05 | 577.78 | 579.58 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 15:15:00 | 557.00 | 555.20 | 560.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-29 13:45:00 | 543.75 | 550.01 | 555.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-01 15:00:00 | 546.45 | 544.03 | 548.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 13:45:00 | 545.55 | 547.18 | 548.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-02 15:00:00 | 546.05 | 546.96 | 548.35 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 09:15:00 | 555.00 | 548.25 | 548.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-03 09:30:00 | 554.25 | 548.25 | 548.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 10:15:00 | 551.30 | 548.86 | 548.92 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-09-03 11:15:00 | 551.80 | 549.45 | 549.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 11:15:00 | 551.80 | 549.45 | 549.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 11:15:00 | 551.80 | 549.45 | 549.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-03 11:15:00 | 551.80 | 549.45 | 549.18 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-09-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-03 11:15:00 | 551.80 | 549.45 | 549.18 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2025-09-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 12:15:00 | 545.00 | 549.60 | 549.81 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-05 11:15:00 | 551.60 | 549.58 | 549.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-05 13:15:00 | 558.20 | 551.67 | 550.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-08 11:15:00 | 555.65 | 556.00 | 553.49 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-08 12:00:00 | 555.65 | 556.00 | 553.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 15:15:00 | 553.00 | 555.97 | 554.37 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-09 09:15:00 | 556.35 | 555.97 | 554.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-09 14:15:00 | 550.10 | 554.93 | 554.67 | SL hit (close<static) qty=1.00 sl=550.50 alert=retest2 |

### Cycle 38 — SELL (started 2025-09-09 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-09 15:15:00 | 550.55 | 554.05 | 554.30 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-09-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 09:15:00 | 561.80 | 555.60 | 554.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 13:15:00 | 570.10 | 564.43 | 560.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-15 10:15:00 | 571.50 | 571.85 | 568.17 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-15 10:45:00 | 572.30 | 571.85 | 568.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 14:15:00 | 566.80 | 570.94 | 568.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 15:00:00 | 566.80 | 570.94 | 568.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 15:15:00 | 576.55 | 572.06 | 569.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-17 09:15:00 | 584.00 | 573.59 | 571.94 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-22 13:15:00 | 578.55 | 584.44 | 584.66 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2025-09-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 13:15:00 | 578.55 | 584.44 | 584.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-22 14:15:00 | 577.35 | 583.02 | 583.99 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-23 12:15:00 | 578.50 | 577.82 | 580.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-23 12:30:00 | 578.25 | 577.82 | 580.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 15:15:00 | 572.15 | 571.37 | 573.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-26 09:15:00 | 563.90 | 571.37 | 573.35 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-26 14:15:00 | 576.55 | 570.49 | 571.65 | SL hit (close>static) qty=1.00 sl=573.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 09:15:00 | 569.45 | 571.43 | 571.97 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-29 10:45:00 | 570.95 | 571.43 | 571.89 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 574.90 | 568.81 | 570.18 | SL hit (close>static) qty=1.00 sl=573.80 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-09-29 14:15:00 | 574.90 | 568.81 | 570.18 | SL hit (close>static) qty=1.00 sl=573.80 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 09:15:00 | 566.35 | 570.03 | 570.61 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 563.40 | 563.86 | 566.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:30:00 | 564.85 | 563.86 | 566.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 567.20 | 564.53 | 566.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 567.20 | 564.53 | 566.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 564.10 | 564.44 | 566.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-01 14:30:00 | 563.15 | 564.53 | 565.88 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 567.60 | 565.14 | 566.03 | SL hit (close>static) qty=1.00 sl=567.25 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-03 11:15:00 | 569.85 | 566.62 | 566.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2025-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-03 11:15:00 | 569.85 | 566.62 | 566.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 13:15:00 | 581.10 | 569.87 | 568.04 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 11:15:00 | 570.85 | 573.15 | 570.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-06 11:15:00 | 570.85 | 573.15 | 570.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 570.85 | 573.15 | 570.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 12:00:00 | 570.85 | 573.15 | 570.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 12:15:00 | 570.90 | 572.70 | 570.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-06 13:00:00 | 570.90 | 572.70 | 570.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 13:15:00 | 571.95 | 572.55 | 570.92 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 14:15:00 | 573.00 | 572.55 | 570.92 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 09:15:00 | 558.20 | 569.23 | 569.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 42 — SELL (started 2025-10-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 09:15:00 | 558.20 | 569.23 | 569.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 10:15:00 | 556.05 | 566.60 | 568.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-08 14:15:00 | 550.55 | 550.40 | 556.19 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-08 15:00:00 | 550.55 | 550.40 | 556.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 09:15:00 | 548.95 | 549.73 | 554.86 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-10-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 14:15:00 | 556.75 | 554.48 | 554.48 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 551.55 | 554.16 | 554.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 11:15:00 | 550.05 | 552.81 | 553.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 13:15:00 | 539.05 | 536.57 | 540.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 13:15:00 | 539.05 | 536.57 | 540.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 13:15:00 | 539.05 | 536.57 | 540.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 14:00:00 | 539.05 | 536.57 | 540.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 14:15:00 | 541.00 | 537.45 | 540.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 15:00:00 | 541.00 | 537.45 | 540.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 15:15:00 | 544.95 | 538.95 | 541.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-16 09:15:00 | 544.50 | 538.95 | 541.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-16 09:15:00 | 542.85 | 539.73 | 541.34 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:30:00 | 540.75 | 539.89 | 541.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 11:00:00 | 540.75 | 540.34 | 540.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 14:00:00 | 541.80 | 540.97 | 541.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-17 14:15:00 | 543.15 | 541.41 | 541.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 14:15:00 | 543.15 | 541.41 | 541.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-17 14:15:00 | 543.15 | 541.41 | 541.19 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — BUY (started 2025-10-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-17 14:15:00 | 543.15 | 541.41 | 541.19 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2025-10-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-20 09:15:00 | 534.50 | 540.12 | 540.65 | EMA200 below EMA400 |

### Cycle 47 — BUY (started 2025-10-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-20 12:15:00 | 549.95 | 541.60 | 541.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 14:15:00 | 557.85 | 546.84 | 543.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-23 09:15:00 | 549.75 | 550.30 | 546.66 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-23 09:45:00 | 550.00 | 550.30 | 546.66 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 11:15:00 | 547.90 | 549.45 | 546.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 11:45:00 | 547.80 | 549.45 | 546.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 546.65 | 548.89 | 546.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 12:30:00 | 546.30 | 548.89 | 546.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 546.30 | 548.37 | 546.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:45:00 | 545.85 | 548.37 | 546.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 544.45 | 547.59 | 546.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:15:00 | 547.40 | 547.59 | 546.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 15:15:00 | 547.40 | 547.55 | 546.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 09:15:00 | 543.60 | 547.55 | 546.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 09:15:00 | 544.65 | 546.97 | 546.48 | EMA400 retest candle locked (from upside) |

### Cycle 48 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 541.60 | 545.90 | 546.04 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-10-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-28 09:15:00 | 550.60 | 544.99 | 544.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-30 10:15:00 | 553.00 | 548.95 | 547.53 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 13:15:00 | 549.55 | 550.50 | 548.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-30 13:15:00 | 549.55 | 550.50 | 548.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 13:15:00 | 549.55 | 550.50 | 548.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-30 14:00:00 | 549.55 | 550.50 | 548.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 14:15:00 | 549.80 | 550.36 | 548.81 | EMA400 retest candle locked (from upside) |

### Cycle 50 — SELL (started 2025-10-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-31 10:15:00 | 534.80 | 545.94 | 547.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-06 09:15:00 | 533.75 | 537.48 | 538.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 13:15:00 | 524.50 | 524.48 | 529.41 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-07 14:00:00 | 524.50 | 524.48 | 529.41 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 515.75 | 522.25 | 527.13 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 09:30:00 | 512.65 | 517.87 | 522.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-11 14:30:00 | 509.70 | 515.14 | 519.16 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-12 10:45:00 | 511.45 | 514.29 | 517.76 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-13 10:15:00 | 513.00 | 511.88 | 514.77 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 512.40 | 511.98 | 514.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:00:00 | 512.40 | 511.98 | 514.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 11:15:00 | 515.05 | 512.60 | 514.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-13 11:45:00 | 515.05 | 512.60 | 514.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 12:15:00 | 514.05 | 512.89 | 514.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 09:15:00 | 513.00 | 513.42 | 514.41 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 10:00:00 | 512.25 | 513.19 | 514.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 10:30:00 | 511.75 | 512.93 | 514.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-14 13:00:00 | 512.55 | 512.61 | 513.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 13:15:00 | 512.50 | 512.59 | 513.55 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 13:45:00 | 512.55 | 512.59 | 513.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 511.65 | 511.66 | 512.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 14:30:00 | 507.90 | 511.46 | 512.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 508.00 | 511.06 | 512.09 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 13:45:00 | 508.00 | 508.22 | 509.95 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 14:15:00 | 507.75 | 508.22 | 509.95 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 14:15:00 | 506.80 | 507.94 | 509.67 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-11-19 14:15:00 | 514.40 | 510.52 | 510.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 14:15:00 | 514.40 | 510.52 | 510.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 14:15:00 | 514.40 | 510.52 | 510.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 14:15:00 | 514.40 | 510.52 | 510.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 14:15:00 | 514.40 | 510.52 | 510.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 14:15:00 | 514.40 | 510.52 | 510.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 14:15:00 | 514.40 | 510.52 | 510.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 14:15:00 | 514.40 | 510.52 | 510.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 14:15:00 | 514.40 | 510.52 | 510.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 14:15:00 | 514.40 | 510.52 | 510.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 14:15:00 | 514.40 | 510.52 | 510.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-19 14:15:00 | 514.40 | 510.52 | 510.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2025-11-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-19 14:15:00 | 514.40 | 510.52 | 510.23 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-11-21 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 12:15:00 | 510.00 | 511.95 | 512.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-24 09:15:00 | 506.45 | 510.09 | 511.07 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-24 12:15:00 | 510.00 | 509.12 | 510.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-24 12:15:00 | 510.00 | 509.12 | 510.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 12:15:00 | 510.00 | 509.12 | 510.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 12:45:00 | 510.35 | 509.12 | 510.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 13:15:00 | 510.60 | 509.42 | 510.32 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 14:15:00 | 510.80 | 509.42 | 510.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 14:15:00 | 514.80 | 510.49 | 510.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-24 15:00:00 | 514.80 | 510.49 | 510.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 53 — BUY (started 2025-11-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-24 15:15:00 | 514.05 | 511.21 | 511.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-25 10:15:00 | 515.50 | 512.29 | 511.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 13:15:00 | 530.10 | 530.19 | 526.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 14:15:00 | 527.45 | 529.64 | 526.81 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 14:15:00 | 527.45 | 529.64 | 526.81 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-01 15:00:00 | 527.45 | 529.64 | 526.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 15:15:00 | 528.25 | 529.37 | 526.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-02 09:30:00 | 525.60 | 528.97 | 526.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-02 10:15:00 | 529.20 | 529.02 | 527.18 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 12:00:00 | 532.65 | 529.74 | 527.68 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 13:15:00 | 533.35 | 529.80 | 527.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 14:00:00 | 531.65 | 530.17 | 528.23 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-02 14:45:00 | 530.40 | 530.16 | 528.41 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 520.45 | 528.27 | 527.86 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 520.45 | 528.27 | 527.86 | SL hit (close<static) qty=1.00 sl=526.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 520.45 | 528.27 | 527.86 | SL hit (close<static) qty=1.00 sl=526.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 520.45 | 528.27 | 527.86 | SL hit (close<static) qty=1.00 sl=526.35 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-03 09:15:00 | 520.45 | 528.27 | 527.86 | SL hit (close<static) qty=1.00 sl=526.35 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-12-03 09:45:00 | 520.30 | 528.27 | 527.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 521.30 | 526.88 | 527.26 | EMA200 below EMA400 |

### Cycle 55 — BUY (started 2025-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-03 13:15:00 | 530.80 | 527.80 | 527.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-04 12:15:00 | 532.75 | 529.82 | 528.68 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-05 09:15:00 | 530.65 | 531.59 | 530.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-05 09:15:00 | 530.65 | 531.59 | 530.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 09:15:00 | 530.65 | 531.59 | 530.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-05 09:45:00 | 530.15 | 531.59 | 530.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-05 15:15:00 | 535.00 | 533.26 | 531.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-08 09:15:00 | 533.70 | 533.26 | 531.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 09:15:00 | 536.85 | 533.97 | 532.15 | EMA400 retest candle locked (from upside) |

### Cycle 56 — SELL (started 2025-12-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 12:15:00 | 521.55 | 531.38 | 531.42 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2025-12-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 11:15:00 | 533.00 | 530.20 | 529.89 | EMA200 above EMA400 |

### Cycle 58 — SELL (started 2025-12-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-11 09:15:00 | 527.00 | 529.91 | 529.95 | EMA200 below EMA400 |

### Cycle 59 — BUY (started 2025-12-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-11 10:15:00 | 530.50 | 530.03 | 530.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 11:15:00 | 530.75 | 530.17 | 530.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 14:15:00 | 526.50 | 531.67 | 531.39 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 14:15:00 | 526.50 | 531.67 | 531.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 526.50 | 531.67 | 531.39 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:45:00 | 526.50 | 531.67 | 531.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 60 — SELL (started 2025-12-12 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-12 15:15:00 | 526.75 | 530.69 | 530.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-16 09:15:00 | 524.80 | 527.44 | 528.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-16 10:15:00 | 527.95 | 527.54 | 528.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-16 11:00:00 | 527.95 | 527.54 | 528.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 11:15:00 | 527.05 | 527.45 | 528.52 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-16 14:00:00 | 525.50 | 527.02 | 528.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 10:15:00 | 525.75 | 526.36 | 527.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 11:30:00 | 525.50 | 526.48 | 527.36 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-17 12:15:00 | 525.80 | 526.48 | 527.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 13:15:00 | 527.65 | 526.58 | 527.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-17 14:00:00 | 527.65 | 526.58 | 527.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 14:15:00 | 531.25 | 527.51 | 527.61 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-12-17 14:15:00 | 531.25 | 527.51 | 527.61 | SL hit (close>static) qty=1.00 sl=529.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 14:15:00 | 531.25 | 527.51 | 527.61 | SL hit (close>static) qty=1.00 sl=529.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 14:15:00 | 531.25 | 527.51 | 527.61 | SL hit (close>static) qty=1.00 sl=529.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-17 14:15:00 | 531.25 | 527.51 | 527.61 | SL hit (close>static) qty=1.00 sl=529.10 alert=retest2 |
| ALERT3_SIDEWAYS | 2025-12-17 15:00:00 | 531.25 | 527.51 | 527.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2025-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-17 15:15:00 | 532.35 | 528.48 | 528.04 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-12-18 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-18 13:15:00 | 525.15 | 527.40 | 527.67 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-12-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-18 14:15:00 | 535.90 | 529.10 | 528.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-18 15:15:00 | 538.10 | 530.90 | 529.30 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-19 09:15:00 | 530.00 | 530.72 | 529.36 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-19 09:15:00 | 530.00 | 530.72 | 529.36 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 530.00 | 530.72 | 529.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 09:30:00 | 530.15 | 530.72 | 529.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 10:15:00 | 529.20 | 530.42 | 529.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:00:00 | 529.20 | 530.42 | 529.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 11:15:00 | 533.10 | 530.95 | 529.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 11:30:00 | 532.40 | 530.95 | 529.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 12:15:00 | 528.60 | 530.48 | 529.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-19 13:00:00 | 528.60 | 530.48 | 529.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 13:15:00 | 531.85 | 530.76 | 529.80 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 14:30:00 | 532.95 | 530.68 | 529.85 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-19 15:15:00 | 533.00 | 530.68 | 529.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 528.00 | 529.90 | 529.77 | SL hit (close<static) qty=1.00 sl=528.10 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-22 11:15:00 | 528.00 | 529.90 | 529.77 | SL hit (close<static) qty=1.00 sl=528.10 alert=retest2 |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-22 15:00:00 | 535.75 | 530.94 | 530.27 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-23 09:30:00 | 532.60 | 534.02 | 531.91 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 11:15:00 | 532.65 | 533.52 | 532.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 11:30:00 | 532.75 | 533.52 | 532.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 12:15:00 | 531.80 | 533.17 | 532.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:00:00 | 531.80 | 533.17 | 532.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 13:15:00 | 532.00 | 532.94 | 532.01 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 13:30:00 | 531.25 | 532.94 | 532.01 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 531.40 | 532.63 | 531.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 532.05 | 532.63 | 531.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 532.15 | 532.54 | 531.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 530.35 | 532.54 | 531.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 528.45 | 531.72 | 531.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-24 10:00:00 | 528.45 | 531.72 | 531.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-12-24 10:15:00 | 529.25 | 531.22 | 531.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-12-24 10:15:00 | 529.25 | 531.22 | 531.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2025-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-24 10:15:00 | 529.25 | 531.22 | 531.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-24 11:15:00 | 525.20 | 530.02 | 530.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-29 11:15:00 | 521.00 | 520.98 | 523.40 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-29 11:45:00 | 521.05 | 520.98 | 523.40 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 522.20 | 521.29 | 522.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-29 15:00:00 | 522.20 | 521.29 | 522.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 522.10 | 520.90 | 521.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 14:00:00 | 522.10 | 520.90 | 521.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 520.05 | 520.73 | 521.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:15:00 | 523.05 | 520.73 | 521.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 15:15:00 | 523.05 | 521.19 | 521.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-31 09:15:00 | 521.30 | 521.19 | 521.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-31 09:15:00 | 520.15 | 520.98 | 521.74 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 526.00 | 521.46 | 521.21 | EMA200 above EMA400 |

### Cycle 66 — SELL (started 2026-01-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-05 14:15:00 | 521.05 | 521.88 | 521.93 | EMA200 below EMA400 |

### Cycle 67 — BUY (started 2026-01-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 09:15:00 | 523.70 | 521.95 | 521.93 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2026-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 11:15:00 | 521.10 | 521.83 | 521.89 | EMA200 below EMA400 |

### Cycle 69 — BUY (started 2026-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-06 15:15:00 | 522.50 | 521.85 | 521.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-07 09:15:00 | 526.00 | 522.68 | 522.22 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-07 10:15:00 | 521.70 | 522.48 | 522.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-07 10:15:00 | 521.70 | 522.48 | 522.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 10:15:00 | 521.70 | 522.48 | 522.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 11:00:00 | 521.70 | 522.48 | 522.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 11:15:00 | 521.50 | 522.29 | 522.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:00:00 | 521.50 | 522.29 | 522.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-07 12:15:00 | 521.15 | 522.06 | 522.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-07 12:45:00 | 521.00 | 522.06 | 522.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 70 — SELL (started 2026-01-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 13:15:00 | 521.00 | 521.85 | 521.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 14:15:00 | 520.05 | 521.49 | 521.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 15:15:00 | 513.00 | 510.17 | 512.85 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-12 15:15:00 | 513.00 | 510.17 | 512.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 513.00 | 510.17 | 512.85 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 512.00 | 510.17 | 512.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 514.55 | 511.04 | 513.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:45:00 | 513.75 | 511.04 | 513.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 10:15:00 | 514.45 | 511.72 | 513.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 11:00:00 | 514.45 | 511.72 | 513.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 12:15:00 | 512.45 | 511.89 | 512.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 12:30:00 | 512.15 | 511.89 | 512.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 13:15:00 | 510.95 | 511.70 | 512.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 13:45:00 | 513.85 | 511.70 | 512.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 510.60 | 511.48 | 512.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 14:30:00 | 511.50 | 511.48 | 512.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 510.85 | 511.43 | 512.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:45:00 | 511.65 | 511.43 | 512.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 508.85 | 511.00 | 512.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:30:00 | 508.25 | 510.34 | 511.52 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:00:00 | 507.25 | 509.72 | 511.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 10:15:00 | 482.84 | 494.65 | 501.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-19 10:15:00 | 481.89 | 494.65 | 501.26 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 14:15:00 | 480.45 | 479.32 | 486.14 | SL hit (close>ema200) qty=0.50 sl=479.32 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-20 14:15:00 | 480.45 | 479.32 | 486.14 | SL hit (close>ema200) qty=0.50 sl=479.32 alert=retest2 |

### Cycle 71 — BUY (started 2026-01-28 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 15:15:00 | 478.00 | 471.16 | 471.11 | EMA200 above EMA400 |

### Cycle 72 — SELL (started 2026-01-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 09:15:00 | 470.00 | 470.93 | 471.01 | EMA200 below EMA400 |

### Cycle 73 — BUY (started 2026-01-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-29 12:15:00 | 471.60 | 471.06 | 471.06 | EMA200 above EMA400 |

### Cycle 74 — SELL (started 2026-01-29 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-29 13:15:00 | 469.45 | 470.74 | 470.91 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2026-01-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-30 09:15:00 | 487.65 | 473.77 | 472.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-30 10:15:00 | 490.55 | 477.13 | 473.89 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 09:15:00 | 480.10 | 482.44 | 478.76 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 09:15:00 | 480.10 | 482.44 | 478.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 480.10 | 482.44 | 478.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 10:00:00 | 480.10 | 482.44 | 478.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 483.75 | 483.65 | 480.33 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 482.00 | 483.65 | 480.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 14:15:00 | 481.10 | 483.63 | 480.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 15:00:00 | 481.10 | 483.63 | 480.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 15:15:00 | 475.05 | 481.92 | 480.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 09:30:00 | 488.05 | 483.70 | 481.35 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 11:30:00 | 483.75 | 482.81 | 481.33 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 12:15:00 | 481.95 | 482.81 | 481.33 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 12:45:00 | 482.05 | 482.68 | 481.40 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 486.65 | 483.47 | 481.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-02 14:30:00 | 491.40 | 484.88 | 482.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-04 14:15:00 | 479.65 | 487.12 | 487.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-04 14:15:00 | 479.65 | 487.12 | 487.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-04 14:15:00 | 479.65 | 487.12 | 487.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-04 14:15:00 | 479.65 | 487.12 | 487.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-04 14:15:00 | 479.65 | 487.12 | 487.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 76 — SELL (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 14:15:00 | 479.65 | 487.12 | 487.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-05 15:15:00 | 478.65 | 481.40 | 483.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-06 10:15:00 | 481.95 | 481.22 | 483.39 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-06 11:00:00 | 481.95 | 481.22 | 483.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 11:15:00 | 485.00 | 481.97 | 483.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 11:45:00 | 484.60 | 481.97 | 483.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 12:15:00 | 488.95 | 483.37 | 484.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-06 13:00:00 | 488.95 | 483.37 | 484.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 77 — BUY (started 2026-02-06 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-06 13:15:00 | 490.55 | 484.80 | 484.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-09 10:15:00 | 493.40 | 488.04 | 486.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-10 10:15:00 | 491.40 | 491.53 | 489.29 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-10 11:00:00 | 491.40 | 491.53 | 489.29 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 11:15:00 | 493.90 | 492.01 | 489.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 11:45:00 | 492.00 | 492.01 | 489.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 12:15:00 | 491.40 | 491.89 | 489.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 12:30:00 | 489.20 | 491.89 | 489.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 13:15:00 | 487.65 | 491.04 | 489.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:00:00 | 487.65 | 491.04 | 489.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-10 14:15:00 | 486.30 | 490.09 | 489.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-10 14:45:00 | 485.30 | 490.09 | 489.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 78 — SELL (started 2026-02-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-11 09:15:00 | 479.70 | 487.70 | 488.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-11 10:15:00 | 477.00 | 485.56 | 487.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 10:15:00 | 473.85 | 472.30 | 475.98 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 11:00:00 | 473.85 | 472.30 | 475.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 11:15:00 | 479.05 | 473.65 | 476.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:00:00 | 479.05 | 473.65 | 476.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-13 12:15:00 | 481.45 | 475.21 | 476.73 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-13 12:30:00 | 482.35 | 475.21 | 476.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 09:15:00 | 483.10 | 477.35 | 477.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 10:00:00 | 483.10 | 477.35 | 477.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 79 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 483.65 | 478.61 | 477.93 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-17 09:15:00 | 493.00 | 481.85 | 479.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-17 15:15:00 | 493.15 | 493.78 | 488.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-18 09:15:00 | 497.90 | 493.78 | 488.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 494.75 | 497.01 | 493.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 494.75 | 497.01 | 493.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 492.50 | 496.10 | 493.86 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:30:00 | 491.05 | 496.10 | 493.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 13:15:00 | 491.20 | 495.12 | 493.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 13:30:00 | 491.25 | 495.12 | 493.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-20 10:15:00 | 492.65 | 492.92 | 492.87 | EMA400 retest candle locked (from upside) |

### Cycle 80 — SELL (started 2026-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-20 12:15:00 | 491.20 | 492.56 | 492.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-20 13:15:00 | 487.00 | 491.45 | 492.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-24 14:15:00 | 479.65 | 477.92 | 481.96 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-24 15:00:00 | 479.65 | 477.92 | 481.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-24 15:15:00 | 481.40 | 478.61 | 481.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 09:15:00 | 486.15 | 478.61 | 481.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-25 09:15:00 | 496.80 | 482.25 | 483.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-25 10:00:00 | 496.80 | 482.25 | 483.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — BUY (started 2026-02-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-25 10:15:00 | 499.95 | 485.79 | 484.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-27 09:15:00 | 503.15 | 498.37 | 494.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-27 14:15:00 | 494.45 | 498.35 | 495.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-27 14:15:00 | 494.45 | 498.35 | 495.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 14:15:00 | 494.45 | 498.35 | 495.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-27 15:00:00 | 494.45 | 498.35 | 495.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-27 15:15:00 | 495.00 | 497.68 | 495.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-02 09:15:00 | 486.45 | 497.68 | 495.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — SELL (started 2026-03-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 10:15:00 | 484.00 | 492.91 | 493.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-02 11:15:00 | 481.70 | 490.67 | 492.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-04 15:15:00 | 474.00 | 473.48 | 479.60 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 09:15:00 | 467.50 | 473.48 | 479.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 09:15:00 | 459.85 | 454.90 | 458.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 09:45:00 | 459.80 | 454.90 | 458.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 10:15:00 | 458.85 | 455.69 | 458.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-10 10:45:00 | 461.70 | 455.69 | 458.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-10 11:15:00 | 456.25 | 455.80 | 458.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-10 12:15:00 | 455.00 | 455.80 | 458.57 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-11 10:00:00 | 454.15 | 454.34 | 456.73 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 12:15:00 | 432.25 | 438.93 | 444.79 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-13 13:15:00 | 431.44 | 437.58 | 443.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 09:15:00 | 437.45 | 435.12 | 440.79 | SL hit (close>ema200) qty=0.50 sl=435.12 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-16 09:15:00 | 437.45 | 435.12 | 440.79 | SL hit (close>ema200) qty=0.50 sl=435.12 alert=retest2 |

### Cycle 83 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 455.15 | 439.37 | 438.22 | EMA200 above EMA400 |

### Cycle 84 — SELL (started 2026-03-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-20 13:15:00 | 443.75 | 444.68 | 444.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-20 14:15:00 | 440.35 | 443.82 | 444.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 10:15:00 | 430.00 | 428.25 | 433.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:00:00 | 430.00 | 428.25 | 433.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 11:15:00 | 434.60 | 429.52 | 433.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 12:00:00 | 434.60 | 429.52 | 433.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 438.65 | 431.35 | 434.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 438.65 | 431.35 | 434.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 439.00 | 432.88 | 434.70 | EMA400 retest candle locked (from downside) |

### Cycle 85 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 449.00 | 437.61 | 436.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 461.05 | 442.30 | 438.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 452.20 | 455.21 | 448.81 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 14:15:00 | 451.55 | 454.78 | 451.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 14:15:00 | 451.55 | 454.78 | 451.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 15:00:00 | 451.55 | 454.78 | 451.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 15:15:00 | 451.50 | 454.12 | 451.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 09:15:00 | 448.00 | 454.12 | 451.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 09:15:00 | 451.20 | 453.54 | 451.13 | EMA400 retest candle locked (from upside) |

### Cycle 86 — SELL (started 2026-03-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 12:15:00 | 442.75 | 449.61 | 449.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 13:15:00 | 438.00 | 447.29 | 448.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 460.55 | 446.44 | 447.66 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 460.55 | 446.44 | 447.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 460.55 | 446.44 | 447.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 461.25 | 446.44 | 447.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 456.45 | 448.44 | 448.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:30:00 | 457.15 | 448.44 | 448.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 87 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 459.30 | 450.61 | 449.45 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-01 12:15:00 | 461.75 | 452.84 | 450.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-02 10:15:00 | 453.85 | 458.14 | 454.66 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-02 10:15:00 | 453.85 | 458.14 | 454.66 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 10:15:00 | 453.85 | 458.14 | 454.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-02 10:45:00 | 454.60 | 458.14 | 454.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-02 11:15:00 | 461.55 | 458.82 | 455.29 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 12:15:00 | 461.90 | 458.82 | 455.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-02 13:15:00 | 463.70 | 459.34 | 455.85 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 10:45:00 | 462.40 | 461.60 | 458.86 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 12:00:00 | 462.50 | 461.78 | 459.19 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-06 12:15:00 | 468.50 | 463.12 | 460.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 13:30:00 | 469.15 | 463.82 | 460.63 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-06 14:30:00 | 469.75 | 465.17 | 461.54 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 10:00:00 | 469.45 | 466.86 | 462.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-07 11:00:00 | 471.40 | 467.77 | 463.75 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 464.95 | 467.96 | 465.21 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 14:45:00 | 464.85 | 467.96 | 465.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 465.70 | 467.51 | 465.25 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 472.30 | 467.51 | 465.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-17 12:15:00 | 486.75 | 489.17 | 489.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 12:15:00 | 486.75 | 489.17 | 489.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 12:15:00 | 486.75 | 489.17 | 489.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 12:15:00 | 486.75 | 489.17 | 489.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 12:15:00 | 486.75 | 489.17 | 489.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 12:15:00 | 486.75 | 489.17 | 489.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 12:15:00 | 486.75 | 489.17 | 489.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 12:15:00 | 486.75 | 489.17 | 489.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-17 12:15:00 | 486.75 | 489.17 | 489.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2026-04-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-17 12:15:00 | 486.75 | 489.17 | 489.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-20 13:15:00 | 483.55 | 486.76 | 487.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-21 12:15:00 | 488.75 | 485.55 | 486.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 12:15:00 | 488.75 | 485.55 | 486.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 488.75 | 485.55 | 486.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-21 13:00:00 | 488.75 | 485.55 | 486.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 487.00 | 485.84 | 486.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-21 14:45:00 | 486.45 | 486.08 | 486.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 09:15:00 | 482.70 | 486.26 | 486.62 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-22 09:45:00 | 486.65 | 486.34 | 486.62 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 10:15:00 | 489.25 | 486.92 | 486.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-22 10:15:00 | 489.25 | 486.92 | 486.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-04-22 10:15:00 | 489.25 | 486.92 | 486.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 89 — BUY (started 2026-04-22 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-22 10:15:00 | 489.25 | 486.92 | 486.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-22 12:15:00 | 492.80 | 488.22 | 487.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-23 12:15:00 | 489.85 | 495.50 | 492.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-23 12:15:00 | 489.85 | 495.50 | 492.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 12:15:00 | 489.85 | 495.50 | 492.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-23 12:45:00 | 489.80 | 495.50 | 492.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-23 13:15:00 | 494.95 | 495.39 | 492.77 | EMA400 retest candle locked (from upside) |

### Cycle 90 — SELL (started 2026-04-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 11:15:00 | 484.40 | 490.68 | 491.40 | EMA200 below EMA400 |

### Cycle 91 — BUY (started 2026-04-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 09:15:00 | 493.95 | 491.86 | 491.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 09:15:00 | 499.85 | 495.21 | 493.63 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-28 11:15:00 | 493.15 | 495.51 | 494.08 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 11:15:00 | 493.15 | 495.51 | 494.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 11:15:00 | 493.15 | 495.51 | 494.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:00:00 | 493.15 | 495.51 | 494.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 12:15:00 | 501.60 | 496.73 | 494.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-28 12:30:00 | 497.80 | 496.73 | 494.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 15:15:00 | 498.85 | 498.37 | 496.11 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-29 09:15:00 | 504.55 | 498.37 | 496.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 09:30:00 | 502.70 | 497.92 | 497.15 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 11:15:00 | 504.15 | 498.22 | 497.36 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-30 12:45:00 | 501.80 | 499.49 | 498.11 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 500.00 | 499.99 | 498.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 498.70 | 499.99 | 498.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 501.00 | 500.19 | 498.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 530.60 | 500.19 | 498.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-05-05 10:15:00 | 552.97 | 532.98 | 519.94 | Target hit (10%) qty=1.00 alert=retest2 |
| Target hit | 2026-05-05 10:15:00 | 551.98 | 532.98 | 519.94 | Target hit (10%) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-08 13:15:00 | 530.00 | 531.19 | 531.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-08 13:15:00 | 530.00 | 531.19 | 531.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-05-08 13:15:00 | 530.00 | 531.19 | 531.23 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 92 — SELL (started 2026-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-08 13:15:00 | 530.00 | 531.19 | 531.23 | EMA200 below EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-20 09:15:00 | 603.35 | 2025-05-20 12:15:00 | 595.65 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-05-21 09:30:00 | 603.80 | 2025-05-23 14:15:00 | 600.20 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-05-21 10:45:00 | 601.65 | 2025-05-23 14:15:00 | 600.20 | STOP_HIT | 1.00 | -0.24% |
| BUY | retest2 | 2025-05-21 11:45:00 | 602.00 | 2025-05-23 14:15:00 | 600.20 | STOP_HIT | 1.00 | -0.30% |
| BUY | retest2 | 2025-05-22 09:15:00 | 603.40 | 2025-05-23 14:15:00 | 600.20 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-05-23 13:45:00 | 601.55 | 2025-05-23 14:15:00 | 600.20 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-06-09 14:15:00 | 590.35 | 2025-06-11 11:15:00 | 599.70 | STOP_HIT | 1.00 | -1.58% |
| SELL | retest2 | 2025-06-09 15:15:00 | 590.00 | 2025-06-11 11:15:00 | 599.70 | STOP_HIT | 1.00 | -1.64% |
| SELL | retest2 | 2025-06-11 09:30:00 | 586.50 | 2025-06-11 11:15:00 | 599.70 | STOP_HIT | 1.00 | -2.25% |
| BUY | retest2 | 2025-06-13 09:15:00 | 601.05 | 2025-06-16 09:15:00 | 586.30 | STOP_HIT | 1.00 | -2.45% |
| BUY | retest2 | 2025-06-13 14:30:00 | 602.50 | 2025-06-16 09:15:00 | 586.30 | STOP_HIT | 1.00 | -2.69% |
| SELL | retest2 | 2025-06-18 13:45:00 | 587.45 | 2025-06-24 13:15:00 | 577.15 | STOP_HIT | 1.00 | 1.75% |
| SELL | retest2 | 2025-07-07 11:15:00 | 567.50 | 2025-07-09 14:15:00 | 575.00 | STOP_HIT | 1.00 | -1.32% |
| SELL | retest2 | 2025-07-07 12:00:00 | 567.55 | 2025-07-09 14:15:00 | 575.00 | STOP_HIT | 1.00 | -1.31% |
| SELL | retest2 | 2025-07-07 15:15:00 | 565.85 | 2025-07-09 14:15:00 | 575.00 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2025-07-08 10:00:00 | 566.85 | 2025-07-09 14:15:00 | 575.00 | STOP_HIT | 1.00 | -1.44% |
| SELL | retest2 | 2025-07-24 09:45:00 | 556.55 | 2025-07-24 10:15:00 | 562.95 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-07-24 15:00:00 | 556.10 | 2025-08-01 09:15:00 | 572.25 | STOP_HIT | 1.00 | -2.90% |
| SELL | retest2 | 2025-07-25 09:45:00 | 556.15 | 2025-08-01 09:15:00 | 572.25 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2025-07-28 09:15:00 | 556.50 | 2025-08-01 09:15:00 | 572.25 | STOP_HIT | 1.00 | -2.83% |
| SELL | retest2 | 2025-07-28 14:15:00 | 548.75 | 2025-08-01 09:15:00 | 572.25 | STOP_HIT | 1.00 | -4.28% |
| BUY | retest2 | 2025-08-05 14:45:00 | 569.30 | 2025-08-06 10:15:00 | 565.20 | STOP_HIT | 1.00 | -0.72% |
| BUY | retest2 | 2025-08-14 09:15:00 | 577.90 | 2025-08-20 13:15:00 | 579.25 | STOP_HIT | 1.00 | 0.23% |
| BUY | retest2 | 2025-08-18 09:15:00 | 579.40 | 2025-08-20 13:15:00 | 579.25 | STOP_HIT | 1.00 | -0.03% |
| BUY | retest2 | 2025-08-20 10:15:00 | 577.35 | 2025-08-20 13:15:00 | 579.25 | STOP_HIT | 1.00 | 0.33% |
| BUY | retest2 | 2025-08-20 10:45:00 | 578.10 | 2025-08-20 13:15:00 | 579.25 | STOP_HIT | 1.00 | 0.20% |
| SELL | retest2 | 2025-08-29 13:45:00 | 543.75 | 2025-09-03 11:15:00 | 551.80 | STOP_HIT | 1.00 | -1.48% |
| SELL | retest2 | 2025-09-01 15:00:00 | 546.45 | 2025-09-03 11:15:00 | 551.80 | STOP_HIT | 1.00 | -0.98% |
| SELL | retest2 | 2025-09-02 13:45:00 | 545.55 | 2025-09-03 11:15:00 | 551.80 | STOP_HIT | 1.00 | -1.15% |
| SELL | retest2 | 2025-09-02 15:00:00 | 546.05 | 2025-09-03 11:15:00 | 551.80 | STOP_HIT | 1.00 | -1.05% |
| BUY | retest2 | 2025-09-09 09:15:00 | 556.35 | 2025-09-09 14:15:00 | 550.10 | STOP_HIT | 1.00 | -1.12% |
| BUY | retest2 | 2025-09-17 09:15:00 | 584.00 | 2025-09-22 13:15:00 | 578.55 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-09-26 09:15:00 | 563.90 | 2025-09-26 14:15:00 | 576.55 | STOP_HIT | 1.00 | -2.24% |
| SELL | retest2 | 2025-09-29 09:15:00 | 569.45 | 2025-09-29 14:15:00 | 574.90 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2025-09-29 10:45:00 | 570.95 | 2025-09-29 14:15:00 | 574.90 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-09-30 09:15:00 | 566.35 | 2025-10-01 15:15:00 | 567.60 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-10-01 14:30:00 | 563.15 | 2025-10-03 11:15:00 | 569.85 | STOP_HIT | 1.00 | -1.19% |
| BUY | retest2 | 2025-10-06 14:15:00 | 573.00 | 2025-10-07 09:15:00 | 558.20 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2025-10-16 10:30:00 | 540.75 | 2025-10-17 14:15:00 | 543.15 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-10-17 11:00:00 | 540.75 | 2025-10-17 14:15:00 | 543.15 | STOP_HIT | 1.00 | -0.44% |
| SELL | retest2 | 2025-10-17 14:00:00 | 541.80 | 2025-10-17 14:15:00 | 543.15 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-11-11 09:30:00 | 512.65 | 2025-11-19 14:15:00 | 514.40 | STOP_HIT | 1.00 | -0.34% |
| SELL | retest2 | 2025-11-11 14:30:00 | 509.70 | 2025-11-19 14:15:00 | 514.40 | STOP_HIT | 1.00 | -0.92% |
| SELL | retest2 | 2025-11-12 10:45:00 | 511.45 | 2025-11-19 14:15:00 | 514.40 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2025-11-13 10:15:00 | 513.00 | 2025-11-19 14:15:00 | 514.40 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-11-14 09:15:00 | 513.00 | 2025-11-19 14:15:00 | 514.40 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-11-14 10:00:00 | 512.25 | 2025-11-19 14:15:00 | 514.40 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-11-14 10:30:00 | 511.75 | 2025-11-19 14:15:00 | 514.40 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-11-14 13:00:00 | 512.55 | 2025-11-19 14:15:00 | 514.40 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-11-17 14:30:00 | 507.90 | 2025-11-19 14:15:00 | 514.40 | STOP_HIT | 1.00 | -1.28% |
| SELL | retest2 | 2025-11-18 09:15:00 | 508.00 | 2025-11-19 14:15:00 | 514.40 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-11-18 13:45:00 | 508.00 | 2025-11-19 14:15:00 | 514.40 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2025-11-18 14:15:00 | 507.75 | 2025-11-19 14:15:00 | 514.40 | STOP_HIT | 1.00 | -1.31% |
| BUY | retest2 | 2025-12-02 12:00:00 | 532.65 | 2025-12-03 09:15:00 | 520.45 | STOP_HIT | 1.00 | -2.29% |
| BUY | retest2 | 2025-12-02 13:15:00 | 533.35 | 2025-12-03 09:15:00 | 520.45 | STOP_HIT | 1.00 | -2.42% |
| BUY | retest2 | 2025-12-02 14:00:00 | 531.65 | 2025-12-03 09:15:00 | 520.45 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-12-02 14:45:00 | 530.40 | 2025-12-03 09:15:00 | 520.45 | STOP_HIT | 1.00 | -1.88% |
| SELL | retest2 | 2025-12-16 14:00:00 | 525.50 | 2025-12-17 14:15:00 | 531.25 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-12-17 10:15:00 | 525.75 | 2025-12-17 14:15:00 | 531.25 | STOP_HIT | 1.00 | -1.05% |
| SELL | retest2 | 2025-12-17 11:30:00 | 525.50 | 2025-12-17 14:15:00 | 531.25 | STOP_HIT | 1.00 | -1.09% |
| SELL | retest2 | 2025-12-17 12:15:00 | 525.80 | 2025-12-17 14:15:00 | 531.25 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2025-12-19 14:30:00 | 532.95 | 2025-12-22 11:15:00 | 528.00 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-12-19 15:15:00 | 533.00 | 2025-12-22 11:15:00 | 528.00 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-12-22 15:00:00 | 535.75 | 2025-12-24 10:15:00 | 529.25 | STOP_HIT | 1.00 | -1.21% |
| BUY | retest2 | 2025-12-23 09:30:00 | 532.60 | 2025-12-24 10:15:00 | 529.25 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2026-01-14 13:30:00 | 508.25 | 2026-01-19 10:15:00 | 482.84 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 15:00:00 | 507.25 | 2026-01-19 10:15:00 | 481.89 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 13:30:00 | 508.25 | 2026-01-20 14:15:00 | 480.45 | STOP_HIT | 0.50 | 5.47% |
| SELL | retest2 | 2026-01-14 15:00:00 | 507.25 | 2026-01-20 14:15:00 | 480.45 | STOP_HIT | 0.50 | 5.28% |
| BUY | retest2 | 2026-02-02 09:30:00 | 488.05 | 2026-02-04 14:15:00 | 479.65 | STOP_HIT | 1.00 | -1.72% |
| BUY | retest2 | 2026-02-02 11:30:00 | 483.75 | 2026-02-04 14:15:00 | 479.65 | STOP_HIT | 1.00 | -0.85% |
| BUY | retest2 | 2026-02-02 12:15:00 | 481.95 | 2026-02-04 14:15:00 | 479.65 | STOP_HIT | 1.00 | -0.48% |
| BUY | retest2 | 2026-02-02 12:45:00 | 482.05 | 2026-02-04 14:15:00 | 479.65 | STOP_HIT | 1.00 | -0.50% |
| BUY | retest2 | 2026-02-02 14:30:00 | 491.40 | 2026-02-04 14:15:00 | 479.65 | STOP_HIT | 1.00 | -2.39% |
| SELL | retest2 | 2026-03-10 12:15:00 | 455.00 | 2026-03-13 12:15:00 | 432.25 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-11 10:00:00 | 454.15 | 2026-03-13 13:15:00 | 431.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-03-10 12:15:00 | 455.00 | 2026-03-16 09:15:00 | 437.45 | STOP_HIT | 0.50 | 3.86% |
| SELL | retest2 | 2026-03-11 10:00:00 | 454.15 | 2026-03-16 09:15:00 | 437.45 | STOP_HIT | 0.50 | 3.68% |
| BUY | retest2 | 2026-04-02 12:15:00 | 461.90 | 2026-04-17 12:15:00 | 486.75 | STOP_HIT | 1.00 | 5.38% |
| BUY | retest2 | 2026-04-02 13:15:00 | 463.70 | 2026-04-17 12:15:00 | 486.75 | STOP_HIT | 1.00 | 4.97% |
| BUY | retest2 | 2026-04-06 10:45:00 | 462.40 | 2026-04-17 12:15:00 | 486.75 | STOP_HIT | 1.00 | 5.27% |
| BUY | retest2 | 2026-04-06 12:00:00 | 462.50 | 2026-04-17 12:15:00 | 486.75 | STOP_HIT | 1.00 | 5.24% |
| BUY | retest2 | 2026-04-06 13:30:00 | 469.15 | 2026-04-17 12:15:00 | 486.75 | STOP_HIT | 1.00 | 3.75% |
| BUY | retest2 | 2026-04-06 14:30:00 | 469.75 | 2026-04-17 12:15:00 | 486.75 | STOP_HIT | 1.00 | 3.62% |
| BUY | retest2 | 2026-04-07 10:00:00 | 469.45 | 2026-04-17 12:15:00 | 486.75 | STOP_HIT | 1.00 | 3.69% |
| BUY | retest2 | 2026-04-07 11:00:00 | 471.40 | 2026-04-17 12:15:00 | 486.75 | STOP_HIT | 1.00 | 3.26% |
| BUY | retest2 | 2026-04-08 09:15:00 | 472.30 | 2026-04-17 12:15:00 | 486.75 | STOP_HIT | 1.00 | 3.06% |
| SELL | retest2 | 2026-04-21 14:45:00 | 486.45 | 2026-04-22 10:15:00 | 489.25 | STOP_HIT | 1.00 | -0.58% |
| SELL | retest2 | 2026-04-22 09:15:00 | 482.70 | 2026-04-22 10:15:00 | 489.25 | STOP_HIT | 1.00 | -1.36% |
| SELL | retest2 | 2026-04-22 09:45:00 | 486.65 | 2026-04-22 10:15:00 | 489.25 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2026-04-29 09:15:00 | 504.55 | 2026-05-05 10:15:00 | 552.97 | TARGET_HIT | 1.00 | 9.60% |
| BUY | retest2 | 2026-04-30 09:30:00 | 502.70 | 2026-05-05 10:15:00 | 551.98 | TARGET_HIT | 1.00 | 9.80% |
| BUY | retest2 | 2026-04-30 11:15:00 | 504.15 | 2026-05-08 13:15:00 | 530.00 | STOP_HIT | 1.00 | 5.13% |
| BUY | retest2 | 2026-04-30 12:45:00 | 501.80 | 2026-05-08 13:15:00 | 530.00 | STOP_HIT | 1.00 | 5.62% |
| BUY | retest2 | 2026-05-04 09:15:00 | 530.60 | 2026-05-08 13:15:00 | 530.00 | STOP_HIT | 1.00 | -0.11% |
