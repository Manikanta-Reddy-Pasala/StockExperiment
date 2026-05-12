# LIC Housing Finance Ltd. (LICHSGFIN)

## Backtest Summary

- **Window:** 2025-03-13 09:15:00 → 2026-05-08 15:15:00 (1976 bars)
- **Last close:** 581.85
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 67 |
| ALERT1 | 49 |
| ALERT2 | 48 |
| ALERT2_SKIP | 25 |
| ALERT3 | 133 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 72 |
| PARTIAL | 2 |
| TARGET_HIT | 0 |
| STOP_HIT | 71 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 72 (incl. partial bookings)
- **Trades open at end:** 2
- **Winners / losers:** 12 / 60
- **Target hits / Stop hits / Partials:** 0 / 70 / 2
- **Avg / median % per leg:** -0.42% / -0.61%
- **Sum % (uncompounded):** -30.31%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 35 | 3 | 8.6% | 0 | 35 | 0 | -0.81% | -28.5% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 35 | 3 | 8.6% | 0 | 35 | 0 | -0.81% | -28.5% |
| SELL (all) | 37 | 9 | 24.3% | 0 | 35 | 2 | -0.05% | -1.8% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 37 | 9 | 24.3% | 0 | 35 | 2 | -0.05% | -1.8% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 72 | 12 | 16.7% | 0 | 70 | 2 | -0.42% | -30.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 12:15:00 | 595.00 | 587.30 | 586.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 13:15:00 | 602.60 | 590.36 | 587.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 14:15:00 | 603.05 | 604.89 | 598.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-13 15:00:00 | 603.05 | 604.89 | 598.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 613.05 | 605.74 | 599.98 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-14 15:00:00 | 618.60 | 610.02 | 604.46 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 12:45:00 | 617.00 | 617.57 | 610.89 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-15 13:15:00 | 617.40 | 617.57 | 610.89 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 10:30:00 | 618.60 | 619.77 | 614.74 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-16 12:15:00 | 617.90 | 618.87 | 615.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-16 12:30:00 | 615.35 | 618.87 | 615.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 09:15:00 | 606.40 | 617.03 | 615.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 09:30:00 | 601.70 | 617.03 | 615.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 10:15:00 | 606.40 | 614.90 | 614.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 10:45:00 | 606.10 | 614.90 | 614.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-05-19 11:15:00 | 604.85 | 612.89 | 613.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2025-05-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-19 11:15:00 | 604.85 | 612.89 | 613.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-19 13:15:00 | 602.20 | 609.43 | 612.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 14:15:00 | 596.30 | 593.11 | 597.62 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-21 14:15:00 | 596.30 | 593.11 | 597.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-21 14:15:00 | 596.30 | 593.11 | 597.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-21 15:00:00 | 596.30 | 593.11 | 597.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 14:15:00 | 593.75 | 590.48 | 593.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-22 15:00:00 | 593.75 | 590.48 | 593.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 15:15:00 | 593.80 | 591.15 | 593.84 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-23 09:15:00 | 594.35 | 591.15 | 593.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 593.35 | 591.59 | 593.80 | EMA400 retest candle locked (from downside) |

### Cycle 3 — BUY (started 2025-05-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-23 12:15:00 | 598.30 | 595.32 | 595.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 09:15:00 | 600.50 | 596.76 | 595.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 600.45 | 600.91 | 598.89 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 600.45 | 600.91 | 598.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 600.45 | 600.91 | 598.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 600.45 | 600.91 | 598.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 14:15:00 | 600.95 | 601.68 | 600.14 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 15:15:00 | 600.90 | 601.68 | 600.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 15:15:00 | 600.90 | 601.53 | 600.21 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-28 09:15:00 | 602.15 | 601.53 | 600.21 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-28 10:15:00 | 598.00 | 600.58 | 599.99 | SL hit (close<static) qty=1.00 sl=599.75 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-28 11:15:00 | 594.15 | 599.29 | 599.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-28 12:15:00 | 592.90 | 598.01 | 598.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-29 11:15:00 | 596.70 | 595.68 | 597.05 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-29 11:15:00 | 596.70 | 595.68 | 597.05 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 596.70 | 595.68 | 597.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:00:00 | 596.70 | 595.68 | 597.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 12:15:00 | 595.75 | 595.70 | 596.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 12:45:00 | 597.05 | 595.70 | 596.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 14:15:00 | 595.80 | 595.12 | 596.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-29 15:00:00 | 595.80 | 595.12 | 596.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 15:15:00 | 595.15 | 595.12 | 596.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-30 09:15:00 | 595.25 | 595.12 | 596.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-30 09:15:00 | 589.90 | 594.08 | 595.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 10:30:00 | 588.60 | 593.04 | 595.10 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-30 11:45:00 | 588.05 | 591.91 | 594.40 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 598.95 | 594.72 | 594.87 | SL hit (close>static) qty=1.00 sl=596.95 alert=retest2 |

### Cycle 5 — BUY (started 2025-06-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-02 10:15:00 | 598.30 | 595.44 | 595.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-02 13:15:00 | 600.25 | 596.79 | 595.90 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-04 09:15:00 | 602.15 | 604.66 | 601.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 09:15:00 | 602.15 | 604.66 | 601.91 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 602.15 | 604.66 | 601.91 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-04 09:30:00 | 601.75 | 604.66 | 601.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 603.65 | 604.45 | 602.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-05 10:00:00 | 606.85 | 604.46 | 603.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-05 15:15:00 | 599.55 | 602.47 | 602.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2025-06-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 15:15:00 | 599.55 | 602.47 | 602.62 | EMA200 below EMA400 |

### Cycle 7 — BUY (started 2025-06-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-06 10:15:00 | 608.45 | 603.46 | 603.03 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-06 12:15:00 | 614.80 | 606.62 | 604.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-10 10:15:00 | 619.00 | 619.41 | 615.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-10 10:45:00 | 617.95 | 619.41 | 615.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 623.50 | 619.39 | 616.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 10:30:00 | 626.10 | 620.24 | 617.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-11 11:15:00 | 625.25 | 620.24 | 617.40 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-12 11:15:00 | 612.35 | 618.83 | 618.54 | SL hit (close<static) qty=1.00 sl=615.50 alert=retest2 |

### Cycle 8 — SELL (started 2025-06-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 12:15:00 | 613.65 | 617.79 | 618.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 606.35 | 615.50 | 617.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 11:15:00 | 601.25 | 600.17 | 604.82 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 12:00:00 | 601.25 | 600.17 | 604.82 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 12:15:00 | 602.05 | 600.55 | 604.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 12:30:00 | 602.65 | 600.55 | 604.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 605.90 | 602.08 | 604.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 605.90 | 602.08 | 604.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 607.20 | 603.11 | 604.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 606.75 | 603.11 | 604.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 10:15:00 | 607.40 | 604.72 | 605.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-17 12:30:00 | 603.85 | 603.86 | 604.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 11:00:00 | 603.50 | 603.22 | 604.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-18 12:15:00 | 602.60 | 603.53 | 604.10 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-24 09:15:00 | 604.75 | 593.55 | 593.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-24 09:15:00 | 604.75 | 593.55 | 593.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 10:15:00 | 608.20 | 596.48 | 594.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 14:15:00 | 601.60 | 601.71 | 598.30 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-24 15:00:00 | 601.60 | 601.71 | 598.30 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 12:15:00 | 611.15 | 614.98 | 611.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-27 13:00:00 | 611.15 | 614.98 | 611.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-27 13:15:00 | 611.25 | 614.23 | 611.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-27 14:45:00 | 612.50 | 613.98 | 612.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-30 09:15:00 | 619.30 | 613.57 | 612.00 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-01 09:45:00 | 612.75 | 616.22 | 614.85 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 11:15:00 | 609.40 | 613.91 | 613.97 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 10 — SELL (started 2025-07-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 11:15:00 | 609.40 | 613.91 | 613.97 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-07 11:15:00 | 607.35 | 609.65 | 610.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-08 14:15:00 | 605.60 | 605.03 | 606.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-08 14:45:00 | 605.70 | 605.03 | 606.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 602.70 | 604.57 | 606.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-11 10:45:00 | 600.85 | 604.37 | 605.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-14 09:15:00 | 610.25 | 605.57 | 605.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 11 — BUY (started 2025-07-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 09:15:00 | 610.25 | 605.57 | 605.33 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 09:15:00 | 612.80 | 607.49 | 606.45 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-18 10:15:00 | 631.00 | 636.02 | 631.10 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-18 10:15:00 | 631.00 | 636.02 | 631.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 10:15:00 | 631.00 | 636.02 | 631.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 11:00:00 | 631.00 | 636.02 | 631.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 11:15:00 | 632.40 | 635.29 | 631.21 | EMA400 retest candle locked (from upside) |

### Cycle 12 — SELL (started 2025-07-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-21 10:15:00 | 624.00 | 628.78 | 629.40 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-22 13:15:00 | 621.75 | 624.69 | 626.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-23 14:15:00 | 619.30 | 617.84 | 621.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-23 15:00:00 | 619.30 | 617.84 | 621.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 621.30 | 618.53 | 621.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-24 09:15:00 | 619.60 | 618.53 | 621.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-24 09:15:00 | 617.75 | 618.38 | 620.85 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 10:30:00 | 617.25 | 618.05 | 620.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-25 09:30:00 | 614.00 | 614.72 | 617.51 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-07-29 10:15:00 | 586.39 | 594.24 | 601.64 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-07-29 12:15:00 | 596.10 | 594.23 | 600.33 | SL hit (close>ema200) qty=0.50 sl=594.23 alert=retest2 |

### Cycle 13 — BUY (started 2025-08-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-04 14:15:00 | 591.95 | 586.96 | 586.70 | EMA200 above EMA400 |

### Cycle 14 — SELL (started 2025-08-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-05 09:15:00 | 578.50 | 585.71 | 586.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-06 09:15:00 | 574.85 | 579.84 | 582.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-07 14:15:00 | 578.05 | 575.21 | 577.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-07 14:15:00 | 578.05 | 575.21 | 577.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 14:15:00 | 578.05 | 575.21 | 577.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-07 15:00:00 | 578.05 | 575.21 | 577.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-07 15:15:00 | 579.90 | 576.15 | 577.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-08 09:15:00 | 581.05 | 576.15 | 577.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-08 09:15:00 | 574.75 | 575.87 | 577.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-08 10:15:00 | 573.90 | 575.87 | 577.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 11:00:00 | 573.50 | 572.76 | 574.67 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 13:00:00 | 574.45 | 573.21 | 574.55 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-11 15:00:00 | 573.90 | 573.61 | 574.51 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 574.50 | 573.78 | 574.51 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:15:00 | 574.60 | 573.78 | 574.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 09:15:00 | 573.40 | 573.71 | 574.41 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 10:30:00 | 573.05 | 573.62 | 574.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 14:00:00 | 573.00 | 573.72 | 574.21 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 15:00:00 | 573.15 | 573.60 | 574.11 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 11:45:00 | 571.80 | 573.38 | 573.83 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 12:15:00 | 574.30 | 573.57 | 573.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:00:00 | 574.30 | 573.57 | 573.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 13:15:00 | 574.75 | 573.80 | 573.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 13:45:00 | 574.70 | 573.80 | 573.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-13 14:15:00 | 574.20 | 573.88 | 573.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-13 14:45:00 | 574.75 | 573.88 | 573.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-08-13 15:15:00 | 574.80 | 574.07 | 574.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 15:15:00 | 574.80 | 574.07 | 574.05 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-08-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-14 09:15:00 | 572.65 | 573.78 | 573.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-14 10:15:00 | 571.00 | 573.23 | 573.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-18 09:15:00 | 574.95 | 571.41 | 572.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-18 09:15:00 | 574.95 | 571.41 | 572.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 09:15:00 | 574.95 | 571.41 | 572.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-18 09:45:00 | 574.20 | 571.41 | 572.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-18 11:15:00 | 566.85 | 570.67 | 571.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 12:45:00 | 566.60 | 569.94 | 571.33 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 13:45:00 | 566.65 | 569.24 | 570.89 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-18 14:30:00 | 566.50 | 568.81 | 570.54 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-19 10:15:00 | 578.80 | 571.57 | 571.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 17 — BUY (started 2025-08-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 10:15:00 | 578.80 | 571.57 | 571.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-19 12:15:00 | 580.00 | 573.82 | 572.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-20 14:15:00 | 580.15 | 580.21 | 577.44 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-20 14:45:00 | 580.05 | 580.21 | 577.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-21 12:15:00 | 578.85 | 580.06 | 578.46 | EMA400 retest candle locked (from upside) |

### Cycle 18 — SELL (started 2025-08-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 09:15:00 | 565.55 | 576.38 | 577.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 13:15:00 | 564.50 | 569.76 | 573.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-26 11:15:00 | 563.55 | 562.64 | 565.92 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-26 11:30:00 | 563.70 | 562.64 | 565.92 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 13:15:00 | 566.00 | 563.58 | 565.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:45:00 | 565.70 | 563.58 | 565.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 14:15:00 | 560.50 | 562.96 | 565.31 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-26 15:15:00 | 560.20 | 562.96 | 565.31 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-01 13:15:00 | 562.00 | 557.65 | 557.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2025-09-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-01 13:15:00 | 562.00 | 557.65 | 557.44 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 14:15:00 | 563.35 | 558.79 | 557.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-02 12:15:00 | 559.30 | 560.89 | 559.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-02 12:15:00 | 559.30 | 560.89 | 559.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 12:15:00 | 559.30 | 560.89 | 559.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 13:00:00 | 559.30 | 560.89 | 559.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 13:15:00 | 557.60 | 560.24 | 559.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-02 14:00:00 | 557.60 | 560.24 | 559.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-09-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-02 15:15:00 | 556.00 | 558.82 | 558.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 554.20 | 556.84 | 557.64 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 13:15:00 | 555.85 | 554.44 | 555.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 13:15:00 | 555.85 | 554.44 | 555.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 555.85 | 554.44 | 555.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 14:00:00 | 555.85 | 554.44 | 555.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 14:15:00 | 553.75 | 554.30 | 555.60 | EMA400 retest candle locked (from downside) |

### Cycle 21 — BUY (started 2025-09-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 11:15:00 | 560.85 | 556.86 | 556.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-09 10:15:00 | 561.80 | 559.46 | 558.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-09 11:15:00 | 558.25 | 559.22 | 558.16 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-09 11:15:00 | 558.25 | 559.22 | 558.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 11:15:00 | 558.25 | 559.22 | 558.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:00:00 | 558.25 | 559.22 | 558.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 12:15:00 | 558.50 | 559.08 | 558.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 12:45:00 | 558.30 | 559.08 | 558.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 13:15:00 | 558.50 | 558.96 | 558.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 13:30:00 | 557.90 | 558.96 | 558.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 14:15:00 | 557.70 | 558.71 | 558.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-09 15:00:00 | 557.70 | 558.71 | 558.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-09 15:15:00 | 558.80 | 558.73 | 558.23 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-10 09:15:00 | 559.60 | 558.73 | 558.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-23 10:15:00 | 580.35 | 585.97 | 586.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — SELL (started 2025-09-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-23 10:15:00 | 580.35 | 585.97 | 586.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-25 09:15:00 | 579.15 | 581.74 | 583.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-29 09:15:00 | 571.80 | 568.86 | 572.97 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-29 10:00:00 | 571.80 | 568.86 | 572.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-30 09:15:00 | 568.60 | 568.16 | 570.53 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-30 13:00:00 | 564.30 | 567.19 | 569.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 12:15:00 | 573.90 | 568.81 | 568.96 | SL hit (close>static) qty=1.00 sl=571.20 alert=retest2 |

### Cycle 23 — BUY (started 2025-10-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 13:15:00 | 575.95 | 570.24 | 569.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-01 14:15:00 | 576.85 | 571.56 | 570.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-06 09:15:00 | 575.70 | 576.98 | 574.59 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-06 10:00:00 | 575.70 | 576.98 | 574.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 11:15:00 | 575.50 | 576.52 | 574.78 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 13:30:00 | 577.55 | 576.64 | 575.13 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-06 15:15:00 | 577.00 | 576.65 | 575.27 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-07 09:15:00 | 573.75 | 576.13 | 575.28 | SL hit (close<static) qty=1.00 sl=574.40 alert=retest2 |

### Cycle 24 — SELL (started 2025-10-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-07 11:15:00 | 569.90 | 574.19 | 574.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-07 12:15:00 | 568.75 | 573.10 | 573.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 11:15:00 | 564.55 | 563.78 | 566.78 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 12:00:00 | 564.55 | 563.78 | 566.78 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 564.50 | 562.57 | 564.93 | EMA400 retest candle locked (from downside) |

### Cycle 25 — BUY (started 2025-10-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 13:15:00 | 568.50 | 566.16 | 566.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-13 10:15:00 | 571.40 | 568.04 | 567.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-14 09:15:00 | 568.00 | 570.12 | 568.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 09:15:00 | 568.00 | 570.12 | 568.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 09:15:00 | 568.00 | 570.12 | 568.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 10:00:00 | 568.00 | 570.12 | 568.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 10:15:00 | 563.75 | 568.84 | 568.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-14 11:00:00 | 563.75 | 568.84 | 568.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 26 — SELL (started 2025-10-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 11:15:00 | 561.40 | 567.36 | 567.73 | EMA200 below EMA400 |

### Cycle 27 — BUY (started 2025-10-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 10:15:00 | 572.20 | 567.62 | 567.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-20 10:15:00 | 576.15 | 570.35 | 569.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 14:15:00 | 572.90 | 573.75 | 571.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 15:00:00 | 572.90 | 573.75 | 571.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 13:15:00 | 579.15 | 580.79 | 578.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 13:45:00 | 578.40 | 580.79 | 578.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 14:15:00 | 579.80 | 580.59 | 578.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-24 14:45:00 | 578.00 | 580.59 | 578.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-27 09:15:00 | 584.90 | 581.52 | 579.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 09:15:00 | 587.45 | 583.28 | 581.18 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 12:45:00 | 586.60 | 584.23 | 582.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 13:15:00 | 586.75 | 584.23 | 582.31 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-28 15:15:00 | 586.90 | 584.56 | 582.80 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-30 09:15:00 | 576.40 | 587.72 | 586.24 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-10-30 09:15:00 | 576.40 | 587.72 | 586.24 | SL hit (close<static) qty=1.00 sl=578.40 alert=retest2 |

### Cycle 28 — SELL (started 2025-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-30 10:15:00 | 573.30 | 584.84 | 585.07 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 11:15:00 | 570.00 | 581.87 | 583.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 09:15:00 | 574.65 | 572.89 | 575.55 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-03 09:15:00 | 574.65 | 572.89 | 575.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 09:15:00 | 574.65 | 572.89 | 575.55 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 09:15:00 | 568.50 | 573.61 | 574.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-10 10:15:00 | 572.15 | 571.22 | 571.20 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — BUY (started 2025-11-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 10:15:00 | 572.15 | 571.22 | 571.20 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-10 12:15:00 | 569.70 | 571.15 | 571.19 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-11-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 13:15:00 | 571.95 | 571.31 | 571.26 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2025-11-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 09:15:00 | 566.10 | 570.27 | 570.80 | EMA200 below EMA400 |

### Cycle 33 — BUY (started 2025-11-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 09:15:00 | 572.15 | 570.40 | 570.39 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 13:15:00 | 568.10 | 570.48 | 570.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-13 14:15:00 | 567.95 | 569.98 | 570.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 14:15:00 | 567.40 | 567.38 | 568.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-14 15:00:00 | 567.40 | 567.38 | 568.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 568.50 | 567.60 | 568.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 09:15:00 | 569.80 | 567.60 | 568.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 09:15:00 | 570.35 | 568.15 | 568.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 10:00:00 | 570.35 | 568.15 | 568.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 10:15:00 | 571.10 | 568.74 | 569.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 12:15:00 | 569.80 | 569.02 | 569.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 13:00:00 | 569.85 | 569.19 | 569.19 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-17 13:15:00 | 571.55 | 569.66 | 569.40 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 35 — BUY (started 2025-11-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-17 13:15:00 | 571.55 | 569.66 | 569.40 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-17 15:15:00 | 571.80 | 570.34 | 569.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 09:15:00 | 567.35 | 569.74 | 569.55 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-18 09:15:00 | 567.35 | 569.74 | 569.55 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 09:15:00 | 567.35 | 569.74 | 569.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-18 09:45:00 | 567.50 | 569.74 | 569.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 36 — SELL (started 2025-11-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-18 10:15:00 | 567.70 | 569.34 | 569.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-19 09:15:00 | 565.40 | 567.76 | 568.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 09:15:00 | 551.40 | 547.69 | 550.63 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-25 09:15:00 | 551.40 | 547.69 | 550.63 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 09:15:00 | 551.40 | 547.69 | 550.63 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:00:00 | 551.40 | 547.69 | 550.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 10:15:00 | 547.95 | 547.74 | 550.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 10:30:00 | 550.65 | 547.74 | 550.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 548.30 | 548.31 | 549.85 | EMA400 retest candle locked (from downside) |

### Cycle 37 — BUY (started 2025-11-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 11:15:00 | 555.70 | 551.49 | 550.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-26 12:15:00 | 556.95 | 552.58 | 551.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-27 10:15:00 | 552.15 | 553.86 | 552.73 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 10:15:00 | 552.15 | 553.86 | 552.73 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 10:15:00 | 552.15 | 553.86 | 552.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 10:45:00 | 552.45 | 553.86 | 552.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 11:15:00 | 550.00 | 553.09 | 552.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-27 11:45:00 | 550.20 | 553.09 | 552.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — SELL (started 2025-11-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-27 13:15:00 | 549.55 | 551.81 | 551.97 | EMA200 below EMA400 |

### Cycle 39 — BUY (started 2025-12-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-02 09:15:00 | 552.90 | 551.34 | 551.22 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-02 10:15:00 | 555.00 | 552.07 | 551.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-03 09:15:00 | 547.00 | 552.59 | 552.28 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 09:15:00 | 547.00 | 552.59 | 552.28 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 09:15:00 | 547.00 | 552.59 | 552.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-03 10:00:00 | 547.00 | 552.59 | 552.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 40 — SELL (started 2025-12-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 10:15:00 | 549.45 | 551.96 | 552.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-05 10:15:00 | 543.00 | 546.99 | 548.76 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-09 11:15:00 | 532.35 | 530.16 | 535.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-09 12:00:00 | 532.35 | 530.16 | 535.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 13:15:00 | 534.55 | 531.81 | 535.43 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 14:15:00 | 533.25 | 531.81 | 535.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 09:15:00 | 537.15 | 533.40 | 535.30 | SL hit (close>static) qty=1.00 sl=535.90 alert=retest2 |

### Cycle 41 — BUY (started 2025-12-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 11:15:00 | 528.55 | 527.66 | 527.60 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2025-12-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-19 12:15:00 | 526.65 | 527.46 | 527.52 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2025-12-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-19 13:15:00 | 530.00 | 527.97 | 527.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-19 14:15:00 | 532.45 | 528.86 | 528.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-23 11:15:00 | 533.70 | 534.18 | 532.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-23 12:00:00 | 533.70 | 534.18 | 532.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 534.45 | 534.40 | 532.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-23 14:45:00 | 533.55 | 534.40 | 532.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-26 14:15:00 | 538.90 | 539.54 | 537.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-26 14:45:00 | 539.50 | 539.54 | 537.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 09:15:00 | 536.55 | 539.06 | 537.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 10:00:00 | 536.55 | 539.06 | 537.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 10:15:00 | 535.30 | 538.31 | 537.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 11:00:00 | 535.30 | 538.31 | 537.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 14:15:00 | 539.40 | 537.93 | 537.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-29 14:45:00 | 538.95 | 537.93 | 537.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-29 15:15:00 | 538.25 | 537.99 | 537.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:15:00 | 536.25 | 537.99 | 537.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 09:15:00 | 537.85 | 537.96 | 537.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 09:30:00 | 537.40 | 537.96 | 537.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 10:15:00 | 538.20 | 538.01 | 537.73 | EMA400 retest candle locked (from upside) |

### Cycle 44 — SELL (started 2025-12-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 12:15:00 | 535.45 | 537.45 | 537.52 | EMA200 below EMA400 |

### Cycle 45 — BUY (started 2025-12-31 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 13:15:00 | 541.35 | 537.95 | 537.61 | EMA200 above EMA400 |

### Cycle 46 — SELL (started 2026-01-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-01 10:15:00 | 535.85 | 537.36 | 537.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 12:15:00 | 534.85 | 536.68 | 537.14 | Break + close below crossover candle low |

### Cycle 47 — BUY (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-02 09:15:00 | 543.45 | 537.71 | 537.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-02 10:15:00 | 546.55 | 539.48 | 538.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-05 10:15:00 | 542.60 | 543.49 | 541.43 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-05 11:00:00 | 542.60 | 543.49 | 541.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 11:15:00 | 542.95 | 543.38 | 541.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:45:00 | 544.10 | 541.87 | 541.34 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 11:30:00 | 543.45 | 542.03 | 541.51 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-06 13:15:00 | 539.40 | 541.42 | 541.33 | SL hit (close<static) qty=1.00 sl=541.00 alert=retest2 |

### Cycle 48 — SELL (started 2026-01-06 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 14:15:00 | 540.00 | 541.14 | 541.20 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 09:15:00 | 542.90 | 541.47 | 541.34 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2026-01-07 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 10:15:00 | 539.55 | 541.09 | 541.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-07 11:15:00 | 536.95 | 540.26 | 540.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-09 09:15:00 | 530.30 | 529.41 | 533.37 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-09 10:00:00 | 530.30 | 529.41 | 533.37 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 10:15:00 | 539.15 | 531.36 | 533.89 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:00:00 | 539.15 | 531.36 | 533.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 11:15:00 | 534.05 | 531.90 | 533.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-09 11:30:00 | 535.95 | 531.90 | 533.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-09 12:15:00 | 535.40 | 532.60 | 534.04 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-09 14:00:00 | 533.30 | 532.74 | 533.97 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-14 10:15:00 | 536.00 | 525.00 | 524.62 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 51 — BUY (started 2026-01-14 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-14 10:15:00 | 536.00 | 525.00 | 524.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-14 15:15:00 | 536.45 | 531.79 | 528.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-16 13:15:00 | 532.95 | 533.92 | 531.07 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-16 14:00:00 | 532.95 | 533.92 | 531.07 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 534.50 | 533.66 | 531.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 09:30:00 | 532.25 | 533.66 | 531.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 533.50 | 533.63 | 531.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:30:00 | 533.00 | 533.63 | 531.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 12:15:00 | 532.65 | 533.33 | 531.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 12:45:00 | 531.40 | 533.33 | 531.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 14:15:00 | 533.35 | 533.14 | 532.11 | EMA400 retest candle locked (from upside) |

### Cycle 52 — SELL (started 2026-01-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 10:15:00 | 526.95 | 531.50 | 531.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 11:15:00 | 525.25 | 530.25 | 531.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-22 11:15:00 | 517.05 | 515.33 | 519.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-22 12:00:00 | 517.05 | 515.33 | 519.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 09:15:00 | 520.00 | 517.17 | 519.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:00:00 | 520.00 | 517.17 | 519.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 10:15:00 | 517.20 | 517.17 | 518.96 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-23 10:30:00 | 518.40 | 517.17 | 518.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 15:15:00 | 514.50 | 508.83 | 511.36 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 09:15:00 | 516.45 | 508.83 | 511.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 513.30 | 509.72 | 511.54 | EMA400 retest candle locked (from downside) |

### Cycle 53 — BUY (started 2026-01-28 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 12:15:00 | 517.60 | 513.24 | 512.85 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 14:15:00 | 518.90 | 514.83 | 513.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-01 11:15:00 | 520.65 | 524.10 | 522.18 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-01 11:15:00 | 520.65 | 524.10 | 522.18 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 11:15:00 | 520.65 | 524.10 | 522.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:00:00 | 520.65 | 524.10 | 522.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 12:15:00 | 513.40 | 521.96 | 521.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-01 12:30:00 | 511.60 | 521.96 | 521.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 54 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 513.35 | 520.24 | 520.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 508.25 | 517.84 | 519.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-03 09:15:00 | 505.85 | 500.94 | 507.14 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-03 09:15:00 | 505.85 | 500.94 | 507.14 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 505.85 | 500.94 | 507.14 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2026-02-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 09:15:00 | 515.75 | 508.66 | 508.58 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-05 09:15:00 | 518.60 | 514.41 | 511.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-05 15:15:00 | 514.00 | 514.86 | 513.32 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-06 09:15:00 | 515.40 | 514.86 | 513.32 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 09:15:00 | 511.80 | 514.25 | 513.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-06 10:00:00 | 511.80 | 514.25 | 513.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-06 10:15:00 | 517.15 | 514.83 | 513.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 11:15:00 | 518.65 | 514.83 | 513.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 12:15:00 | 518.65 | 515.41 | 513.93 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 13:45:00 | 518.15 | 516.28 | 514.61 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-06 14:30:00 | 518.50 | 517.01 | 515.10 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 10:15:00 | 519.00 | 518.00 | 516.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-09 13:15:00 | 521.60 | 518.38 | 516.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 10:15:00 | 520.80 | 521.55 | 520.31 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-11 11:30:00 | 521.45 | 521.81 | 520.65 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-12 10:30:00 | 520.75 | 522.24 | 521.50 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-12 12:15:00 | 517.60 | 520.52 | 520.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 56 — SELL (started 2026-02-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 12:15:00 | 517.60 | 520.52 | 520.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 15:15:00 | 515.65 | 518.96 | 519.97 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-16 13:15:00 | 510.00 | 509.27 | 512.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-16 13:30:00 | 509.30 | 509.27 | 512.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-16 14:15:00 | 511.05 | 509.62 | 512.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-16 15:00:00 | 511.05 | 509.62 | 512.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 513.45 | 510.74 | 512.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 513.45 | 510.74 | 512.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 514.90 | 511.57 | 512.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:30:00 | 515.00 | 511.57 | 512.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 57 — BUY (started 2026-02-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-17 12:15:00 | 520.10 | 514.42 | 513.66 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-18 13:15:00 | 522.50 | 519.96 | 517.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 14:15:00 | 519.45 | 522.99 | 520.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-19 14:15:00 | 519.45 | 522.99 | 520.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 14:15:00 | 519.45 | 522.99 | 520.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 15:00:00 | 519.45 | 522.99 | 520.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 15:15:00 | 518.10 | 522.01 | 520.57 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 09:30:00 | 525.05 | 523.18 | 521.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-02 09:15:00 | 529.40 | 538.50 | 538.95 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2026-03-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-02 09:15:00 | 529.40 | 538.50 | 538.95 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-04 09:15:00 | 517.70 | 528.08 | 532.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 12:15:00 | 518.15 | 518.13 | 522.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 13:00:00 | 518.15 | 518.13 | 522.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 520.50 | 518.03 | 521.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 15:00:00 | 520.50 | 518.03 | 521.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 520.10 | 518.44 | 521.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 525.95 | 518.44 | 521.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 522.65 | 519.28 | 521.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:30:00 | 524.10 | 519.28 | 521.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 516.75 | 518.78 | 521.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-06 14:45:00 | 512.30 | 516.93 | 519.67 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-11 09:15:00 | 516.55 | 510.86 | 510.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 59 — BUY (started 2026-03-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 09:15:00 | 516.55 | 510.86 | 510.29 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2026-03-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 14:15:00 | 504.90 | 510.34 | 510.50 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-12 09:15:00 | 495.30 | 506.46 | 508.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 11:15:00 | 501.90 | 500.35 | 503.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 11:15:00 | 501.90 | 500.35 | 503.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 11:15:00 | 501.90 | 500.35 | 503.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-13 11:30:00 | 502.65 | 500.35 | 503.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 501.50 | 496.45 | 498.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 14:45:00 | 503.10 | 496.45 | 498.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 500.00 | 497.16 | 499.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 09:15:00 | 497.55 | 497.16 | 499.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 14:15:00 | 499.35 | 496.03 | 497.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-17 14:45:00 | 500.00 | 496.03 | 497.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 15:15:00 | 498.50 | 496.52 | 497.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:15:00 | 504.35 | 496.52 | 497.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 504.45 | 498.11 | 498.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 09:45:00 | 506.15 | 498.11 | 498.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 61 — BUY (started 2026-03-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 10:15:00 | 503.25 | 499.14 | 498.62 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 493.90 | 499.52 | 499.65 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 491.45 | 497.29 | 498.58 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 471.10 | 468.13 | 476.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 471.10 | 468.13 | 476.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 471.10 | 468.13 | 476.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-24 10:30:00 | 467.30 | 468.10 | 475.89 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-24 12:15:00 | 479.80 | 471.04 | 475.91 | SL hit (close>static) qty=1.00 sl=478.70 alert=retest2 |

### Cycle 63 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 495.00 | 478.57 | 478.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 10:15:00 | 497.90 | 482.44 | 479.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-30 11:15:00 | 502.55 | 503.75 | 497.90 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-03-30 11:45:00 | 502.30 | 503.75 | 497.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 13:15:00 | 498.00 | 502.21 | 498.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 14:00:00 | 498.00 | 502.21 | 498.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 14:15:00 | 495.90 | 500.95 | 497.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-30 15:00:00 | 495.90 | 500.95 | 497.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-30 15:15:00 | 496.30 | 500.02 | 497.83 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-01 09:15:00 | 510.50 | 500.02 | 497.83 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 11:15:00 | 523.20 | 526.62 | 526.72 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 64 — SELL (started 2026-04-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 11:15:00 | 523.20 | 526.62 | 526.72 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-13 12:15:00 | 519.80 | 525.25 | 526.09 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-15 09:15:00 | 530.00 | 524.44 | 525.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-15 09:15:00 | 530.00 | 524.44 | 525.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-15 09:15:00 | 530.00 | 524.44 | 525.23 | EMA400 retest candle locked (from downside) |

### Cycle 65 — BUY (started 2026-04-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 11:15:00 | 528.00 | 525.93 | 525.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-15 15:15:00 | 530.05 | 527.08 | 526.40 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 13:15:00 | 537.65 | 538.57 | 536.03 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 15:15:00 | 535.75 | 537.75 | 536.08 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 535.75 | 537.75 | 536.08 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:15:00 | 552.30 | 537.75 | 536.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-23 14:15:00 | 545.40 | 551.34 | 552.02 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2026-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-23 14:15:00 | 545.40 | 551.34 | 552.02 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 15:15:00 | 545.00 | 550.07 | 551.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 13:15:00 | 544.15 | 539.40 | 542.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 13:15:00 | 544.15 | 539.40 | 542.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 13:15:00 | 544.15 | 539.40 | 542.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 14:00:00 | 544.15 | 539.40 | 542.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 14:15:00 | 543.05 | 540.13 | 542.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 09:15:00 | 541.15 | 540.80 | 542.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-28 09:15:00 | 547.05 | 542.05 | 542.83 | SL hit (close>static) qty=1.00 sl=545.40 alert=retest2 |

### Cycle 67 — BUY (started 2026-04-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-28 10:15:00 | 548.65 | 543.37 | 543.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-28 14:15:00 | 550.60 | 546.27 | 544.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-30 09:15:00 | 551.90 | 557.35 | 552.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 09:15:00 | 551.90 | 557.35 | 552.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 09:15:00 | 551.90 | 557.35 | 552.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:00:00 | 551.90 | 557.35 | 552.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 10:15:00 | 553.05 | 556.49 | 552.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 10:30:00 | 551.50 | 556.49 | 552.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 11:15:00 | 552.25 | 555.64 | 552.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 11:45:00 | 551.00 | 555.64 | 552.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 12:15:00 | 553.45 | 555.20 | 552.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-30 12:30:00 | 552.45 | 555.20 | 552.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 554.80 | 555.10 | 553.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 09:15:00 | 559.20 | 555.10 | 553.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 14:00:00 | 556.95 | 556.15 | 554.71 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-04 15:00:00 | 557.35 | 556.39 | 554.95 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 09:30:00 | 556.60 | 555.85 | 554.95 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 10:15:00 | 553.25 | 555.33 | 554.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 10:30:00 | 552.20 | 555.33 | 554.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2026-05-05 11:15:00 | 552.50 | 554.77 | 554.59 | SL hit (close<static) qty=1.00 sl=553.00 alert=retest2 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| SELL | retest2 | 2025-05-12 10:00:00 | 591.15 | 2025-05-12 12:15:00 | 595.00 | STOP_HIT | 1.00 | -0.65% |
| BUY | retest2 | 2025-05-14 15:00:00 | 618.60 | 2025-05-19 11:15:00 | 604.85 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-05-15 12:45:00 | 617.00 | 2025-05-19 11:15:00 | 604.85 | STOP_HIT | 1.00 | -1.97% |
| BUY | retest2 | 2025-05-15 13:15:00 | 617.40 | 2025-05-19 11:15:00 | 604.85 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2025-05-16 10:30:00 | 618.60 | 2025-05-19 11:15:00 | 604.85 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-05-28 09:15:00 | 602.15 | 2025-05-28 10:15:00 | 598.00 | STOP_HIT | 1.00 | -0.69% |
| SELL | retest2 | 2025-05-30 10:30:00 | 588.60 | 2025-06-02 09:15:00 | 598.95 | STOP_HIT | 1.00 | -1.76% |
| SELL | retest2 | 2025-05-30 11:45:00 | 588.05 | 2025-06-02 09:15:00 | 598.95 | STOP_HIT | 1.00 | -1.85% |
| BUY | retest2 | 2025-06-05 10:00:00 | 606.85 | 2025-06-05 15:15:00 | 599.55 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2025-06-11 10:30:00 | 626.10 | 2025-06-12 11:15:00 | 612.35 | STOP_HIT | 1.00 | -2.20% |
| BUY | retest2 | 2025-06-11 11:15:00 | 625.25 | 2025-06-12 11:15:00 | 612.35 | STOP_HIT | 1.00 | -2.06% |
| SELL | retest2 | 2025-06-17 12:30:00 | 603.85 | 2025-06-24 09:15:00 | 604.75 | STOP_HIT | 1.00 | -0.15% |
| SELL | retest2 | 2025-06-18 11:00:00 | 603.50 | 2025-06-24 09:15:00 | 604.75 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-06-18 12:15:00 | 602.60 | 2025-06-24 09:15:00 | 604.75 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-06-27 14:45:00 | 612.50 | 2025-07-01 11:15:00 | 609.40 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2025-06-30 09:15:00 | 619.30 | 2025-07-01 11:15:00 | 609.40 | STOP_HIT | 1.00 | -1.60% |
| BUY | retest2 | 2025-07-01 09:45:00 | 612.75 | 2025-07-01 11:15:00 | 609.40 | STOP_HIT | 1.00 | -0.55% |
| SELL | retest2 | 2025-07-11 10:45:00 | 600.85 | 2025-07-14 09:15:00 | 610.25 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-07-24 10:30:00 | 617.25 | 2025-07-29 10:15:00 | 586.39 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-24 10:30:00 | 617.25 | 2025-07-29 12:15:00 | 596.10 | STOP_HIT | 0.50 | 3.43% |
| SELL | retest2 | 2025-07-25 09:30:00 | 614.00 | 2025-07-31 09:15:00 | 583.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-25 09:30:00 | 614.00 | 2025-08-04 09:15:00 | 585.20 | STOP_HIT | 0.50 | 4.69% |
| SELL | retest2 | 2025-08-08 10:15:00 | 573.90 | 2025-08-13 15:15:00 | 574.80 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-08-11 11:00:00 | 573.50 | 2025-08-13 15:15:00 | 574.80 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-08-11 13:00:00 | 574.45 | 2025-08-13 15:15:00 | 574.80 | STOP_HIT | 1.00 | -0.06% |
| SELL | retest2 | 2025-08-11 15:00:00 | 573.90 | 2025-08-13 15:15:00 | 574.80 | STOP_HIT | 1.00 | -0.16% |
| SELL | retest2 | 2025-08-12 10:30:00 | 573.05 | 2025-08-13 15:15:00 | 574.80 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-08-12 14:00:00 | 573.00 | 2025-08-13 15:15:00 | 574.80 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-08-12 15:00:00 | 573.15 | 2025-08-13 15:15:00 | 574.80 | STOP_HIT | 1.00 | -0.29% |
| SELL | retest2 | 2025-08-13 11:45:00 | 571.80 | 2025-08-13 15:15:00 | 574.80 | STOP_HIT | 1.00 | -0.52% |
| SELL | retest2 | 2025-08-18 12:45:00 | 566.60 | 2025-08-19 10:15:00 | 578.80 | STOP_HIT | 1.00 | -2.15% |
| SELL | retest2 | 2025-08-18 13:45:00 | 566.65 | 2025-08-19 10:15:00 | 578.80 | STOP_HIT | 1.00 | -2.14% |
| SELL | retest2 | 2025-08-18 14:30:00 | 566.50 | 2025-08-19 10:15:00 | 578.80 | STOP_HIT | 1.00 | -2.17% |
| SELL | retest2 | 2025-08-26 15:15:00 | 560.20 | 2025-09-01 13:15:00 | 562.00 | STOP_HIT | 1.00 | -0.32% |
| BUY | retest2 | 2025-09-10 09:15:00 | 559.60 | 2025-09-23 10:15:00 | 580.35 | STOP_HIT | 1.00 | 3.71% |
| SELL | retest2 | 2025-09-30 13:00:00 | 564.30 | 2025-10-01 12:15:00 | 573.90 | STOP_HIT | 1.00 | -1.70% |
| BUY | retest2 | 2025-10-06 13:30:00 | 577.55 | 2025-10-07 09:15:00 | 573.75 | STOP_HIT | 1.00 | -0.66% |
| BUY | retest2 | 2025-10-06 15:15:00 | 577.00 | 2025-10-07 09:15:00 | 573.75 | STOP_HIT | 1.00 | -0.56% |
| BUY | retest2 | 2025-10-28 09:15:00 | 587.45 | 2025-10-30 09:15:00 | 576.40 | STOP_HIT | 1.00 | -1.88% |
| BUY | retest2 | 2025-10-28 12:45:00 | 586.60 | 2025-10-30 09:15:00 | 576.40 | STOP_HIT | 1.00 | -1.74% |
| BUY | retest2 | 2025-10-28 13:15:00 | 586.75 | 2025-10-30 09:15:00 | 576.40 | STOP_HIT | 1.00 | -1.76% |
| BUY | retest2 | 2025-10-28 15:15:00 | 586.90 | 2025-10-30 09:15:00 | 576.40 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-11-06 09:15:00 | 568.50 | 2025-11-10 10:15:00 | 572.15 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-11-17 12:15:00 | 569.80 | 2025-11-17 13:15:00 | 571.55 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2025-11-17 13:00:00 | 569.85 | 2025-11-17 13:15:00 | 571.55 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2025-12-09 14:15:00 | 533.25 | 2025-12-10 09:15:00 | 537.15 | STOP_HIT | 1.00 | -0.73% |
| SELL | retest2 | 2025-12-10 10:45:00 | 533.05 | 2025-12-19 11:15:00 | 528.55 | STOP_HIT | 1.00 | 0.84% |
| SELL | retest2 | 2025-12-11 10:45:00 | 533.40 | 2025-12-19 11:15:00 | 528.55 | STOP_HIT | 1.00 | 0.91% |
| SELL | retest2 | 2025-12-11 11:30:00 | 532.50 | 2025-12-19 11:15:00 | 528.55 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest2 | 2025-12-12 10:15:00 | 533.25 | 2025-12-19 11:15:00 | 528.55 | STOP_HIT | 1.00 | 0.88% |
| SELL | retest2 | 2025-12-12 14:15:00 | 533.00 | 2025-12-19 11:15:00 | 528.55 | STOP_HIT | 1.00 | 0.83% |
| BUY | retest2 | 2026-01-06 09:45:00 | 544.10 | 2026-01-06 13:15:00 | 539.40 | STOP_HIT | 1.00 | -0.86% |
| BUY | retest2 | 2026-01-06 11:30:00 | 543.45 | 2026-01-06 13:15:00 | 539.40 | STOP_HIT | 1.00 | -0.75% |
| SELL | retest2 | 2026-01-09 14:00:00 | 533.30 | 2026-01-14 10:15:00 | 536.00 | STOP_HIT | 1.00 | -0.51% |
| BUY | retest2 | 2026-02-06 11:15:00 | 518.65 | 2026-02-12 12:15:00 | 517.60 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2026-02-06 12:15:00 | 518.65 | 2026-02-12 12:15:00 | 517.60 | STOP_HIT | 1.00 | -0.20% |
| BUY | retest2 | 2026-02-06 13:45:00 | 518.15 | 2026-02-12 12:15:00 | 517.60 | STOP_HIT | 1.00 | -0.11% |
| BUY | retest2 | 2026-02-06 14:30:00 | 518.50 | 2026-02-12 12:15:00 | 517.60 | STOP_HIT | 1.00 | -0.17% |
| BUY | retest2 | 2026-02-09 13:15:00 | 521.60 | 2026-02-12 12:15:00 | 517.60 | STOP_HIT | 1.00 | -0.77% |
| BUY | retest2 | 2026-02-11 10:15:00 | 520.80 | 2026-02-12 12:15:00 | 517.60 | STOP_HIT | 1.00 | -0.61% |
| BUY | retest2 | 2026-02-11 11:30:00 | 521.45 | 2026-02-12 12:15:00 | 517.60 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2026-02-12 10:30:00 | 520.75 | 2026-02-12 12:15:00 | 517.60 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2026-02-20 09:30:00 | 525.05 | 2026-03-02 09:15:00 | 529.40 | STOP_HIT | 1.00 | 0.83% |
| SELL | retest2 | 2026-03-06 14:45:00 | 512.30 | 2026-03-11 09:15:00 | 516.55 | STOP_HIT | 1.00 | -0.83% |
| SELL | retest2 | 2026-03-24 10:30:00 | 467.30 | 2026-03-24 12:15:00 | 479.80 | STOP_HIT | 1.00 | -2.67% |
| BUY | retest2 | 2026-04-01 09:15:00 | 510.50 | 2026-04-13 11:15:00 | 523.20 | STOP_HIT | 1.00 | 2.49% |
| BUY | retest2 | 2026-04-21 09:15:00 | 552.30 | 2026-04-23 14:15:00 | 545.40 | STOP_HIT | 1.00 | -1.25% |
| SELL | retest2 | 2026-04-28 09:15:00 | 541.15 | 2026-04-28 09:15:00 | 547.05 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2026-05-04 09:15:00 | 559.20 | 2026-05-05 11:15:00 | 552.50 | STOP_HIT | 1.00 | -1.20% |
| BUY | retest2 | 2026-05-04 14:00:00 | 556.95 | 2026-05-05 11:15:00 | 552.50 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2026-05-04 15:00:00 | 557.35 | 2026-05-05 11:15:00 | 552.50 | STOP_HIT | 1.00 | -0.87% |
| BUY | retest2 | 2026-05-05 09:30:00 | 556.60 | 2026-05-05 11:15:00 | 552.50 | STOP_HIT | 1.00 | -0.74% |
