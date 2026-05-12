# Marico Ltd. (MARICO)

## Backtest Summary

- **Window:** 2024-03-13 09:15:00 → 2026-05-08 15:15:00 (3710 bars)
- **Last close:** 830.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 166 |
| ALERT1 | 104 |
| ALERT2 | 102 |
| ALERT2_SKIP | 57 |
| ALERT3 | 294 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 2 |
| ENTRY2 | 138 |
| PARTIAL | 4 |
| TARGET_HIT | 0 |
| STOP_HIT | 139 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 143 (incl. partial bookings)
- **Trades open at end:** 1
- **Winners / losers:** 38 / 105
- **Target hits / Stop hits / Partials:** 0 / 139 / 4
- **Avg / median % per leg:** -0.14% / -0.74%
- **Sum % (uncompounded):** -20.14%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 79 | 17 | 21.5% | 0 | 79 | 0 | -0.33% | -26.4% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 79 | 17 | 21.5% | 0 | 79 | 0 | -0.33% | -26.4% |
| SELL (all) | 64 | 21 | 32.8% | 0 | 60 | 4 | 0.10% | 6.2% |
| SELL @ 2nd Alert (retest1) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.43% | -0.9% |
| SELL @ 3rd Alert (retest2) | 62 | 20 | 32.3% | 0 | 58 | 4 | 0.11% | 7.1% |
| retest1 (combined) | 2 | 1 | 50.0% | 0 | 2 | 0 | -0.43% | -0.9% |
| retest2 (combined) | 141 | 37 | 26.2% | 0 | 137 | 4 | -0.14% | -19.3% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — SELL (started 2024-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-16 13:15:00 | 587.20 | 593.00 | 593.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-17 09:15:00 | 583.95 | 590.97 | 592.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-17 11:15:00 | 592.30 | 590.74 | 591.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-17 11:15:00 | 592.30 | 590.74 | 591.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 11:15:00 | 592.30 | 590.74 | 591.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 12:00:00 | 592.30 | 590.74 | 591.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 12:15:00 | 594.10 | 591.41 | 592.00 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 13:00:00 | 594.10 | 591.41 | 592.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-17 13:15:00 | 594.50 | 592.03 | 592.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-17 13:45:00 | 596.20 | 592.03 | 592.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 2 — BUY (started 2024-05-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-17 14:15:00 | 597.30 | 593.08 | 592.69 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-17 15:15:00 | 599.00 | 594.27 | 593.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-18 09:15:00 | 592.80 | 593.97 | 593.22 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-18 09:15:00 | 592.80 | 593.97 | 593.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 09:15:00 | 592.80 | 593.97 | 593.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-18 11:45:00 | 592.95 | 593.98 | 593.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-18 12:15:00 | 596.00 | 594.38 | 593.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:15:00 | 589.90 | 594.38 | 593.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 09:15:00 | 589.55 | 593.42 | 593.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-21 09:45:00 | 588.35 | 593.42 | 593.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 3 — SELL (started 2024-05-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-21 10:15:00 | 588.75 | 592.48 | 592.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-21 12:15:00 | 587.00 | 590.68 | 591.86 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-21 14:15:00 | 592.05 | 590.47 | 591.53 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-21 14:15:00 | 592.05 | 590.47 | 591.53 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 14:15:00 | 592.05 | 590.47 | 591.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-21 15:00:00 | 592.05 | 590.47 | 591.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 590.90 | 590.55 | 591.47 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-22 09:15:00 | 599.60 | 590.55 | 591.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 4 — BUY (started 2024-05-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-22 09:15:00 | 600.00 | 592.44 | 592.25 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-23 09:15:00 | 609.70 | 601.91 | 598.07 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-24 09:15:00 | 601.00 | 605.54 | 602.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-24 09:15:00 | 601.00 | 605.54 | 602.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-24 09:15:00 | 601.00 | 605.54 | 602.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 09:15:00 | 606.80 | 603.85 | 602.71 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 10:15:00 | 607.35 | 604.32 | 603.02 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-27 10:45:00 | 608.00 | 604.84 | 603.38 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-28 10:00:00 | 607.40 | 604.96 | 604.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 11:15:00 | 607.60 | 609.12 | 607.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-05-29 12:00:00 | 607.60 | 609.12 | 607.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-29 12:15:00 | 610.25 | 609.35 | 607.58 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-29 13:15:00 | 610.40 | 609.35 | 607.58 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 15:15:00 | 604.00 | 607.85 | 607.31 | SL hit (close<static) qty=1.00 sl=607.25 alert=retest2 |

### Cycle 5 — SELL (started 2024-05-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-30 09:15:00 | 595.65 | 605.41 | 606.25 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 10:15:00 | 593.40 | 603.01 | 605.08 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-05-30 15:15:00 | 601.00 | 598.96 | 601.87 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-05-31 10:15:00 | 601.45 | 599.08 | 601.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 10:15:00 | 601.45 | 599.08 | 601.39 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-05-31 11:00:00 | 601.45 | 599.08 | 601.39 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-31 11:15:00 | 601.65 | 599.59 | 601.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 12:45:00 | 597.50 | 599.30 | 601.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 14:15:00 | 599.05 | 599.65 | 601.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-05-31 15:00:00 | 595.60 | 598.84 | 600.61 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 11:00:00 | 599.05 | 599.06 | 600.25 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 11:15:00 | 599.30 | 599.11 | 600.17 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 11:45:00 | 601.50 | 599.11 | 600.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 12:15:00 | 599.00 | 599.08 | 600.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-03 13:15:00 | 600.30 | 599.08 | 600.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 13:15:00 | 599.65 | 599.20 | 600.02 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 14:45:00 | 596.85 | 598.14 | 599.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-04 09:15:00 | 606.50 | 599.31 | 599.74 | SL hit (close>static) qty=1.00 sl=602.20 alert=retest2 |

### Cycle 6 — BUY (started 2024-06-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-04 11:15:00 | 601.85 | 600.19 | 600.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-04 12:15:00 | 609.75 | 602.10 | 600.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-06 09:15:00 | 637.65 | 640.64 | 628.07 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-06 09:15:00 | 637.65 | 640.64 | 628.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-06 09:15:00 | 637.65 | 640.64 | 628.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-06 09:30:00 | 634.00 | 640.64 | 628.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 09:15:00 | 647.10 | 642.25 | 635.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 12:15:00 | 650.05 | 644.64 | 637.51 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 14:45:00 | 651.75 | 647.34 | 640.55 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 10:30:00 | 651.45 | 648.94 | 643.09 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-10 11:15:00 | 651.35 | 648.94 | 643.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-11 09:15:00 | 655.30 | 651.92 | 647.36 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2024-06-12 09:15:00 | 636.90 | 646.80 | 646.91 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — SELL (started 2024-06-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-12 09:15:00 | 636.90 | 646.80 | 646.91 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-12 10:15:00 | 630.60 | 643.56 | 645.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-14 09:15:00 | 618.30 | 615.52 | 624.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-14 09:30:00 | 614.85 | 615.52 | 624.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-18 09:15:00 | 619.60 | 618.49 | 621.79 | EMA400 retest candle locked (from downside) |

### Cycle 8 — BUY (started 2024-06-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-19 10:15:00 | 624.45 | 622.31 | 622.29 | EMA200 above EMA400 |

### Cycle 9 — SELL (started 2024-06-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-21 10:15:00 | 620.55 | 624.04 | 624.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-21 13:15:00 | 615.90 | 621.13 | 622.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-24 10:15:00 | 617.40 | 617.11 | 619.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-06-24 10:45:00 | 616.80 | 617.11 | 619.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 12:15:00 | 616.50 | 617.30 | 619.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 12:45:00 | 620.95 | 617.30 | 619.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 13:15:00 | 620.95 | 618.03 | 619.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 14:00:00 | 620.95 | 618.03 | 619.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-24 14:15:00 | 623.25 | 619.08 | 619.93 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-06-24 15:00:00 | 623.25 | 619.08 | 619.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — BUY (started 2024-06-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-25 09:15:00 | 624.05 | 620.79 | 620.61 | EMA200 above EMA400 |

### Cycle 11 — SELL (started 2024-06-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-25 14:15:00 | 615.60 | 619.91 | 620.37 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-25 15:15:00 | 614.15 | 618.76 | 619.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-26 12:15:00 | 618.10 | 617.19 | 618.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-26 12:15:00 | 618.10 | 617.19 | 618.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-26 12:15:00 | 618.10 | 617.19 | 618.54 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-26 13:30:00 | 615.60 | 616.57 | 618.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-27 10:00:00 | 615.40 | 615.86 | 617.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-28 09:15:00 | 620.25 | 614.76 | 615.78 | SL hit (close>static) qty=1.00 sl=619.90 alert=retest2 |

### Cycle 12 — BUY (started 2024-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 10:15:00 | 621.15 | 616.24 | 615.78 | EMA200 above EMA400 |

### Cycle 13 — SELL (started 2024-07-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-02 10:15:00 | 608.10 | 614.64 | 615.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-02 11:15:00 | 606.05 | 612.93 | 614.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-03 12:15:00 | 607.85 | 607.00 | 609.62 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-03 13:00:00 | 607.85 | 607.00 | 609.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 15:15:00 | 608.00 | 607.19 | 609.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-04 09:15:00 | 609.20 | 607.19 | 609.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-04 09:15:00 | 607.15 | 607.18 | 608.87 | EMA400 retest candle locked (from downside) |

### Cycle 14 — BUY (started 2024-07-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-05 11:15:00 | 611.40 | 609.06 | 608.96 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-05 12:15:00 | 616.40 | 610.53 | 609.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-09 10:15:00 | 632.10 | 635.76 | 628.26 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-07-09 11:00:00 | 632.10 | 635.76 | 628.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 637.95 | 637.85 | 633.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 634.55 | 637.85 | 633.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-11 09:15:00 | 643.05 | 643.97 | 639.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-11 09:30:00 | 643.10 | 643.97 | 639.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 09:15:00 | 641.35 | 643.33 | 641.05 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 09:30:00 | 640.35 | 643.33 | 641.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 10:15:00 | 642.95 | 643.25 | 641.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 10:30:00 | 640.20 | 643.25 | 641.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 642.75 | 643.15 | 641.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 11:30:00 | 641.75 | 643.15 | 641.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 11:15:00 | 670.50 | 675.00 | 669.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 11:45:00 | 669.80 | 675.00 | 669.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 12:15:00 | 668.00 | 673.60 | 669.54 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 12:30:00 | 669.20 | 673.60 | 669.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 13:15:00 | 667.40 | 672.36 | 669.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 14:00:00 | 667.40 | 672.36 | 669.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 14:15:00 | 668.85 | 671.66 | 669.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-19 14:30:00 | 667.90 | 671.66 | 669.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-19 15:15:00 | 669.35 | 671.20 | 669.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:15:00 | 668.85 | 671.20 | 669.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 09:15:00 | 669.95 | 670.95 | 669.36 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 09:30:00 | 665.05 | 670.95 | 669.36 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 10:15:00 | 669.60 | 670.68 | 669.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 10:45:00 | 670.05 | 670.68 | 669.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 11:15:00 | 667.55 | 670.05 | 669.22 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-22 12:00:00 | 667.55 | 670.05 | 669.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-22 12:15:00 | 667.45 | 669.53 | 669.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-22 13:15:00 | 669.45 | 669.53 | 669.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-22 15:15:00 | 665.40 | 668.40 | 668.64 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — SELL (started 2024-07-22 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-22 15:15:00 | 665.40 | 668.40 | 668.64 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-23 09:15:00 | 664.55 | 667.63 | 668.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-23 11:15:00 | 672.45 | 668.58 | 668.59 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-23 11:15:00 | 672.45 | 668.58 | 668.59 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-23 11:15:00 | 672.45 | 668.58 | 668.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-23 12:00:00 | 672.45 | 668.58 | 668.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 16 — BUY (started 2024-07-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-23 12:15:00 | 674.15 | 669.69 | 669.09 | EMA200 above EMA400 |

### Cycle 17 — SELL (started 2024-07-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-24 09:15:00 | 653.95 | 667.72 | 668.55 | EMA200 below EMA400 |

### Cycle 18 — BUY (started 2024-07-25 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-25 12:15:00 | 670.40 | 665.76 | 665.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 13:15:00 | 677.15 | 668.04 | 666.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-30 11:15:00 | 679.70 | 681.45 | 678.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-30 11:15:00 | 679.70 | 681.45 | 678.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 11:15:00 | 679.70 | 681.45 | 678.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-30 12:00:00 | 679.70 | 681.45 | 678.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-30 12:15:00 | 682.70 | 681.70 | 678.79 | EMA400 retest candle locked (from upside) |

### Cycle 19 — SELL (started 2024-07-31 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 11:15:00 | 672.35 | 677.94 | 678.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 09:15:00 | 667.90 | 673.58 | 675.29 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-05 09:15:00 | 673.85 | 667.50 | 670.45 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-05 09:15:00 | 673.85 | 667.50 | 670.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 09:15:00 | 673.85 | 667.50 | 670.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-05 10:00:00 | 673.85 | 667.50 | 670.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-05 10:15:00 | 674.30 | 668.86 | 670.80 | EMA400 retest candle locked (from downside) |

### Cycle 20 — BUY (started 2024-08-05 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-05 14:15:00 | 673.45 | 671.71 | 671.70 | EMA200 above EMA400 |

### Cycle 21 — SELL (started 2024-08-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-06 09:15:00 | 646.20 | 666.65 | 669.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-06 12:15:00 | 638.95 | 655.31 | 663.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-07 10:15:00 | 642.65 | 641.85 | 652.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-07 10:45:00 | 642.65 | 641.85 | 652.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-07 15:15:00 | 650.00 | 645.39 | 650.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-08 09:15:00 | 653.00 | 645.39 | 650.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-08 09:15:00 | 654.00 | 647.11 | 650.47 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 10:15:00 | 647.55 | 647.11 | 650.47 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 13:45:00 | 650.95 | 648.29 | 649.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-09 09:15:00 | 658.05 | 651.54 | 651.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 22 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 658.05 | 651.54 | 651.11 | EMA200 above EMA400 |

### Cycle 23 — SELL (started 2024-08-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-12 09:15:00 | 640.45 | 649.57 | 650.51 | EMA200 below EMA400 |

### Cycle 24 — BUY (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 09:15:00 | 661.30 | 649.67 | 649.62 | EMA200 above EMA400 |

### Cycle 25 — SELL (started 2024-08-14 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-14 13:15:00 | 650.65 | 651.72 | 651.77 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-14 15:15:00 | 648.15 | 650.73 | 651.30 | Break + close below crossover candle low |

### Cycle 26 — BUY (started 2024-08-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-16 09:15:00 | 657.20 | 652.03 | 651.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-16 14:15:00 | 661.45 | 655.31 | 653.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 10:15:00 | 666.20 | 667.55 | 663.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-20 10:45:00 | 665.40 | 667.55 | 663.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 663.60 | 666.76 | 663.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 12:00:00 | 663.60 | 666.76 | 663.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 664.65 | 666.34 | 663.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 12:45:00 | 662.90 | 666.34 | 663.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 680.65 | 680.76 | 676.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 10:00:00 | 680.65 | 680.76 | 676.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 11:15:00 | 678.05 | 680.11 | 677.18 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:00:00 | 678.05 | 680.11 | 677.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 12:15:00 | 675.50 | 679.19 | 677.03 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 12:45:00 | 676.30 | 679.19 | 677.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 13:15:00 | 671.90 | 677.73 | 676.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 14:00:00 | 671.90 | 677.73 | 676.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 15:15:00 | 682.90 | 678.44 | 677.07 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 09:30:00 | 689.80 | 680.40 | 678.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 11:15:00 | 688.70 | 681.66 | 678.87 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-08-26 15:00:00 | 691.00 | 684.36 | 681.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-27 14:15:00 | 675.30 | 680.07 | 680.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — SELL (started 2024-08-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 14:15:00 | 675.30 | 680.07 | 680.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-28 09:15:00 | 669.90 | 677.86 | 679.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-02 14:15:00 | 651.65 | 649.58 | 654.09 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-02 15:00:00 | 651.65 | 649.58 | 654.09 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 14:15:00 | 646.40 | 640.87 | 644.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-04 15:00:00 | 646.40 | 640.87 | 644.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-04 15:15:00 | 647.35 | 642.17 | 644.79 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-09-05 09:15:00 | 641.20 | 642.17 | 644.79 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-06 09:15:00 | 661.45 | 646.82 | 645.53 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 28 — BUY (started 2024-09-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-06 09:15:00 | 661.45 | 646.82 | 645.53 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-09 11:15:00 | 674.50 | 666.23 | 658.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-10 15:15:00 | 677.95 | 678.84 | 672.34 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-09-11 09:15:00 | 682.50 | 678.84 | 672.34 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 09:15:00 | 682.90 | 684.11 | 681.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 10:00:00 | 682.90 | 684.11 | 681.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 10:15:00 | 682.00 | 683.69 | 681.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 11:00:00 | 682.00 | 683.69 | 681.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 11:15:00 | 679.80 | 682.91 | 681.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-13 12:00:00 | 679.80 | 682.91 | 681.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-13 12:15:00 | 679.25 | 682.18 | 681.13 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-09-13 13:30:00 | 680.20 | 682.28 | 681.27 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-09-24 11:15:00 | 695.85 | 699.78 | 699.87 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 29 — SELL (started 2024-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 11:15:00 | 695.85 | 699.78 | 699.87 | EMA200 below EMA400 |

### Cycle 30 — BUY (started 2024-09-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-24 12:15:00 | 704.85 | 700.80 | 700.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-24 13:15:00 | 706.80 | 702.00 | 700.91 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-25 09:15:00 | 695.50 | 701.67 | 701.12 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-25 09:15:00 | 695.50 | 701.67 | 701.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-25 09:15:00 | 695.50 | 701.67 | 701.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-25 10:00:00 | 695.50 | 701.67 | 701.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 31 — SELL (started 2024-09-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-25 10:15:00 | 693.50 | 700.03 | 700.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-25 11:15:00 | 683.65 | 696.76 | 698.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 11:15:00 | 692.85 | 690.55 | 693.88 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-09-26 12:00:00 | 692.85 | 690.55 | 693.88 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 12:15:00 | 694.85 | 691.41 | 693.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 12:45:00 | 696.30 | 691.41 | 693.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 13:15:00 | 688.25 | 690.78 | 693.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 13:30:00 | 697.40 | 690.78 | 693.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 14:15:00 | 692.20 | 691.06 | 693.34 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 15:00:00 | 692.20 | 691.06 | 693.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 15:15:00 | 692.25 | 691.30 | 693.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:15:00 | 709.85 | 691.30 | 693.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-27 09:15:00 | 703.90 | 693.82 | 694.21 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-27 09:30:00 | 706.45 | 693.82 | 694.21 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 32 — BUY (started 2024-09-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-27 10:15:00 | 706.10 | 696.28 | 695.29 | EMA200 above EMA400 |

### Cycle 33 — SELL (started 2024-09-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-30 10:15:00 | 691.40 | 695.62 | 695.74 | EMA200 below EMA400 |

### Cycle 34 — BUY (started 2024-09-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-30 11:15:00 | 699.90 | 696.47 | 696.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-09-30 13:15:00 | 703.45 | 698.57 | 697.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-09-30 14:15:00 | 694.75 | 697.81 | 696.95 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 14:15:00 | 694.75 | 697.81 | 696.95 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 14:15:00 | 694.75 | 697.81 | 696.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-09-30 15:00:00 | 694.75 | 697.81 | 696.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 695.00 | 697.24 | 696.78 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:15:00 | 691.70 | 697.24 | 696.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 35 — SELL (started 2024-10-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-01 09:15:00 | 687.75 | 695.35 | 695.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-01 10:15:00 | 685.95 | 693.47 | 695.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-01 13:15:00 | 693.10 | 691.87 | 693.78 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-01 13:15:00 | 693.10 | 691.87 | 693.78 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 13:15:00 | 693.10 | 691.87 | 693.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 14:00:00 | 693.10 | 691.87 | 693.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 14:15:00 | 693.90 | 692.27 | 693.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 15:00:00 | 693.90 | 692.27 | 693.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 15:15:00 | 691.55 | 692.13 | 693.59 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-03 09:15:00 | 700.65 | 692.13 | 693.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-03 09:15:00 | 696.20 | 692.94 | 693.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-03 10:15:00 | 693.00 | 692.94 | 693.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-03 11:15:00 | 700.10 | 694.61 | 694.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 36 — BUY (started 2024-10-03 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-03 11:15:00 | 700.10 | 694.61 | 694.44 | EMA200 above EMA400 |

### Cycle 37 — SELL (started 2024-10-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-04 13:15:00 | 688.80 | 694.75 | 695.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-07 09:15:00 | 681.00 | 690.60 | 693.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-08 09:15:00 | 688.05 | 683.41 | 687.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-08 09:15:00 | 688.05 | 683.41 | 687.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 09:15:00 | 688.05 | 683.41 | 687.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:00:00 | 688.05 | 683.41 | 687.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 10:15:00 | 696.35 | 686.00 | 688.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 10:45:00 | 696.30 | 686.00 | 688.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-08 11:15:00 | 697.15 | 688.23 | 688.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-08 12:00:00 | 697.15 | 688.23 | 688.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 38 — BUY (started 2024-10-08 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-08 12:15:00 | 694.15 | 689.41 | 689.34 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-09 13:15:00 | 700.20 | 696.69 | 693.84 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-09 15:15:00 | 696.70 | 696.85 | 694.42 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-10-10 09:15:00 | 692.30 | 696.85 | 694.42 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 09:15:00 | 695.85 | 696.65 | 694.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 09:30:00 | 690.20 | 696.65 | 694.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 10:15:00 | 689.85 | 695.29 | 694.12 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 11:00:00 | 689.85 | 695.29 | 694.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-10 11:15:00 | 688.55 | 693.94 | 693.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-10 12:00:00 | 688.55 | 693.94 | 693.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 39 — SELL (started 2024-10-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-10 12:15:00 | 687.55 | 692.66 | 693.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-10 14:15:00 | 685.25 | 690.06 | 691.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-11 10:15:00 | 689.85 | 688.43 | 690.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-11 10:15:00 | 689.85 | 688.43 | 690.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 10:15:00 | 689.85 | 688.43 | 690.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:00:00 | 689.85 | 688.43 | 690.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 11:15:00 | 690.60 | 688.86 | 690.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 11:45:00 | 692.75 | 688.86 | 690.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-11 12:15:00 | 687.15 | 688.52 | 690.14 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-11 12:30:00 | 690.70 | 688.52 | 690.14 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-14 09:15:00 | 678.55 | 685.52 | 688.14 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-14 10:15:00 | 675.80 | 685.52 | 688.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-10-14 14:15:00 | 690.00 | 685.91 | 687.15 | SL hit (close>static) qty=1.00 sl=688.50 alert=retest2 |

### Cycle 40 — BUY (started 2024-10-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-30 10:15:00 | 651.75 | 638.67 | 637.48 | EMA200 above EMA400 |

### Cycle 41 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 624.80 | 638.30 | 639.97 | EMA200 below EMA400 |

### Cycle 42 — BUY (started 2024-11-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-06 10:15:00 | 641.90 | 636.26 | 636.17 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 12:15:00 | 646.00 | 639.24 | 637.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-07 09:15:00 | 642.10 | 642.95 | 640.23 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-07 10:00:00 | 642.10 | 642.95 | 640.23 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 10:15:00 | 635.50 | 641.46 | 639.80 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 10:30:00 | 633.65 | 641.46 | 639.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-07 11:15:00 | 635.05 | 640.18 | 639.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-07 12:00:00 | 635.05 | 640.18 | 639.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 43 — SELL (started 2024-11-07 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-07 13:15:00 | 633.15 | 638.00 | 638.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-08 11:15:00 | 628.75 | 633.89 | 636.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-13 10:15:00 | 601.70 | 601.43 | 610.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-11-13 11:15:00 | 601.45 | 601.43 | 610.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 09:15:00 | 596.75 | 594.61 | 596.94 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-19 14:45:00 | 592.15 | 594.95 | 596.42 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-11-22 13:15:00 | 600.00 | 593.86 | 593.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 44 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 600.00 | 593.86 | 593.70 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-25 09:15:00 | 602.15 | 597.11 | 595.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-28 14:15:00 | 645.30 | 648.62 | 638.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-28 15:00:00 | 645.30 | 648.62 | 638.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 640.85 | 646.33 | 639.03 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 10:45:00 | 645.60 | 646.61 | 639.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 15:15:00 | 645.45 | 642.88 | 641.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-12-03 13:15:00 | 638.90 | 641.34 | 641.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 45 — SELL (started 2024-12-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-03 13:15:00 | 638.90 | 641.34 | 641.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-04 09:15:00 | 638.05 | 640.56 | 641.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-05 14:15:00 | 633.40 | 631.97 | 634.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-05 14:30:00 | 631.25 | 631.97 | 634.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-05 15:15:00 | 632.00 | 631.98 | 634.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 09:15:00 | 636.10 | 631.98 | 634.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 09:15:00 | 637.30 | 633.04 | 634.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-06 10:00:00 | 637.30 | 633.04 | 634.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-06 10:15:00 | 636.50 | 633.73 | 634.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-06 11:15:00 | 635.20 | 633.73 | 634.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-09 09:15:00 | 603.44 | 628.18 | 631.89 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-10 12:15:00 | 611.30 | 611.07 | 617.22 | SL hit (close>ema200) qty=0.50 sl=611.07 alert=retest2 |

### Cycle 46 — BUY (started 2024-12-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-11 11:15:00 | 631.80 | 620.81 | 619.57 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-11 13:15:00 | 632.90 | 624.64 | 621.61 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-12 12:15:00 | 632.00 | 633.81 | 628.33 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-12-12 13:00:00 | 632.00 | 633.81 | 628.33 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-13 09:15:00 | 633.70 | 634.61 | 630.54 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-13 11:15:00 | 638.25 | 635.07 | 631.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 10:30:00 | 638.50 | 637.83 | 634.91 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 11:15:00 | 638.15 | 637.83 | 634.91 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-16 12:15:00 | 637.80 | 637.61 | 635.08 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 12:15:00 | 635.85 | 637.26 | 635.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:00:00 | 635.85 | 637.26 | 635.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-16 13:15:00 | 639.90 | 637.79 | 635.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-16 13:30:00 | 637.00 | 637.79 | 635.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 12:15:00 | 636.60 | 639.52 | 637.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:00:00 | 636.60 | 639.52 | 637.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 13:15:00 | 636.15 | 638.85 | 637.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 13:30:00 | 638.25 | 638.85 | 637.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-17 14:15:00 | 634.60 | 638.00 | 637.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-17 14:45:00 | 633.55 | 638.00 | 637.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-12-17 15:15:00 | 630.00 | 636.40 | 636.73 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 47 — SELL (started 2024-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-17 15:15:00 | 630.00 | 636.40 | 636.73 | EMA200 below EMA400 |

### Cycle 48 — BUY (started 2024-12-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-18 11:15:00 | 642.05 | 637.50 | 637.10 | EMA200 above EMA400 |

### Cycle 49 — SELL (started 2024-12-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-20 12:15:00 | 634.95 | 638.32 | 638.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-20 13:15:00 | 632.95 | 637.25 | 637.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-23 11:15:00 | 634.35 | 632.02 | 634.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-23 11:15:00 | 634.35 | 632.02 | 634.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 11:15:00 | 634.35 | 632.02 | 634.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 12:00:00 | 634.35 | 632.02 | 634.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 12:15:00 | 631.00 | 631.81 | 634.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 12:30:00 | 635.10 | 631.81 | 634.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 14:15:00 | 636.50 | 632.74 | 634.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-23 15:00:00 | 636.50 | 632.74 | 634.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-23 15:15:00 | 636.25 | 633.44 | 634.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-24 09:15:00 | 643.25 | 633.44 | 634.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 50 — BUY (started 2024-12-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 10:15:00 | 641.45 | 635.74 | 635.34 | EMA200 above EMA400 |

### Cycle 51 — SELL (started 2024-12-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 13:15:00 | 634.20 | 635.12 | 635.13 | EMA200 below EMA400 |

### Cycle 52 — BUY (started 2024-12-24 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-24 14:15:00 | 636.00 | 635.30 | 635.21 | EMA200 above EMA400 |

### Cycle 53 — SELL (started 2024-12-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-24 15:15:00 | 629.90 | 634.22 | 634.72 | EMA200 below EMA400 |

### Cycle 54 — BUY (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 09:15:00 | 634.65 | 633.18 | 633.12 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-12-30 10:15:00 | 640.45 | 634.63 | 633.79 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-12-31 09:15:00 | 637.05 | 637.98 | 636.11 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-31 09:15:00 | 637.05 | 637.98 | 636.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 09:15:00 | 637.05 | 637.98 | 636.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 09:45:00 | 638.60 | 637.98 | 636.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 10:15:00 | 637.00 | 637.78 | 636.19 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 10:45:00 | 638.25 | 637.78 | 636.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 11:15:00 | 635.20 | 637.27 | 636.10 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-12-31 12:00:00 | 635.20 | 637.27 | 636.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 637.70 | 637.35 | 636.24 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-31 13:45:00 | 639.50 | 637.82 | 636.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-01 10:30:00 | 638.75 | 639.34 | 637.76 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-06 10:00:00 | 643.80 | 654.40 | 650.93 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-06 15:15:00 | 647.00 | 649.17 | 649.29 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 55 — SELL (started 2025-01-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 15:15:00 | 647.00 | 649.17 | 649.29 | EMA200 below EMA400 |

### Cycle 56 — BUY (started 2025-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-07 09:15:00 | 652.75 | 649.89 | 649.60 | EMA200 above EMA400 |

### Cycle 57 — SELL (started 2025-01-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-07 14:15:00 | 644.55 | 649.17 | 649.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-07 15:15:00 | 639.75 | 647.28 | 648.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-09 09:15:00 | 652.70 | 642.30 | 643.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-09 09:15:00 | 652.70 | 642.30 | 643.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-09 09:15:00 | 652.70 | 642.30 | 643.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-09 10:00:00 | 652.70 | 642.30 | 643.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 58 — BUY (started 2025-01-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-09 10:15:00 | 660.75 | 645.99 | 645.50 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-09 12:15:00 | 664.20 | 652.36 | 648.65 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-13 10:15:00 | 670.00 | 671.47 | 665.02 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-13 11:00:00 | 670.00 | 671.47 | 665.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-13 15:15:00 | 664.45 | 669.74 | 666.63 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-14 09:15:00 | 663.55 | 669.74 | 666.63 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 09:15:00 | 666.15 | 669.02 | 666.59 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-14 09:30:00 | 667.25 | 669.02 | 666.59 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 10:15:00 | 661.95 | 667.61 | 666.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-14 10:45:00 | 659.85 | 667.61 | 666.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-14 11:15:00 | 657.45 | 665.57 | 665.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-14 12:00:00 | 657.45 | 665.57 | 665.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — SELL (started 2025-01-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-14 12:15:00 | 661.30 | 664.72 | 665.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-15 09:15:00 | 651.95 | 660.25 | 662.62 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-15 14:15:00 | 660.00 | 659.34 | 661.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-15 15:00:00 | 660.00 | 659.34 | 661.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-15 15:15:00 | 659.75 | 659.42 | 661.04 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-16 09:15:00 | 658.75 | 659.42 | 661.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-16 09:15:00 | 657.70 | 659.08 | 660.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-16 10:30:00 | 652.70 | 657.41 | 659.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-17 10:15:00 | 666.45 | 657.34 | 657.88 | SL hit (close>static) qty=1.00 sl=662.80 alert=retest2 |

### Cycle 60 — BUY (started 2025-01-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-17 12:15:00 | 659.50 | 658.30 | 658.26 | EMA200 above EMA400 |

### Cycle 61 — SELL (started 2025-01-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 14:15:00 | 657.30 | 658.86 | 658.94 | EMA200 below EMA400 |

### Cycle 62 — BUY (started 2025-01-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-21 09:15:00 | 662.55 | 659.40 | 659.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-21 14:15:00 | 666.40 | 662.17 | 660.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-22 12:15:00 | 664.15 | 664.34 | 662.48 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-22 12:15:00 | 664.15 | 664.34 | 662.48 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 12:15:00 | 664.15 | 664.34 | 662.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-22 12:30:00 | 663.45 | 664.34 | 662.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-22 13:15:00 | 663.65 | 664.20 | 662.59 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 14:15:00 | 665.65 | 664.20 | 662.59 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-22 14:45:00 | 665.40 | 664.70 | 662.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-23 09:30:00 | 667.60 | 665.88 | 663.79 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 10:00:00 | 666.85 | 667.77 | 666.06 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 10:15:00 | 668.50 | 667.92 | 666.28 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 11:30:00 | 672.10 | 668.29 | 666.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 12:15:00 | 669.95 | 668.29 | 666.60 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-24 13:00:00 | 672.35 | 669.10 | 667.13 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-01-27 09:15:00 | 661.40 | 666.28 | 666.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 63 — SELL (started 2025-01-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-27 09:15:00 | 661.40 | 666.28 | 666.33 | EMA200 below EMA400 |

### Cycle 64 — BUY (started 2025-01-29 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-29 10:15:00 | 673.85 | 665.09 | 663.92 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-29 12:15:00 | 676.50 | 668.63 | 665.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-30 11:15:00 | 671.40 | 672.82 | 669.62 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-30 12:00:00 | 671.40 | 672.82 | 669.62 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 13:15:00 | 670.60 | 672.35 | 669.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 13:45:00 | 668.65 | 672.35 | 669.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 14:15:00 | 671.35 | 672.15 | 670.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-30 15:00:00 | 671.35 | 672.15 | 670.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-30 15:15:00 | 680.00 | 673.72 | 670.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-31 09:15:00 | 676.80 | 673.72 | 670.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-31 09:15:00 | 671.65 | 673.31 | 671.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 12:30:00 | 682.15 | 675.84 | 672.81 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-01-31 13:45:00 | 683.80 | 675.40 | 672.88 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-02-01 09:15:00 | 668.50 | 672.72 | 672.17 | SL hit (close<static) qty=1.00 sl=670.00 alert=retest2 |

### Cycle 65 — SELL (started 2025-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 13:15:00 | 672.60 | 677.29 | 677.33 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-05 09:15:00 | 664.05 | 673.19 | 674.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-05 13:15:00 | 672.00 | 671.55 | 673.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-05 14:00:00 | 672.00 | 671.55 | 673.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 09:15:00 | 672.60 | 670.62 | 672.31 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 09:45:00 | 673.15 | 670.62 | 672.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 10:15:00 | 675.50 | 671.60 | 672.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-06 10:30:00 | 675.50 | 671.60 | 672.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-06 11:15:00 | 669.55 | 671.19 | 672.32 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-06 12:15:00 | 666.40 | 671.19 | 672.32 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 10:15:00 | 633.08 | 641.14 | 648.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-12 12:15:00 | 633.55 | 631.85 | 637.98 | SL hit (close>ema200) qty=0.50 sl=631.85 alert=retest2 |

### Cycle 66 — BUY (started 2025-02-19 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-19 12:15:00 | 631.20 | 627.97 | 627.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-19 13:15:00 | 632.60 | 628.89 | 628.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 627.45 | 636.89 | 634.45 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 627.45 | 636.89 | 634.45 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 627.45 | 636.89 | 634.45 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 627.45 | 636.89 | 634.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 623.10 | 634.13 | 633.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 11:00:00 | 623.10 | 634.13 | 633.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 67 — SELL (started 2025-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 11:15:00 | 619.75 | 631.26 | 632.18 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-28 09:15:00 | 608.70 | 619.82 | 621.89 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-04 15:15:00 | 588.00 | 587.77 | 594.78 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-03-05 10:00:00 | 585.70 | 587.36 | 593.95 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-05 15:15:00 | 592.35 | 589.49 | 592.30 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-03-05 15:15:00 | 592.35 | 589.49 | 592.30 | SL hit (close>ema400) qty=1.00 sl=592.30 alert=retest1 |

### Cycle 68 — BUY (started 2025-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-06 10:15:00 | 607.95 | 595.65 | 594.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-06 11:15:00 | 610.00 | 598.52 | 596.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-10 10:15:00 | 608.70 | 609.80 | 606.04 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-10 11:00:00 | 608.70 | 609.80 | 606.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 607.25 | 609.29 | 606.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 607.25 | 609.29 | 606.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 12:15:00 | 608.95 | 609.22 | 606.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:30:00 | 608.15 | 609.22 | 606.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 14:15:00 | 607.55 | 608.77 | 606.68 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 14:45:00 | 606.75 | 608.77 | 606.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 15:15:00 | 605.85 | 608.19 | 606.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:15:00 | 594.20 | 608.19 | 606.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 09:15:00 | 605.55 | 607.66 | 606.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-11 09:30:00 | 598.95 | 607.66 | 606.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — SELL (started 2025-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-11 11:15:00 | 601.85 | 605.58 | 605.71 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-11 12:15:00 | 599.85 | 604.44 | 605.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-11 15:15:00 | 605.00 | 603.94 | 604.70 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-11 15:15:00 | 605.00 | 603.94 | 604.70 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-11 15:15:00 | 605.00 | 603.94 | 604.70 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-12 09:15:00 | 602.75 | 603.94 | 604.70 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-12 09:15:00 | 600.70 | 603.29 | 604.34 | EMA400 retest candle locked (from downside) |

### Cycle 70 — BUY (started 2025-03-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-13 10:15:00 | 610.85 | 603.98 | 603.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-13 11:15:00 | 611.20 | 605.42 | 604.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-13 14:15:00 | 606.10 | 606.48 | 605.28 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-13 15:00:00 | 606.10 | 606.48 | 605.28 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-13 15:15:00 | 606.10 | 606.41 | 605.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-17 09:15:00 | 606.95 | 606.41 | 605.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-17 09:15:00 | 605.70 | 606.26 | 605.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-18 09:15:00 | 617.50 | 606.87 | 606.03 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-03-25 14:15:00 | 624.60 | 628.76 | 629.01 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 71 — SELL (started 2025-03-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-25 14:15:00 | 624.60 | 628.76 | 629.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-25 15:15:00 | 622.95 | 627.60 | 628.46 | Break + close below crossover candle low |

### Cycle 72 — BUY (started 2025-03-26 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-26 09:15:00 | 636.65 | 629.41 | 629.20 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-27 09:15:00 | 643.00 | 637.33 | 633.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-28 13:15:00 | 649.95 | 650.70 | 645.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-28 13:45:00 | 649.65 | 650.70 | 645.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 09:15:00 | 655.70 | 652.19 | 647.32 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-03 15:00:00 | 661.00 | 654.68 | 652.95 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-04 09:30:00 | 664.60 | 657.28 | 654.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 11:30:00 | 661.50 | 666.32 | 663.10 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-07 13:00:00 | 660.80 | 665.21 | 662.89 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 14:15:00 | 662.65 | 664.29 | 662.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-07 15:15:00 | 662.00 | 664.29 | 662.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-07 15:15:00 | 662.00 | 663.83 | 662.77 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-08 09:30:00 | 665.00 | 664.93 | 663.37 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-21 11:15:00 | 702.00 | 710.82 | 711.22 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — SELL (started 2025-04-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-21 11:15:00 | 702.00 | 710.82 | 711.22 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-21 12:15:00 | 699.80 | 708.62 | 710.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-22 09:15:00 | 704.95 | 702.85 | 706.47 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-22 10:00:00 | 704.95 | 702.85 | 706.47 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 10:15:00 | 708.50 | 703.98 | 706.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-22 11:00:00 | 708.50 | 703.98 | 706.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 11:15:00 | 711.65 | 705.52 | 707.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-22 12:00:00 | 711.65 | 705.52 | 707.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 12:15:00 | 707.70 | 705.95 | 707.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-22 12:30:00 | 708.60 | 705.95 | 707.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-22 13:15:00 | 706.90 | 706.14 | 707.14 | EMA400 retest candle locked (from downside) |

### Cycle 74 — BUY (started 2025-04-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-23 09:15:00 | 712.05 | 708.05 | 707.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-24 09:15:00 | 721.50 | 714.26 | 711.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-24 10:15:00 | 706.05 | 712.61 | 710.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-24 10:15:00 | 706.05 | 712.61 | 710.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 10:15:00 | 706.05 | 712.61 | 710.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-24 11:00:00 | 706.05 | 712.61 | 710.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-24 11:15:00 | 710.75 | 712.24 | 710.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-24 12:30:00 | 714.45 | 712.75 | 711.22 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 10:00:00 | 712.05 | 712.90 | 711.83 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 10:30:00 | 712.85 | 713.03 | 711.99 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-25 11:00:00 | 713.55 | 713.03 | 711.99 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-25 11:15:00 | 712.35 | 712.90 | 712.02 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-04-25 15:15:00 | 710.05 | 711.32 | 711.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 75 — SELL (started 2025-04-25 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-25 15:15:00 | 710.05 | 711.32 | 711.46 | EMA200 below EMA400 |

### Cycle 76 — BUY (started 2025-04-28 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-28 09:15:00 | 716.20 | 712.30 | 711.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-28 10:15:00 | 717.95 | 713.43 | 712.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-28 15:15:00 | 715.05 | 715.82 | 714.24 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-28 15:15:00 | 715.05 | 715.82 | 714.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-28 15:15:00 | 715.05 | 715.82 | 714.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-29 09:15:00 | 717.15 | 715.82 | 714.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 718.65 | 716.39 | 714.64 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-29 11:45:00 | 722.10 | 718.34 | 715.87 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-04-29 14:15:00 | 710.75 | 715.66 | 715.15 | SL hit (close<static) qty=1.00 sl=713.95 alert=retest2 |

### Cycle 77 — SELL (started 2025-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-29 15:15:00 | 710.25 | 714.58 | 714.71 | EMA200 below EMA400 |

### Cycle 78 — BUY (started 2025-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-30 09:15:00 | 717.85 | 715.23 | 714.99 | EMA200 above EMA400 |

### Cycle 79 — SELL (started 2025-04-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-30 10:15:00 | 710.70 | 714.33 | 714.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-02 11:15:00 | 703.30 | 709.28 | 711.66 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-05 09:15:00 | 723.10 | 707.18 | 709.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-05 09:15:00 | 723.10 | 707.18 | 709.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-05 09:15:00 | 723.10 | 707.18 | 709.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-05 09:45:00 | 729.90 | 707.18 | 709.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 80 — BUY (started 2025-05-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 11:15:00 | 727.30 | 713.08 | 711.54 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-07 10:15:00 | 733.00 | 723.32 | 719.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-08 10:15:00 | 730.55 | 731.44 | 726.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-08 10:30:00 | 729.20 | 731.44 | 726.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 11:15:00 | 723.55 | 729.86 | 726.24 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 12:00:00 | 723.55 | 729.86 | 726.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 12:15:00 | 724.90 | 728.87 | 726.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 12:30:00 | 726.35 | 728.87 | 726.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 13:15:00 | 720.20 | 727.13 | 725.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-08 14:00:00 | 720.20 | 727.13 | 725.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 81 — SELL (started 2025-05-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 15:15:00 | 718.00 | 724.29 | 724.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-09 10:15:00 | 716.05 | 721.73 | 723.23 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-09 13:15:00 | 721.50 | 721.07 | 722.50 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-09 13:45:00 | 720.75 | 721.07 | 722.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 14:15:00 | 722.70 | 721.39 | 722.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-09 15:00:00 | 722.70 | 721.39 | 722.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-09 15:15:00 | 721.00 | 721.31 | 722.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 09:15:00 | 730.50 | 721.31 | 722.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 82 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 733.40 | 723.73 | 723.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 14:15:00 | 738.20 | 731.01 | 727.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-13 11:15:00 | 730.10 | 732.85 | 729.71 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-13 11:15:00 | 730.10 | 732.85 | 729.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 11:15:00 | 730.10 | 732.85 | 729.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 11:45:00 | 730.35 | 732.85 | 729.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 12:15:00 | 727.30 | 731.74 | 729.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 12:45:00 | 726.70 | 731.74 | 729.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 13:15:00 | 727.25 | 730.84 | 729.28 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-13 13:30:00 | 725.90 | 730.84 | 729.28 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-13 15:15:00 | 727.60 | 729.68 | 729.00 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-14 09:15:00 | 730.40 | 729.68 | 729.00 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-14 09:15:00 | 729.00 | 729.55 | 729.00 | EMA400 retest candle locked (from upside) |

### Cycle 83 — SELL (started 2025-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-14 11:15:00 | 725.80 | 728.44 | 728.57 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-14 12:15:00 | 722.75 | 727.30 | 728.04 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-15 10:15:00 | 726.00 | 723.87 | 725.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 10:15:00 | 726.00 | 723.87 | 725.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 726.00 | 723.87 | 725.80 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:00:00 | 726.00 | 723.87 | 725.80 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 730.15 | 725.13 | 726.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-15 12:00:00 | 730.15 | 725.13 | 726.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 12:15:00 | 726.55 | 725.41 | 726.23 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-15 14:00:00 | 723.45 | 725.02 | 725.98 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 11:00:00 | 725.00 | 723.63 | 724.87 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-16 14:15:00 | 724.95 | 724.54 | 725.00 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 13:15:00 | 706.80 | 705.83 | 705.77 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 84 — BUY (started 2025-05-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 13:15:00 | 706.80 | 705.83 | 705.77 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-26 14:15:00 | 712.30 | 707.13 | 706.37 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-27 09:15:00 | 705.90 | 707.73 | 706.83 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-27 09:15:00 | 705.90 | 707.73 | 706.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 09:15:00 | 705.90 | 707.73 | 706.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-27 10:00:00 | 705.90 | 707.73 | 706.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-27 10:15:00 | 709.70 | 708.13 | 707.09 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-27 11:15:00 | 712.00 | 708.13 | 707.09 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-02 09:15:00 | 707.60 | 716.31 | 716.92 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 85 — SELL (started 2025-06-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-02 09:15:00 | 707.60 | 716.31 | 716.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-02 11:15:00 | 703.35 | 712.25 | 714.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-03 13:15:00 | 704.20 | 703.72 | 707.76 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-03 14:00:00 | 704.20 | 703.72 | 707.76 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 09:15:00 | 704.00 | 703.95 | 706.87 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-06 12:15:00 | 699.40 | 703.09 | 704.15 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 09:15:00 | 705.00 | 702.50 | 702.24 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 86 — BUY (started 2025-06-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 09:15:00 | 705.00 | 702.50 | 702.24 | EMA200 above EMA400 |

### Cycle 87 — SELL (started 2025-06-10 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-10 10:15:00 | 699.90 | 701.98 | 702.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-10 12:15:00 | 696.80 | 700.48 | 701.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-11 09:15:00 | 706.30 | 701.09 | 701.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-11 09:15:00 | 706.30 | 701.09 | 701.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 09:15:00 | 706.30 | 701.09 | 701.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-11 10:00:00 | 706.30 | 701.09 | 701.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 88 — BUY (started 2025-06-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-11 10:15:00 | 706.25 | 702.13 | 701.71 | EMA200 above EMA400 |

### Cycle 89 — SELL (started 2025-06-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-11 15:15:00 | 700.05 | 701.56 | 701.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 10:15:00 | 693.35 | 699.60 | 700.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 691.65 | 690.60 | 693.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 14:30:00 | 690.75 | 690.60 | 693.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 14:15:00 | 691.40 | 688.48 | 690.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-16 15:00:00 | 691.40 | 688.48 | 690.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-16 15:15:00 | 692.00 | 689.18 | 690.82 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-17 09:15:00 | 692.05 | 689.18 | 690.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-17 09:15:00 | 693.40 | 690.03 | 691.06 | EMA400 retest candle locked (from downside) |

### Cycle 90 — BUY (started 2025-06-17 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-17 12:15:00 | 693.75 | 691.61 | 691.60 | EMA200 above EMA400 |

### Cycle 91 — SELL (started 2025-06-17 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-17 14:15:00 | 687.60 | 690.88 | 691.28 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-17 15:15:00 | 686.00 | 689.91 | 690.80 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-18 09:15:00 | 691.55 | 690.24 | 690.86 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-18 09:15:00 | 691.55 | 690.24 | 690.86 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 09:15:00 | 691.55 | 690.24 | 690.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 09:30:00 | 693.35 | 690.24 | 690.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 10:15:00 | 690.30 | 690.25 | 690.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 11:00:00 | 690.30 | 690.25 | 690.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-18 14:15:00 | 689.60 | 689.76 | 690.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-18 15:00:00 | 689.60 | 689.76 | 690.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — BUY (started 2025-06-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-19 09:15:00 | 695.00 | 690.85 | 690.76 | EMA200 above EMA400 |

### Cycle 93 — SELL (started 2025-06-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-19 13:15:00 | 689.25 | 690.75 | 690.82 | EMA200 below EMA400 |

### Cycle 94 — BUY (started 2025-06-20 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 09:15:00 | 693.30 | 691.15 | 690.98 | EMA200 above EMA400 |

### Cycle 95 — SELL (started 2025-06-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 11:15:00 | 689.40 | 690.88 | 690.89 | EMA200 below EMA400 |

### Cycle 96 — BUY (started 2025-06-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-20 14:15:00 | 693.20 | 691.17 | 691.00 | EMA200 above EMA400 |

### Cycle 97 — SELL (started 2025-06-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-20 15:15:00 | 689.00 | 690.73 | 690.82 | EMA200 below EMA400 |

### Cycle 98 — BUY (started 2025-06-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 10:15:00 | 696.05 | 691.84 | 691.31 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-23 11:15:00 | 697.00 | 692.87 | 691.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-23 14:15:00 | 693.05 | 694.26 | 692.85 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-23 14:15:00 | 693.05 | 694.26 | 692.85 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 14:15:00 | 693.05 | 694.26 | 692.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-23 15:00:00 | 693.05 | 694.26 | 692.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-23 15:15:00 | 692.95 | 694.00 | 692.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-24 09:15:00 | 700.00 | 694.00 | 692.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 13:15:00 | 717.70 | 720.32 | 720.59 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 99 — SELL (started 2025-07-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 13:15:00 | 717.70 | 720.32 | 720.59 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 15:15:00 | 715.10 | 718.82 | 719.83 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 11:15:00 | 716.90 | 716.25 | 718.20 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 11:15:00 | 716.90 | 716.25 | 718.20 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 11:15:00 | 716.90 | 716.25 | 718.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:00:00 | 716.90 | 716.25 | 718.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 712.60 | 715.52 | 717.69 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 13:15:00 | 709.40 | 715.52 | 717.69 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-02 15:15:00 | 708.55 | 713.98 | 716.54 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 09:15:00 | 728.80 | 716.01 | 715.58 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 100 — BUY (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 09:15:00 | 728.80 | 716.01 | 715.58 | EMA200 above EMA400 |

### Cycle 101 — SELL (started 2025-07-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-10 14:15:00 | 726.90 | 728.13 | 728.14 | EMA200 below EMA400 |

### Cycle 102 — BUY (started 2025-07-11 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-11 09:15:00 | 731.20 | 728.64 | 728.37 | EMA200 above EMA400 |

### Cycle 103 — SELL (started 2025-07-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-11 14:15:00 | 725.55 | 728.10 | 728.33 | EMA200 below EMA400 |

### Cycle 104 — BUY (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-14 11:15:00 | 729.85 | 728.64 | 728.50 | EMA200 above EMA400 |

### Cycle 105 — SELL (started 2025-07-14 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 15:15:00 | 726.90 | 728.25 | 728.38 | EMA200 below EMA400 |

### Cycle 106 — BUY (started 2025-07-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 11:15:00 | 732.65 | 729.20 | 728.78 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 12:15:00 | 734.35 | 730.23 | 729.28 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-16 10:15:00 | 732.80 | 734.27 | 732.02 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-16 10:15:00 | 732.80 | 734.27 | 732.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 10:15:00 | 732.80 | 734.27 | 732.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 11:00:00 | 732.80 | 734.27 | 732.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 11:15:00 | 731.45 | 733.71 | 731.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-16 12:00:00 | 731.45 | 733.71 | 731.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-16 12:15:00 | 731.60 | 733.29 | 731.93 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-16 13:45:00 | 733.80 | 733.30 | 732.06 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-17 09:15:00 | 735.65 | 732.64 | 731.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-18 09:15:00 | 728.35 | 732.97 | 732.86 | SL hit (close<static) qty=1.00 sl=729.85 alert=retest2 |

### Cycle 107 — SELL (started 2025-07-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-18 10:15:00 | 724.60 | 731.30 | 732.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-18 12:15:00 | 721.65 | 728.32 | 730.55 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-21 09:15:00 | 725.15 | 724.42 | 727.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-21 10:00:00 | 725.15 | 724.42 | 727.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-22 09:15:00 | 719.75 | 720.45 | 723.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-22 14:30:00 | 716.25 | 718.72 | 721.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-24 09:30:00 | 712.00 | 715.66 | 718.13 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-30 14:15:00 | 703.00 | 701.06 | 700.80 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — BUY (started 2025-07-30 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-30 14:15:00 | 703.00 | 701.06 | 700.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-31 09:15:00 | 704.50 | 701.84 | 701.20 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-01 14:15:00 | 710.00 | 712.82 | 709.51 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-08-01 15:00:00 | 710.00 | 712.82 | 709.51 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-01 15:15:00 | 709.95 | 712.25 | 709.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:15:00 | 711.10 | 712.25 | 709.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 09:15:00 | 712.60 | 712.32 | 709.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 09:45:00 | 709.35 | 712.32 | 709.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 10:15:00 | 709.95 | 711.84 | 709.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:00:00 | 709.95 | 711.84 | 709.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 11:15:00 | 711.50 | 711.78 | 709.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 11:30:00 | 710.10 | 711.78 | 709.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 13:15:00 | 711.35 | 711.85 | 710.34 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-04 13:30:00 | 710.75 | 711.85 | 710.34 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-04 14:15:00 | 723.10 | 714.10 | 711.50 | EMA400 retest candle locked (from upside) |

### Cycle 109 — SELL (started 2025-08-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-07 12:15:00 | 708.70 | 712.97 | 713.45 | EMA200 below EMA400 |

### Cycle 110 — BUY (started 2025-08-07 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-07 14:15:00 | 718.75 | 713.70 | 713.67 | EMA200 above EMA400 |

### Cycle 111 — SELL (started 2025-08-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-08 09:15:00 | 709.05 | 713.47 | 713.61 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-08 14:15:00 | 703.55 | 709.29 | 711.40 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-11 12:15:00 | 705.70 | 705.66 | 708.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-08-11 12:45:00 | 706.55 | 705.66 | 708.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 14:15:00 | 705.60 | 705.55 | 707.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-11 14:45:00 | 707.90 | 705.55 | 707.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-11 15:15:00 | 708.00 | 706.04 | 707.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-12 09:30:00 | 710.40 | 706.71 | 708.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 10:15:00 | 707.50 | 706.87 | 708.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-12 15:00:00 | 703.95 | 706.77 | 707.64 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-13 09:45:00 | 705.25 | 706.58 | 707.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-13 15:15:00 | 710.25 | 707.82 | 707.52 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 112 — BUY (started 2025-08-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-13 15:15:00 | 710.25 | 707.82 | 707.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-14 12:15:00 | 711.70 | 709.21 | 708.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-19 09:15:00 | 715.85 | 717.42 | 714.31 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 715.85 | 717.42 | 714.31 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 715.85 | 717.42 | 714.31 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-19 09:30:00 | 712.30 | 717.42 | 714.31 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 716.30 | 717.20 | 714.49 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-19 11:15:00 | 718.10 | 717.20 | 714.49 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-22 12:15:00 | 732.65 | 734.97 | 735.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — SELL (started 2025-08-22 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-22 12:15:00 | 732.65 | 734.97 | 735.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-22 14:15:00 | 729.85 | 733.60 | 734.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-25 10:15:00 | 733.05 | 732.84 | 733.79 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-25 10:15:00 | 733.05 | 732.84 | 733.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-25 10:15:00 | 733.05 | 732.84 | 733.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-25 10:45:00 | 733.90 | 732.84 | 733.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-28 09:15:00 | 712.60 | 720.17 | 724.58 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-08-28 13:15:00 | 710.15 | 717.09 | 721.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-29 14:15:00 | 725.90 | 720.65 | 720.51 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 114 — BUY (started 2025-08-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-29 14:15:00 | 725.90 | 720.65 | 720.51 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-01 09:15:00 | 729.80 | 723.27 | 721.78 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-03 10:15:00 | 737.10 | 738.89 | 734.38 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-03 11:00:00 | 737.10 | 738.89 | 734.38 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 12:15:00 | 734.20 | 737.31 | 734.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 13:00:00 | 734.20 | 737.31 | 734.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 13:15:00 | 735.05 | 736.86 | 734.46 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:00:00 | 735.05 | 736.86 | 734.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 14:15:00 | 735.45 | 736.58 | 734.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-03 14:45:00 | 734.30 | 736.58 | 734.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-03 15:15:00 | 733.05 | 735.87 | 734.41 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-04 09:15:00 | 744.50 | 735.87 | 734.41 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-05 09:15:00 | 732.40 | 737.65 | 736.67 | SL hit (close<static) qty=1.00 sl=733.00 alert=retest2 |

### Cycle 115 — SELL (started 2025-09-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-05 11:15:00 | 727.45 | 734.45 | 735.32 | EMA200 below EMA400 |

### Cycle 116 — BUY (started 2025-09-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 14:15:00 | 735.00 | 732.36 | 732.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-10 15:15:00 | 736.50 | 733.19 | 732.67 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-11 12:15:00 | 733.60 | 734.60 | 733.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-11 12:15:00 | 733.60 | 734.60 | 733.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 12:15:00 | 733.60 | 734.60 | 733.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 13:00:00 | 733.60 | 734.60 | 733.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 13:15:00 | 733.35 | 734.35 | 733.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-11 14:00:00 | 733.35 | 734.35 | 733.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-11 14:15:00 | 733.65 | 734.21 | 733.60 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-11 15:15:00 | 735.40 | 734.21 | 733.60 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 10:45:00 | 735.10 | 734.83 | 734.06 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 12:45:00 | 735.25 | 735.08 | 734.30 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-09-12 13:45:00 | 735.20 | 735.18 | 734.42 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 14:15:00 | 734.55 | 735.06 | 734.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-12 14:45:00 | 733.55 | 735.06 | 734.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-12 15:15:00 | 735.70 | 735.19 | 734.55 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-15 09:15:00 | 730.00 | 735.19 | 734.55 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-09-15 09:15:00 | 728.50 | 733.85 | 734.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — SELL (started 2025-09-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-15 09:15:00 | 728.50 | 733.85 | 734.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-15 10:15:00 | 724.90 | 732.06 | 733.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-18 09:15:00 | 721.60 | 716.95 | 720.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-18 09:15:00 | 721.60 | 716.95 | 720.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 721.60 | 716.95 | 720.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 09:45:00 | 720.00 | 716.95 | 720.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 10:15:00 | 720.65 | 717.69 | 720.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-18 10:45:00 | 720.00 | 717.69 | 720.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 11:15:00 | 719.85 | 718.12 | 720.59 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 12:15:00 | 718.65 | 718.12 | 720.59 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 14:15:00 | 719.05 | 718.49 | 720.34 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-18 15:00:00 | 719.00 | 718.59 | 720.21 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 09:45:00 | 717.80 | 719.09 | 720.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-19 10:15:00 | 717.80 | 718.83 | 719.96 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 11:15:00 | 717.00 | 718.83 | 719.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-19 11:45:00 | 716.95 | 718.44 | 719.68 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-09-19 15:15:00 | 722.00 | 719.75 | 719.96 | SL hit (close>static) qty=1.00 sl=720.95 alert=retest2 |

### Cycle 118 — BUY (started 2025-09-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-22 09:15:00 | 722.75 | 720.35 | 720.21 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-22 10:15:00 | 726.75 | 721.63 | 720.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 13:15:00 | 716.30 | 721.30 | 720.94 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-22 13:15:00 | 716.30 | 721.30 | 720.94 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-22 13:15:00 | 716.30 | 721.30 | 720.94 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-22 14:00:00 | 716.30 | 721.30 | 720.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 119 — SELL (started 2025-09-22 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-22 14:15:00 | 716.75 | 720.39 | 720.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 09:15:00 | 712.00 | 718.12 | 719.46 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 11:15:00 | 711.90 | 708.51 | 712.43 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-24 12:00:00 | 711.90 | 708.51 | 712.43 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 711.90 | 709.19 | 712.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 12:45:00 | 712.60 | 709.19 | 712.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 710.10 | 709.37 | 712.18 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-24 14:15:00 | 709.65 | 709.37 | 712.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 12:30:00 | 707.75 | 708.08 | 710.17 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-01 15:15:00 | 704.00 | 699.43 | 699.28 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 120 — BUY (started 2025-10-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-01 15:15:00 | 704.00 | 699.43 | 699.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-03 10:15:00 | 708.40 | 701.94 | 700.49 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 13:15:00 | 716.45 | 716.56 | 712.46 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 14:00:00 | 716.45 | 716.56 | 712.46 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 14:15:00 | 715.00 | 716.25 | 712.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 14:30:00 | 713.60 | 716.25 | 712.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 09:15:00 | 708.70 | 714.49 | 712.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-08 10:00:00 | 708.70 | 714.49 | 712.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-08 10:15:00 | 709.20 | 713.43 | 712.19 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 12:00:00 | 711.80 | 713.11 | 712.16 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 13:00:00 | 712.00 | 712.88 | 712.14 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-08 13:45:00 | 711.90 | 712.38 | 711.98 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 15:15:00 | 710.00 | 711.65 | 711.70 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 121 — SELL (started 2025-10-08 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 15:15:00 | 710.00 | 711.65 | 711.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-09 09:15:00 | 709.95 | 711.31 | 711.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 12:15:00 | 712.20 | 710.90 | 711.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-09 12:15:00 | 712.20 | 710.90 | 711.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-09 12:15:00 | 712.20 | 710.90 | 711.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-09 13:00:00 | 712.20 | 710.90 | 711.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 122 — BUY (started 2025-10-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-09 13:15:00 | 715.95 | 711.91 | 711.68 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-09 14:15:00 | 716.20 | 712.77 | 712.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-10 11:15:00 | 715.00 | 715.16 | 713.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-10 12:00:00 | 715.00 | 715.16 | 713.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 14:15:00 | 714.15 | 714.90 | 713.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-10 15:00:00 | 714.15 | 714.90 | 713.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 15:15:00 | 712.50 | 714.42 | 713.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-13 09:15:00 | 715.20 | 714.42 | 713.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-13 09:15:00 | 714.40 | 714.41 | 713.81 | EMA400 retest candle locked (from upside) |

### Cycle 123 — SELL (started 2025-10-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 11:15:00 | 710.05 | 712.94 | 713.21 | EMA200 below EMA400 |

### Cycle 124 — BUY (started 2025-10-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-13 13:15:00 | 714.50 | 713.42 | 713.39 | EMA200 above EMA400 |

### Cycle 125 — SELL (started 2025-10-14 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-14 09:15:00 | 712.05 | 713.33 | 713.38 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-14 12:15:00 | 711.00 | 712.52 | 712.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-15 10:15:00 | 710.65 | 710.42 | 711.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-15 10:15:00 | 710.65 | 710.42 | 711.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 710.65 | 710.42 | 711.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-15 11:00:00 | 710.65 | 710.42 | 711.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 11:15:00 | 706.30 | 709.60 | 711.09 | EMA400 retest candle locked (from downside) |

### Cycle 126 — BUY (started 2025-10-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 11:15:00 | 716.85 | 711.34 | 711.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-16 12:15:00 | 718.05 | 712.68 | 711.76 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 09:15:00 | 731.95 | 731.96 | 725.56 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-20 10:00:00 | 731.95 | 731.96 | 725.56 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 727.80 | 729.70 | 726.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 15:00:00 | 727.80 | 729.70 | 726.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 727.00 | 729.16 | 726.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 13:45:00 | 724.40 | 728.10 | 726.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-21 14:15:00 | 720.95 | 726.67 | 726.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-21 14:30:00 | 720.95 | 726.67 | 726.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 12:15:00 | 727.95 | 729.65 | 727.92 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:00:00 | 727.95 | 729.65 | 727.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 13:15:00 | 728.50 | 729.42 | 727.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 13:30:00 | 728.45 | 729.42 | 727.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-23 14:15:00 | 723.70 | 728.28 | 727.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-23 15:00:00 | 723.70 | 728.28 | 727.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 127 — SELL (started 2025-10-23 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-23 15:15:00 | 722.10 | 727.04 | 727.08 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-24 09:15:00 | 720.45 | 725.72 | 726.48 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-24 11:15:00 | 726.45 | 725.81 | 726.38 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-24 11:15:00 | 726.45 | 725.81 | 726.38 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 11:15:00 | 726.45 | 725.81 | 726.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-24 11:30:00 | 726.35 | 725.81 | 726.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-24 12:15:00 | 725.75 | 725.80 | 726.33 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-24 13:30:00 | 722.20 | 725.83 | 726.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-27 09:15:00 | 728.35 | 726.39 | 726.44 | SL hit (close>static) qty=1.00 sl=727.25 alert=retest2 |

### Cycle 128 — BUY (started 2025-10-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-31 09:15:00 | 724.50 | 722.13 | 721.83 | EMA200 above EMA400 |

### Cycle 129 — SELL (started 2025-11-03 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-03 09:15:00 | 719.75 | 721.63 | 721.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 13:15:00 | 716.00 | 718.81 | 720.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-04 15:15:00 | 718.95 | 718.66 | 719.82 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-06 09:15:00 | 717.20 | 718.37 | 719.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-06 09:15:00 | 717.20 | 718.37 | 719.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-06 10:00:00 | 717.20 | 718.37 | 719.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 09:15:00 | 717.55 | 712.90 | 714.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 10:00:00 | 717.55 | 712.90 | 714.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-10 10:15:00 | 719.00 | 714.12 | 714.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-10 11:00:00 | 719.00 | 714.12 | 714.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 130 — BUY (started 2025-11-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-10 12:15:00 | 719.00 | 715.62 | 715.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-10 13:15:00 | 719.75 | 716.45 | 715.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-11 09:15:00 | 714.00 | 716.96 | 716.29 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-11 09:15:00 | 714.00 | 716.96 | 716.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-11 09:15:00 | 714.00 | 716.96 | 716.29 | EMA400 retest candle locked (from upside) |

### Cycle 131 — SELL (started 2025-11-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-11 11:15:00 | 713.80 | 715.45 | 715.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-12 09:15:00 | 709.20 | 713.17 | 714.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-12 10:15:00 | 715.00 | 713.53 | 714.44 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-12 10:15:00 | 715.00 | 713.53 | 714.44 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 10:15:00 | 715.00 | 713.53 | 714.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 10:45:00 | 715.00 | 713.53 | 714.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-12 11:15:00 | 715.35 | 713.90 | 714.53 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-12 12:00:00 | 715.35 | 713.90 | 714.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 132 — BUY (started 2025-11-12 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-12 12:15:00 | 719.20 | 714.96 | 714.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-12 13:15:00 | 721.40 | 716.25 | 715.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-14 09:15:00 | 718.25 | 720.70 | 719.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 09:15:00 | 718.25 | 720.70 | 719.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 09:15:00 | 718.25 | 720.70 | 719.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-14 13:00:00 | 728.00 | 721.35 | 719.70 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 11:15:00 | 743.50 | 749.32 | 749.44 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 133 — SELL (started 2025-11-20 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-20 11:15:00 | 743.50 | 749.32 | 749.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-20 12:15:00 | 740.30 | 747.51 | 748.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-21 13:15:00 | 742.15 | 740.82 | 743.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-21 13:45:00 | 741.75 | 740.82 | 743.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-24 09:15:00 | 737.00 | 739.32 | 742.19 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 13:00:00 | 730.70 | 733.94 | 737.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-25 15:15:00 | 728.00 | 733.46 | 736.37 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 09:15:00 | 730.05 | 733.53 | 734.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 09:15:00 | 724.10 | 715.72 | 715.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 134 — BUY (started 2025-12-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 09:15:00 | 724.10 | 715.72 | 715.42 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-05 11:15:00 | 729.00 | 719.91 | 717.47 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-08 14:15:00 | 730.40 | 730.65 | 726.60 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-08 15:00:00 | 730.40 | 730.65 | 726.60 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 15:15:00 | 727.15 | 729.95 | 726.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-09 09:15:00 | 724.40 | 729.95 | 726.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-09 09:15:00 | 726.85 | 729.33 | 726.67 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 12:00:00 | 730.85 | 729.32 | 727.11 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-09 15:00:00 | 730.70 | 730.32 | 728.19 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-10 14:15:00 | 724.90 | 727.60 | 727.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 135 — SELL (started 2025-12-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-10 14:15:00 | 724.90 | 727.60 | 727.88 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-12 09:15:00 | 720.80 | 725.47 | 726.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-12 11:15:00 | 727.00 | 725.66 | 726.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-12 11:15:00 | 727.00 | 725.66 | 726.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 11:15:00 | 727.00 | 725.66 | 726.42 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-12 13:00:00 | 723.30 | 725.19 | 726.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-15 09:15:00 | 734.55 | 727.43 | 726.89 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 136 — BUY (started 2025-12-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-15 09:15:00 | 734.55 | 727.43 | 726.89 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-15 12:15:00 | 736.00 | 731.29 | 728.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-16 12:15:00 | 738.50 | 738.96 | 734.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-16 13:00:00 | 738.50 | 738.96 | 734.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 737.90 | 738.64 | 735.95 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 10:30:00 | 739.80 | 738.92 | 736.31 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 12:15:00 | 739.15 | 738.93 | 736.56 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 09:15:00 | 739.45 | 738.29 | 737.01 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-18 11:30:00 | 739.45 | 737.75 | 737.02 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-19 09:15:00 | 737.20 | 739.58 | 738.42 | EMA400 retest candle locked (from upside) |
| Stop hit — per-position SL triggered | 2025-12-23 09:15:00 | 735.55 | 738.96 | 739.06 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 137 — SELL (started 2025-12-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 09:15:00 | 735.55 | 738.96 | 739.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 11:15:00 | 729.55 | 736.29 | 737.79 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-23 14:15:00 | 737.30 | 734.54 | 736.43 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-23 14:15:00 | 737.30 | 734.54 | 736.43 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 14:15:00 | 737.30 | 734.54 | 736.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-23 15:00:00 | 737.30 | 734.54 | 736.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 15:15:00 | 735.60 | 734.75 | 736.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 09:15:00 | 736.40 | 734.75 | 736.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 09:15:00 | 737.10 | 735.22 | 736.42 | EMA400 retest candle locked (from downside) |

### Cycle 138 — BUY (started 2025-12-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-26 11:15:00 | 738.15 | 736.29 | 736.28 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-26 12:15:00 | 739.50 | 736.94 | 736.57 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-30 11:15:00 | 744.50 | 748.22 | 745.06 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-30 11:15:00 | 744.50 | 748.22 | 745.06 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 11:15:00 | 744.50 | 748.22 | 745.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 12:00:00 | 744.50 | 748.22 | 745.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 12:15:00 | 741.15 | 746.80 | 744.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:00:00 | 741.15 | 746.80 | 744.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 13:15:00 | 741.30 | 745.70 | 744.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 13:45:00 | 741.75 | 745.70 | 744.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-30 14:15:00 | 737.05 | 743.97 | 743.73 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-30 15:00:00 | 737.05 | 743.97 | 743.73 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 139 — SELL (started 2025-12-30 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-30 15:15:00 | 738.15 | 742.81 | 743.22 | EMA200 below EMA400 |

### Cycle 140 — BUY (started 2025-12-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-31 10:15:00 | 746.20 | 743.86 | 743.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-31 14:15:00 | 750.70 | 746.78 | 745.24 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-02 11:15:00 | 756.75 | 757.12 | 753.04 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-02 11:15:00 | 756.75 | 757.12 | 753.04 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 756.75 | 757.12 | 753.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:30:00 | 756.95 | 757.12 | 753.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 12:15:00 | 753.55 | 756.40 | 753.08 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 12:30:00 | 754.40 | 756.40 | 753.08 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 13:15:00 | 756.40 | 756.40 | 753.38 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-02 13:30:00 | 754.10 | 756.40 | 753.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 760.50 | 757.67 | 754.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-05 13:15:00 | 770.20 | 760.52 | 756.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-06 09:30:00 | 768.30 | 767.73 | 761.86 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-08 11:15:00 | 757.05 | 767.19 | 768.42 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 141 — SELL (started 2026-01-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 11:15:00 | 757.05 | 767.19 | 768.42 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-09 12:15:00 | 756.25 | 760.14 | 763.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 13:15:00 | 754.70 | 753.74 | 757.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 14:00:00 | 754.70 | 753.74 | 757.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 15:15:00 | 757.00 | 754.75 | 757.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 09:15:00 | 757.45 | 754.75 | 757.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 09:15:00 | 754.60 | 754.72 | 757.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:00:00 | 753.40 | 754.46 | 756.76 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 11:30:00 | 752.65 | 754.36 | 756.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:00:00 | 753.35 | 754.16 | 756.22 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 14:00:00 | 753.20 | 753.96 | 755.94 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 14:15:00 | 756.05 | 754.38 | 755.95 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-13 15:00:00 | 756.05 | 754.38 | 755.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-13 15:15:00 | 754.20 | 754.35 | 755.79 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 09:45:00 | 756.35 | 754.51 | 755.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 756.60 | 754.92 | 755.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 11:00:00 | 756.60 | 754.92 | 755.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 11:15:00 | 755.75 | 755.09 | 755.81 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 12:15:00 | 753.65 | 755.09 | 755.81 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 754.80 | 755.03 | 755.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:30:00 | 750.55 | 754.21 | 755.28 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 09:15:00 | 760.65 | 754.07 | 754.82 | SL hit (close>static) qty=1.00 sl=760.50 alert=retest2 |

### Cycle 142 — BUY (started 2026-01-16 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 11:15:00 | 759.40 | 756.06 | 755.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-16 12:15:00 | 763.25 | 757.50 | 756.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-19 09:15:00 | 753.65 | 757.44 | 756.77 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-19 09:15:00 | 753.65 | 757.44 | 756.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 09:15:00 | 753.65 | 757.44 | 756.77 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-19 10:00:00 | 753.65 | 757.44 | 756.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-19 10:15:00 | 754.15 | 756.78 | 756.53 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-19 12:30:00 | 756.85 | 756.93 | 756.64 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 09:30:00 | 758.85 | 758.77 | 757.75 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-20 12:45:00 | 756.85 | 758.24 | 757.77 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-20 13:15:00 | 753.60 | 757.31 | 757.39 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 143 — SELL (started 2026-01-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-20 13:15:00 | 753.60 | 757.31 | 757.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-20 14:15:00 | 749.50 | 755.75 | 756.67 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 14:15:00 | 748.60 | 748.23 | 751.61 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 15:00:00 | 748.60 | 748.23 | 751.61 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 752.00 | 748.79 | 751.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 10:15:00 | 745.90 | 750.80 | 751.48 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-23 12:00:00 | 744.00 | 748.62 | 750.32 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-02 11:15:00 | 708.60 | 722.14 | 726.05 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-02 11:15:00 | 722.40 | 722.14 | 726.05 | SL hit (close>static) qty=0.50 sl=722.14 alert=retest2 |

### Cycle 144 — BUY (started 2026-02-03 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 13:15:00 | 732.70 | 726.51 | 725.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-04 09:15:00 | 737.45 | 729.98 | 727.83 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-04 14:15:00 | 733.45 | 733.60 | 730.80 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-04 14:45:00 | 734.10 | 733.60 | 730.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-04 15:15:00 | 731.40 | 733.16 | 730.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-05 09:15:00 | 748.80 | 733.16 | 730.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-13 14:15:00 | 759.30 | 763.99 | 764.54 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 145 — SELL (started 2026-02-13 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-13 14:15:00 | 759.30 | 763.99 | 764.54 | EMA200 below EMA400 |

### Cycle 146 — BUY (started 2026-02-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-16 10:15:00 | 769.00 | 765.33 | 765.00 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-02-16 13:15:00 | 770.95 | 767.57 | 766.19 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-02-19 10:15:00 | 783.55 | 788.88 | 782.55 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-02-19 11:00:00 | 783.55 | 788.88 | 782.55 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 11:15:00 | 782.25 | 787.55 | 782.53 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-02-19 12:00:00 | 782.25 | 787.55 | 782.53 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-19 12:15:00 | 785.00 | 787.04 | 782.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-20 10:15:00 | 787.10 | 782.76 | 781.82 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-27 09:15:00 | 792.90 | 802.57 | 802.78 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 147 — SELL (started 2026-02-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-27 09:15:00 | 792.90 | 802.57 | 802.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-27 10:15:00 | 791.75 | 800.41 | 801.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-05 11:15:00 | 777.00 | 774.84 | 779.70 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-05 12:00:00 | 777.00 | 774.84 | 779.70 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 13:15:00 | 776.55 | 775.59 | 779.22 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 13:30:00 | 778.50 | 775.59 | 779.22 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 14:15:00 | 777.55 | 775.98 | 779.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-05 14:45:00 | 782.15 | 775.98 | 779.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-05 15:15:00 | 780.00 | 776.78 | 779.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 09:15:00 | 787.50 | 776.78 | 779.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 09:15:00 | 789.90 | 779.41 | 780.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:00:00 | 789.90 | 779.41 | 780.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 148 — BUY (started 2026-03-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-06 10:15:00 | 786.55 | 780.83 | 780.71 | EMA200 above EMA400 |

### Cycle 149 — SELL (started 2026-03-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 10:15:00 | 775.70 | 781.12 | 781.52 | EMA200 below EMA400 |

### Cycle 150 — BUY (started 2026-03-10 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-10 11:15:00 | 788.95 | 782.17 | 781.24 | EMA200 above EMA400 |

### Cycle 151 — SELL (started 2026-03-11 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-11 11:15:00 | 771.40 | 780.18 | 781.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 12:15:00 | 765.85 | 777.31 | 779.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-13 12:15:00 | 761.35 | 761.24 | 765.42 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2026-03-13 14:45:00 | 754.95 | 759.62 | 763.96 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-17 13:15:00 | 752.90 | 749.67 | 752.89 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2026-03-17 13:15:00 | 752.90 | 749.67 | 752.89 | SL hit (close>ema400) qty=1.00 sl=752.89 alert=retest1 |

### Cycle 152 — BUY (started 2026-03-18 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 14:15:00 | 756.80 | 754.28 | 754.02 | EMA200 above EMA400 |

### Cycle 153 — SELL (started 2026-03-19 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 09:15:00 | 736.10 | 750.42 | 752.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-23 09:15:00 | 727.80 | 741.19 | 744.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 09:15:00 | 731.20 | 730.64 | 736.22 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-24 09:15:00 | 731.20 | 730.64 | 736.22 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 09:15:00 | 731.20 | 730.64 | 736.22 | EMA400 retest candle locked (from downside) |

### Cycle 154 — BUY (started 2026-03-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 09:15:00 | 751.25 | 740.10 | 738.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 12:15:00 | 756.25 | 746.75 | 742.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-27 09:15:00 | 746.85 | 749.28 | 745.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-27 09:15:00 | 746.85 | 749.28 | 745.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 746.85 | 749.28 | 745.17 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-03-27 12:15:00 | 752.25 | 749.69 | 746.08 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-03-27 14:15:00 | 741.30 | 747.66 | 746.04 | SL hit (close<static) qty=1.00 sl=743.00 alert=retest2 |

### Cycle 155 — SELL (started 2026-03-30 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-30 10:15:00 | 738.80 | 744.55 | 744.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-30 14:15:00 | 737.60 | 741.76 | 743.37 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 749.55 | 741.72 | 742.98 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 749.55 | 741.72 | 742.98 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 749.55 | 741.72 | 742.98 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 09:45:00 | 750.65 | 741.72 | 742.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 748.25 | 743.03 | 743.46 | EMA400 retest candle locked (from downside) |

### Cycle 156 — BUY (started 2026-04-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 11:15:00 | 751.45 | 744.71 | 744.19 | EMA200 above EMA400 |

### Cycle 157 — SELL (started 2026-04-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 09:15:00 | 734.55 | 742.95 | 743.74 | EMA200 below EMA400 |

### Cycle 158 — BUY (started 2026-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-02 13:15:00 | 763.15 | 746.93 | 745.16 | EMA200 above EMA400 |

### Cycle 159 — SELL (started 2026-04-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 09:15:00 | 747.90 | 750.21 | 750.29 | EMA200 below EMA400 |

### Cycle 160 — BUY (started 2026-04-09 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-09 11:15:00 | 752.60 | 750.48 | 750.39 | EMA200 above EMA400 |

### Cycle 161 — SELL (started 2026-04-09 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-09 13:15:00 | 747.80 | 750.07 | 750.22 | EMA200 below EMA400 |

### Cycle 162 — BUY (started 2026-04-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-10 09:15:00 | 756.35 | 750.66 | 750.39 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-10 11:15:00 | 758.70 | 753.11 | 751.60 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-13 09:15:00 | 756.15 | 757.65 | 754.79 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-13 09:15:00 | 756.15 | 757.65 | 754.79 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 756.15 | 757.65 | 754.79 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:45:00 | 761.80 | 758.18 | 755.29 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 13:00:00 | 762.20 | 759.38 | 756.36 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-16 09:15:00 | 748.60 | 754.58 | 755.27 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 163 — SELL (started 2026-04-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-16 09:15:00 | 748.60 | 754.58 | 755.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-16 12:15:00 | 741.95 | 750.49 | 753.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-17 09:15:00 | 754.20 | 748.66 | 751.10 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-17 09:15:00 | 754.20 | 748.66 | 751.10 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 09:15:00 | 754.20 | 748.66 | 751.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 10:00:00 | 754.20 | 748.66 | 751.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 10:15:00 | 756.80 | 750.29 | 751.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 11:00:00 | 756.80 | 750.29 | 751.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-17 11:15:00 | 756.85 | 751.60 | 752.10 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-17 11:45:00 | 758.00 | 751.60 | 752.10 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 164 — BUY (started 2026-04-17 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-17 13:15:00 | 754.60 | 752.72 | 752.55 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-17 14:15:00 | 756.60 | 753.50 | 752.92 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-21 11:15:00 | 758.65 | 760.14 | 757.93 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-21 11:15:00 | 758.65 | 760.14 | 757.93 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 11:15:00 | 758.65 | 760.14 | 757.93 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 11:45:00 | 757.95 | 760.14 | 757.93 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 12:15:00 | 754.80 | 759.07 | 757.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 13:00:00 | 754.80 | 759.07 | 757.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 13:15:00 | 755.40 | 758.34 | 757.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-21 14:00:00 | 755.40 | 758.34 | 757.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-21 14:15:00 | 762.00 | 759.07 | 757.86 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 15:15:00 | 762.80 | 759.07 | 757.86 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-22 09:30:00 | 764.60 | 760.46 | 758.74 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 15:15:00 | 776.60 | 781.67 | 782.09 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 165 — SELL (started 2026-04-29 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-29 15:15:00 | 776.60 | 781.67 | 782.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 09:15:00 | 767.65 | 778.87 | 780.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-30 13:15:00 | 775.85 | 775.51 | 778.23 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-30 13:15:00 | 775.85 | 775.51 | 778.23 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 13:15:00 | 775.85 | 775.51 | 778.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 13:45:00 | 777.10 | 775.51 | 778.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 14:15:00 | 775.05 | 775.42 | 777.94 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-30 14:30:00 | 777.60 | 775.42 | 777.94 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-30 15:15:00 | 776.20 | 775.58 | 777.78 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:15:00 | 783.00 | 775.58 | 777.78 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 783.45 | 777.15 | 778.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 09:45:00 | 782.85 | 777.15 | 778.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 10:15:00 | 783.40 | 778.40 | 778.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 10:45:00 | 783.95 | 778.40 | 778.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 166 — BUY (started 2026-05-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 11:15:00 | 783.00 | 779.32 | 779.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-05 09:15:00 | 790.70 | 783.92 | 781.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-05 13:15:00 | 781.85 | 786.43 | 783.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-05 13:15:00 | 781.85 | 786.43 | 783.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 13:15:00 | 781.85 | 786.43 | 783.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-05 13:45:00 | 766.50 | 786.43 | 783.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-05 14:15:00 | 808.05 | 790.76 | 786.02 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-05-05 15:15:00 | 810.00 | 790.76 | 786.02 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-13 09:15:00 | 591.65 | 2024-05-16 13:15:00 | 587.20 | STOP_HIT | 1.00 | -0.75% |
| BUY | retest2 | 2024-05-14 09:30:00 | 595.25 | 2024-05-16 13:15:00 | 587.20 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2024-05-27 09:15:00 | 606.80 | 2024-05-29 15:15:00 | 604.00 | STOP_HIT | 1.00 | -0.46% |
| BUY | retest2 | 2024-05-27 10:15:00 | 607.35 | 2024-05-30 09:15:00 | 595.65 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-05-27 10:45:00 | 608.00 | 2024-05-30 09:15:00 | 595.65 | STOP_HIT | 1.00 | -2.03% |
| BUY | retest2 | 2024-05-28 10:00:00 | 607.40 | 2024-05-30 09:15:00 | 595.65 | STOP_HIT | 1.00 | -1.93% |
| BUY | retest2 | 2024-05-29 13:15:00 | 610.40 | 2024-05-30 09:15:00 | 595.65 | STOP_HIT | 1.00 | -2.42% |
| SELL | retest2 | 2024-05-31 12:45:00 | 597.50 | 2024-06-04 09:15:00 | 606.50 | STOP_HIT | 1.00 | -1.51% |
| SELL | retest2 | 2024-05-31 14:15:00 | 599.05 | 2024-06-04 09:15:00 | 606.50 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-05-31 15:00:00 | 595.60 | 2024-06-04 09:15:00 | 606.50 | STOP_HIT | 1.00 | -1.83% |
| SELL | retest2 | 2024-06-03 11:00:00 | 599.05 | 2024-06-04 09:15:00 | 606.50 | STOP_HIT | 1.00 | -1.24% |
| SELL | retest2 | 2024-06-03 14:45:00 | 596.85 | 2024-06-04 09:15:00 | 606.50 | STOP_HIT | 1.00 | -1.62% |
| BUY | retest2 | 2024-06-07 12:15:00 | 650.05 | 2024-06-12 09:15:00 | 636.90 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2024-06-07 14:45:00 | 651.75 | 2024-06-12 09:15:00 | 636.90 | STOP_HIT | 1.00 | -2.28% |
| BUY | retest2 | 2024-06-10 10:30:00 | 651.45 | 2024-06-12 09:15:00 | 636.90 | STOP_HIT | 1.00 | -2.23% |
| BUY | retest2 | 2024-06-10 11:15:00 | 651.35 | 2024-06-12 09:15:00 | 636.90 | STOP_HIT | 1.00 | -2.22% |
| SELL | retest2 | 2024-06-26 13:30:00 | 615.60 | 2024-06-28 09:15:00 | 620.25 | STOP_HIT | 1.00 | -0.76% |
| SELL | retest2 | 2024-06-27 10:00:00 | 615.40 | 2024-06-28 09:15:00 | 620.25 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2024-06-28 10:15:00 | 615.45 | 2024-07-01 10:15:00 | 621.15 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2024-07-22 13:15:00 | 669.45 | 2024-07-22 15:15:00 | 665.40 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2024-08-08 10:15:00 | 647.55 | 2024-08-09 09:15:00 | 658.05 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2024-08-08 13:45:00 | 650.95 | 2024-08-09 09:15:00 | 658.05 | STOP_HIT | 1.00 | -1.09% |
| BUY | retest2 | 2024-08-26 09:30:00 | 689.80 | 2024-08-27 14:15:00 | 675.30 | STOP_HIT | 1.00 | -2.10% |
| BUY | retest2 | 2024-08-26 11:15:00 | 688.70 | 2024-08-27 14:15:00 | 675.30 | STOP_HIT | 1.00 | -1.95% |
| BUY | retest2 | 2024-08-26 15:00:00 | 691.00 | 2024-08-27 14:15:00 | 675.30 | STOP_HIT | 1.00 | -2.27% |
| SELL | retest2 | 2024-09-05 09:15:00 | 641.20 | 2024-09-06 09:15:00 | 661.45 | STOP_HIT | 1.00 | -3.16% |
| BUY | retest2 | 2024-09-13 13:30:00 | 680.20 | 2024-09-24 11:15:00 | 695.85 | STOP_HIT | 1.00 | 2.30% |
| SELL | retest2 | 2024-10-03 10:15:00 | 693.00 | 2024-10-03 11:15:00 | 700.10 | STOP_HIT | 1.00 | -1.02% |
| SELL | retest2 | 2024-10-14 10:15:00 | 675.80 | 2024-10-14 14:15:00 | 690.00 | STOP_HIT | 1.00 | -2.10% |
| SELL | retest2 | 2024-10-17 09:15:00 | 676.45 | 2024-10-24 11:15:00 | 642.63 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-17 09:15:00 | 676.45 | 2024-10-25 09:15:00 | 642.15 | STOP_HIT | 0.50 | 5.07% |
| SELL | retest2 | 2024-10-30 09:30:00 | 662.65 | 2024-10-30 10:15:00 | 651.75 | STOP_HIT | 1.00 | 1.64% |
| SELL | retest2 | 2024-11-19 14:45:00 | 592.15 | 2024-11-22 13:15:00 | 600.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-11-29 10:45:00 | 645.60 | 2024-12-03 13:15:00 | 638.90 | STOP_HIT | 1.00 | -1.04% |
| BUY | retest2 | 2024-12-02 15:15:00 | 645.45 | 2024-12-03 13:15:00 | 638.90 | STOP_HIT | 1.00 | -1.01% |
| SELL | retest2 | 2024-12-06 11:15:00 | 635.20 | 2024-12-09 09:15:00 | 603.44 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-06 11:15:00 | 635.20 | 2024-12-10 12:15:00 | 611.30 | STOP_HIT | 0.50 | 3.76% |
| BUY | retest2 | 2024-12-13 11:15:00 | 638.25 | 2024-12-17 15:15:00 | 630.00 | STOP_HIT | 1.00 | -1.29% |
| BUY | retest2 | 2024-12-16 10:30:00 | 638.50 | 2024-12-17 15:15:00 | 630.00 | STOP_HIT | 1.00 | -1.33% |
| BUY | retest2 | 2024-12-16 11:15:00 | 638.15 | 2024-12-17 15:15:00 | 630.00 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2024-12-16 12:15:00 | 637.80 | 2024-12-17 15:15:00 | 630.00 | STOP_HIT | 1.00 | -1.22% |
| BUY | retest2 | 2024-12-31 13:45:00 | 639.50 | 2025-01-06 15:15:00 | 647.00 | STOP_HIT | 1.00 | 1.17% |
| BUY | retest2 | 2025-01-01 10:30:00 | 638.75 | 2025-01-06 15:15:00 | 647.00 | STOP_HIT | 1.00 | 1.29% |
| BUY | retest2 | 2025-01-06 10:00:00 | 643.80 | 2025-01-06 15:15:00 | 647.00 | STOP_HIT | 1.00 | 0.50% |
| SELL | retest2 | 2025-01-16 10:30:00 | 652.70 | 2025-01-17 10:15:00 | 666.45 | STOP_HIT | 1.00 | -2.11% |
| BUY | retest2 | 2025-01-22 14:15:00 | 665.65 | 2025-01-27 09:15:00 | 661.40 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-01-22 14:45:00 | 665.40 | 2025-01-27 09:15:00 | 661.40 | STOP_HIT | 1.00 | -0.60% |
| BUY | retest2 | 2025-01-23 09:30:00 | 667.60 | 2025-01-27 09:15:00 | 661.40 | STOP_HIT | 1.00 | -0.93% |
| BUY | retest2 | 2025-01-24 10:00:00 | 666.85 | 2025-01-27 09:15:00 | 661.40 | STOP_HIT | 1.00 | -0.82% |
| BUY | retest2 | 2025-01-24 11:30:00 | 672.10 | 2025-01-27 09:15:00 | 661.40 | STOP_HIT | 1.00 | -1.59% |
| BUY | retest2 | 2025-01-24 12:15:00 | 669.95 | 2025-01-27 09:15:00 | 661.40 | STOP_HIT | 1.00 | -1.28% |
| BUY | retest2 | 2025-01-24 13:00:00 | 672.35 | 2025-01-27 09:15:00 | 661.40 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-01-31 12:30:00 | 682.15 | 2025-02-01 09:15:00 | 668.50 | STOP_HIT | 1.00 | -2.00% |
| BUY | retest2 | 2025-01-31 13:45:00 | 683.80 | 2025-02-01 09:15:00 | 668.50 | STOP_HIT | 1.00 | -2.24% |
| BUY | retest2 | 2025-02-01 12:15:00 | 685.75 | 2025-02-03 12:15:00 | 669.70 | STOP_HIT | 1.00 | -2.34% |
| BUY | retest2 | 2025-02-03 09:30:00 | 681.35 | 2025-02-03 12:15:00 | 669.70 | STOP_HIT | 1.00 | -1.71% |
| SELL | retest2 | 2025-02-06 12:15:00 | 666.40 | 2025-02-11 10:15:00 | 633.08 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-06 12:15:00 | 666.40 | 2025-02-12 12:15:00 | 633.55 | STOP_HIT | 0.50 | 4.93% |
| SELL | retest1 | 2025-03-05 10:00:00 | 585.70 | 2025-03-05 15:15:00 | 592.35 | STOP_HIT | 1.00 | -1.14% |
| BUY | retest2 | 2025-03-18 09:15:00 | 617.50 | 2025-03-25 14:15:00 | 624.60 | STOP_HIT | 1.00 | 1.15% |
| BUY | retest2 | 2025-04-03 15:00:00 | 661.00 | 2025-04-21 11:15:00 | 702.00 | STOP_HIT | 1.00 | 6.20% |
| BUY | retest2 | 2025-04-04 09:30:00 | 664.60 | 2025-04-21 11:15:00 | 702.00 | STOP_HIT | 1.00 | 5.63% |
| BUY | retest2 | 2025-04-07 11:30:00 | 661.50 | 2025-04-21 11:15:00 | 702.00 | STOP_HIT | 1.00 | 6.12% |
| BUY | retest2 | 2025-04-07 13:00:00 | 660.80 | 2025-04-21 11:15:00 | 702.00 | STOP_HIT | 1.00 | 6.23% |
| BUY | retest2 | 2025-04-08 09:30:00 | 665.00 | 2025-04-21 11:15:00 | 702.00 | STOP_HIT | 1.00 | 5.56% |
| BUY | retest2 | 2025-04-24 12:30:00 | 714.45 | 2025-04-25 15:15:00 | 710.05 | STOP_HIT | 1.00 | -0.62% |
| BUY | retest2 | 2025-04-25 10:00:00 | 712.05 | 2025-04-25 15:15:00 | 710.05 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-04-25 10:30:00 | 712.85 | 2025-04-25 15:15:00 | 710.05 | STOP_HIT | 1.00 | -0.39% |
| BUY | retest2 | 2025-04-25 11:00:00 | 713.55 | 2025-04-25 15:15:00 | 710.05 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-04-29 11:45:00 | 722.10 | 2025-04-29 14:15:00 | 710.75 | STOP_HIT | 1.00 | -1.57% |
| SELL | retest2 | 2025-05-15 14:00:00 | 723.45 | 2025-05-26 13:15:00 | 706.80 | STOP_HIT | 1.00 | 2.30% |
| SELL | retest2 | 2025-05-16 11:00:00 | 725.00 | 2025-05-26 13:15:00 | 706.80 | STOP_HIT | 1.00 | 2.51% |
| SELL | retest2 | 2025-05-16 14:15:00 | 724.95 | 2025-05-26 13:15:00 | 706.80 | STOP_HIT | 1.00 | 2.50% |
| BUY | retest2 | 2025-05-27 11:15:00 | 712.00 | 2025-06-02 09:15:00 | 707.60 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-06-06 12:15:00 | 699.40 | 2025-06-10 09:15:00 | 705.00 | STOP_HIT | 1.00 | -0.80% |
| BUY | retest2 | 2025-06-24 09:15:00 | 700.00 | 2025-07-01 13:15:00 | 717.70 | STOP_HIT | 1.00 | 2.53% |
| SELL | retest2 | 2025-07-02 13:15:00 | 709.40 | 2025-07-04 09:15:00 | 728.80 | STOP_HIT | 1.00 | -2.73% |
| SELL | retest2 | 2025-07-02 15:15:00 | 708.55 | 2025-07-04 09:15:00 | 728.80 | STOP_HIT | 1.00 | -2.86% |
| BUY | retest2 | 2025-07-16 13:45:00 | 733.80 | 2025-07-18 09:15:00 | 728.35 | STOP_HIT | 1.00 | -0.74% |
| BUY | retest2 | 2025-07-17 09:15:00 | 735.65 | 2025-07-18 09:15:00 | 728.35 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2025-07-22 14:30:00 | 716.25 | 2025-07-30 14:15:00 | 703.00 | STOP_HIT | 1.00 | 1.85% |
| SELL | retest2 | 2025-07-24 09:30:00 | 712.00 | 2025-07-30 14:15:00 | 703.00 | STOP_HIT | 1.00 | 1.26% |
| SELL | retest2 | 2025-08-12 15:00:00 | 703.95 | 2025-08-13 15:15:00 | 710.25 | STOP_HIT | 1.00 | -0.89% |
| SELL | retest2 | 2025-08-13 09:45:00 | 705.25 | 2025-08-13 15:15:00 | 710.25 | STOP_HIT | 1.00 | -0.71% |
| BUY | retest2 | 2025-08-19 11:15:00 | 718.10 | 2025-08-22 12:15:00 | 732.65 | STOP_HIT | 1.00 | 2.03% |
| SELL | retest2 | 2025-08-28 13:15:00 | 710.15 | 2025-08-29 14:15:00 | 725.90 | STOP_HIT | 1.00 | -2.22% |
| BUY | retest2 | 2025-09-04 09:15:00 | 744.50 | 2025-09-05 09:15:00 | 732.40 | STOP_HIT | 1.00 | -1.63% |
| BUY | retest2 | 2025-09-11 15:15:00 | 735.40 | 2025-09-15 09:15:00 | 728.50 | STOP_HIT | 1.00 | -0.94% |
| BUY | retest2 | 2025-09-12 10:45:00 | 735.10 | 2025-09-15 09:15:00 | 728.50 | STOP_HIT | 1.00 | -0.90% |
| BUY | retest2 | 2025-09-12 12:45:00 | 735.25 | 2025-09-15 09:15:00 | 728.50 | STOP_HIT | 1.00 | -0.92% |
| BUY | retest2 | 2025-09-12 13:45:00 | 735.20 | 2025-09-15 09:15:00 | 728.50 | STOP_HIT | 1.00 | -0.91% |
| SELL | retest2 | 2025-09-18 12:15:00 | 718.65 | 2025-09-19 15:15:00 | 722.00 | STOP_HIT | 1.00 | -0.47% |
| SELL | retest2 | 2025-09-18 14:15:00 | 719.05 | 2025-09-19 15:15:00 | 722.00 | STOP_HIT | 1.00 | -0.41% |
| SELL | retest2 | 2025-09-18 15:00:00 | 719.00 | 2025-09-19 15:15:00 | 722.00 | STOP_HIT | 1.00 | -0.42% |
| SELL | retest2 | 2025-09-19 09:45:00 | 717.80 | 2025-09-19 15:15:00 | 722.00 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-09-19 11:15:00 | 717.00 | 2025-09-19 15:15:00 | 722.00 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-09-19 11:45:00 | 716.95 | 2025-09-19 15:15:00 | 722.00 | STOP_HIT | 1.00 | -0.70% |
| SELL | retest2 | 2025-09-24 14:15:00 | 709.65 | 2025-10-01 15:15:00 | 704.00 | STOP_HIT | 1.00 | 0.80% |
| SELL | retest2 | 2025-09-25 12:30:00 | 707.75 | 2025-10-01 15:15:00 | 704.00 | STOP_HIT | 1.00 | 0.53% |
| BUY | retest2 | 2025-10-08 12:00:00 | 711.80 | 2025-10-08 15:15:00 | 710.00 | STOP_HIT | 1.00 | -0.25% |
| BUY | retest2 | 2025-10-08 13:00:00 | 712.00 | 2025-10-08 15:15:00 | 710.00 | STOP_HIT | 1.00 | -0.28% |
| BUY | retest2 | 2025-10-08 13:45:00 | 711.90 | 2025-10-08 15:15:00 | 710.00 | STOP_HIT | 1.00 | -0.27% |
| SELL | retest2 | 2025-10-24 13:30:00 | 722.20 | 2025-10-27 09:15:00 | 728.35 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2025-10-27 09:30:00 | 723.15 | 2025-10-30 15:15:00 | 723.80 | STOP_HIT | 1.00 | -0.09% |
| SELL | retest2 | 2025-10-27 11:30:00 | 720.70 | 2025-10-31 09:15:00 | 724.50 | STOP_HIT | 1.00 | -0.53% |
| SELL | retest2 | 2025-10-27 15:15:00 | 723.00 | 2025-10-31 09:15:00 | 724.50 | STOP_HIT | 1.00 | -0.21% |
| SELL | retest2 | 2025-10-28 09:15:00 | 719.95 | 2025-10-31 09:15:00 | 724.50 | STOP_HIT | 1.00 | -0.63% |
| SELL | retest2 | 2025-10-29 11:15:00 | 722.65 | 2025-10-31 09:15:00 | 724.50 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-10-29 12:00:00 | 721.90 | 2025-10-31 09:15:00 | 724.50 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-10-29 12:30:00 | 721.80 | 2025-10-31 09:15:00 | 724.50 | STOP_HIT | 1.00 | -0.37% |
| SELL | retest2 | 2025-10-30 09:45:00 | 717.35 | 2025-10-31 09:15:00 | 724.50 | STOP_HIT | 1.00 | -1.00% |
| BUY | retest2 | 2025-11-14 13:00:00 | 728.00 | 2025-11-20 11:15:00 | 743.50 | STOP_HIT | 1.00 | 2.13% |
| SELL | retest2 | 2025-11-25 13:00:00 | 730.70 | 2025-12-05 09:15:00 | 724.10 | STOP_HIT | 1.00 | 0.90% |
| SELL | retest2 | 2025-11-25 15:15:00 | 728.00 | 2025-12-05 09:15:00 | 724.10 | STOP_HIT | 1.00 | 0.54% |
| SELL | retest2 | 2025-11-27 09:15:00 | 730.05 | 2025-12-05 09:15:00 | 724.10 | STOP_HIT | 1.00 | 0.82% |
| BUY | retest2 | 2025-12-09 12:00:00 | 730.85 | 2025-12-10 14:15:00 | 724.90 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-12-09 15:00:00 | 730.70 | 2025-12-10 14:15:00 | 724.90 | STOP_HIT | 1.00 | -0.79% |
| SELL | retest2 | 2025-12-12 13:00:00 | 723.30 | 2025-12-15 09:15:00 | 734.55 | STOP_HIT | 1.00 | -1.56% |
| BUY | retest2 | 2025-12-17 10:30:00 | 739.80 | 2025-12-23 09:15:00 | 735.55 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-12-17 12:15:00 | 739.15 | 2025-12-23 09:15:00 | 735.55 | STOP_HIT | 1.00 | -0.49% |
| BUY | retest2 | 2025-12-18 09:15:00 | 739.45 | 2025-12-23 09:15:00 | 735.55 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2025-12-18 11:30:00 | 739.45 | 2025-12-23 09:15:00 | 735.55 | STOP_HIT | 1.00 | -0.53% |
| BUY | retest2 | 2026-01-05 13:15:00 | 770.20 | 2026-01-08 11:15:00 | 757.05 | STOP_HIT | 1.00 | -1.71% |
| BUY | retest2 | 2026-01-06 09:30:00 | 768.30 | 2026-01-08 11:15:00 | 757.05 | STOP_HIT | 1.00 | -1.46% |
| SELL | retest2 | 2026-01-13 11:00:00 | 753.40 | 2026-01-16 09:15:00 | 760.65 | STOP_HIT | 1.00 | -0.96% |
| SELL | retest2 | 2026-01-13 11:30:00 | 752.65 | 2026-01-16 09:15:00 | 760.65 | STOP_HIT | 1.00 | -1.06% |
| SELL | retest2 | 2026-01-13 13:00:00 | 753.35 | 2026-01-16 09:15:00 | 760.65 | STOP_HIT | 1.00 | -0.97% |
| SELL | retest2 | 2026-01-13 14:00:00 | 753.20 | 2026-01-16 09:15:00 | 760.65 | STOP_HIT | 1.00 | -0.99% |
| SELL | retest2 | 2026-01-14 13:30:00 | 750.55 | 2026-01-16 09:15:00 | 760.65 | STOP_HIT | 1.00 | -1.35% |
| BUY | retest2 | 2026-01-19 12:30:00 | 756.85 | 2026-01-20 13:15:00 | 753.60 | STOP_HIT | 1.00 | -0.43% |
| BUY | retest2 | 2026-01-20 09:30:00 | 758.85 | 2026-01-20 13:15:00 | 753.60 | STOP_HIT | 1.00 | -0.69% |
| BUY | retest2 | 2026-01-20 12:45:00 | 756.85 | 2026-01-20 13:15:00 | 753.60 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2026-01-23 10:15:00 | 745.90 | 2026-02-02 11:15:00 | 708.60 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-23 10:15:00 | 745.90 | 2026-02-02 11:15:00 | 722.40 | STOP_HIT | 0.50 | 3.15% |
| SELL | retest2 | 2026-01-23 12:00:00 | 744.00 | 2026-02-03 13:15:00 | 732.70 | STOP_HIT | 1.00 | 1.52% |
| BUY | retest2 | 2026-02-05 09:15:00 | 748.80 | 2026-02-13 14:15:00 | 759.30 | STOP_HIT | 1.00 | 1.40% |
| BUY | retest2 | 2026-02-20 10:15:00 | 787.10 | 2026-02-27 09:15:00 | 792.90 | STOP_HIT | 1.00 | 0.74% |
| SELL | retest1 | 2026-03-13 14:45:00 | 754.95 | 2026-03-17 13:15:00 | 752.90 | STOP_HIT | 1.00 | 0.27% |
| BUY | retest2 | 2026-03-27 12:15:00 | 752.25 | 2026-03-27 14:15:00 | 741.30 | STOP_HIT | 1.00 | -1.46% |
| BUY | retest2 | 2026-04-13 10:45:00 | 761.80 | 2026-04-16 09:15:00 | 748.60 | STOP_HIT | 1.00 | -1.73% |
| BUY | retest2 | 2026-04-13 13:00:00 | 762.20 | 2026-04-16 09:15:00 | 748.60 | STOP_HIT | 1.00 | -1.78% |
| BUY | retest2 | 2026-04-21 15:15:00 | 762.80 | 2026-04-29 15:15:00 | 776.60 | STOP_HIT | 1.00 | 1.81% |
| BUY | retest2 | 2026-04-22 09:30:00 | 764.60 | 2026-04-29 15:15:00 | 776.60 | STOP_HIT | 1.00 | 1.57% |
