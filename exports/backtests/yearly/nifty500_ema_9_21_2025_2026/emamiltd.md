# Emami Ltd. (EMAMILTD)

## Backtest Summary

- **Window:** 2025-03-12 09:15:00 → 2026-05-08 15:15:00 (1521 bars)
- **Last close:** 456.50
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 57 |
| ALERT1 | 31 |
| ALERT2 | 30 |
| ALERT2_SKIP | 15 |
| ALERT3 | 74 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 0 |
| ENTRY2 | 60 |
| PARTIAL | 12 |
| TARGET_HIT | 4 |
| STOP_HIT | 56 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 72 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 30 / 42
- **Target hits / Stop hits / Partials:** 4 / 56 / 12
- **Avg / median % per leg:** 1.44% / -0.25%
- **Sum % (uncompounded):** 103.44%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 13 | 5 | 38.5% | 4 | 9 | 0 | 2.77% | 36.1% |
| BUY @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| BUY @ 3rd Alert (retest2) | 13 | 5 | 38.5% | 4 | 9 | 0 | 2.77% | 36.1% |
| SELL (all) | 59 | 25 | 42.4% | 0 | 47 | 12 | 1.14% | 67.4% |
| SELL @ 2nd Alert (retest1) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| SELL @ 3rd Alert (retest2) | 59 | 25 | 42.4% | 0 | 47 | 12 | 1.14% | 67.4% |
| retest1 (combined) | 0 | 0 | 0.0% | 0 | 0 | 0 | 0.00% | 0.0% |
| retest2 (combined) | 72 | 30 | 41.7% | 4 | 56 | 12 | 1.44% | 103.4% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2025-05-12 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 09:15:00 | 634.95 | 628.06 | 627.41 | EMA200 above EMA400 |

### Cycle 2 — SELL (started 2025-05-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-13 12:15:00 | 621.60 | 630.23 | 630.24 | EMA200 below EMA400 |

### Cycle 3 — BUY (started 2025-05-14 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-14 12:15:00 | 630.90 | 629.49 | 629.35 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-14 14:15:00 | 639.40 | 632.58 | 630.85 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-15 09:15:00 | 628.70 | 632.99 | 631.41 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-15 09:15:00 | 628.70 | 632.99 | 631.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 09:15:00 | 628.70 | 632.99 | 631.41 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 10:00:00 | 628.70 | 632.99 | 631.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 10:15:00 | 627.40 | 631.87 | 631.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:00:00 | 627.40 | 631.87 | 631.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 11:15:00 | 630.00 | 631.50 | 630.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 11:30:00 | 628.40 | 631.50 | 630.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 13:15:00 | 632.20 | 631.61 | 631.09 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:15:00 | 624.80 | 631.61 | 631.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 14:15:00 | 630.00 | 631.29 | 630.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-15 14:30:00 | 630.20 | 631.29 | 630.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-15 15:15:00 | 630.05 | 631.04 | 630.91 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-05-16 09:15:00 | 634.20 | 631.04 | 630.91 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-16 13:15:00 | 625.00 | 630.78 | 630.99 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2025-05-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-16 13:15:00 | 625.00 | 630.78 | 630.99 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2025-05-16 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-16 14:15:00 | 636.85 | 632.00 | 631.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-19 09:15:00 | 639.55 | 633.80 | 632.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-19 11:15:00 | 634.25 | 634.73 | 633.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-19 11:15:00 | 634.25 | 634.73 | 633.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-19 11:15:00 | 634.25 | 634.73 | 633.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-19 12:00:00 | 634.25 | 634.73 | 633.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-20 09:15:00 | 621.05 | 634.48 | 634.04 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-20 09:30:00 | 620.00 | 634.48 | 634.04 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 6 — SELL (started 2025-05-20 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 10:15:00 | 620.65 | 631.72 | 632.83 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 11:15:00 | 618.70 | 629.11 | 631.54 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-22 14:15:00 | 604.55 | 601.85 | 608.14 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 15:00:00 | 604.55 | 601.85 | 608.14 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-23 09:15:00 | 604.20 | 602.21 | 607.21 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-23 15:00:00 | 594.45 | 599.59 | 603.96 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-26 12:30:00 | 594.65 | 597.97 | 601.47 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-27 10:45:00 | 594.50 | 594.10 | 597.83 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-04 12:15:00 | 596.00 | 584.99 | 583.50 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 7 — BUY (started 2025-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 12:15:00 | 596.00 | 584.99 | 583.50 | EMA200 above EMA400 |

### Cycle 8 — SELL (started 2025-06-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-05 13:15:00 | 579.60 | 584.19 | 584.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-05 15:15:00 | 578.00 | 582.12 | 583.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-06 10:15:00 | 581.80 | 581.75 | 583.06 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-06 14:15:00 | 582.65 | 581.96 | 582.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 14:15:00 | 582.65 | 581.96 | 582.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-06 15:00:00 | 582.65 | 581.96 | 582.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-06 15:15:00 | 584.10 | 582.39 | 582.88 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-09 09:15:00 | 582.60 | 582.39 | 582.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-09 09:15:00 | 580.05 | 581.92 | 582.62 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-09 14:00:00 | 578.50 | 580.49 | 581.66 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-10 09:45:00 | 579.25 | 579.10 | 580.66 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-10 14:15:00 | 583.10 | 581.20 | 581.15 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 9 — BUY (started 2025-06-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-10 14:15:00 | 583.10 | 581.20 | 581.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-10 15:15:00 | 583.50 | 581.66 | 581.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-12 09:15:00 | 588.15 | 588.39 | 585.84 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-12 10:00:00 | 588.15 | 588.39 | 585.84 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 10:15:00 | 584.50 | 587.61 | 585.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 11:00:00 | 584.50 | 587.61 | 585.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 11:15:00 | 583.15 | 586.72 | 585.48 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-12 12:00:00 | 583.15 | 586.72 | 585.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-12 12:15:00 | 584.15 | 586.21 | 585.36 | EMA400 retest candle locked (from upside) |

### Cycle 10 — SELL (started 2025-06-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 14:15:00 | 574.35 | 583.01 | 584.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-13 09:15:00 | 570.15 | 579.62 | 582.24 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-13 14:15:00 | 573.45 | 573.35 | 577.63 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-13 14:45:00 | 571.55 | 573.35 | 577.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-18 09:15:00 | 602.15 | 579.04 | 579.47 | EMA400 retest candle locked (from downside) |

### Cycle 11 — BUY (started 2025-09-18 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-18 10:15:00 | 609.15 | 585.06 | 582.17 | EMA200 above EMA400 |

### Cycle 12 — SELL (started 2025-09-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-19 15:15:00 | 583.00 | 589.76 | 589.85 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-23 14:15:00 | 575.45 | 580.60 | 583.82 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 12:15:00 | 579.20 | 578.19 | 581.16 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 12:15:00 | 579.20 | 578.19 | 581.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 12:15:00 | 579.20 | 578.19 | 581.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:00:00 | 579.20 | 578.19 | 581.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 09:15:00 | 577.75 | 577.80 | 580.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-09-25 11:00:00 | 574.50 | 577.14 | 579.50 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-09-29 12:15:00 | 545.77 | 557.33 | 563.67 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-10-01 13:15:00 | 544.25 | 541.43 | 547.33 | SL hit (close>ema200) qty=0.50 sl=541.43 alert=retest2 |

### Cycle 13 — BUY (started 2025-10-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-06 10:15:00 | 550.70 | 547.80 | 547.65 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-06 11:15:00 | 553.35 | 548.91 | 548.17 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-07 10:15:00 | 552.95 | 553.46 | 551.20 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-07 10:30:00 | 553.00 | 553.46 | 551.20 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 11:15:00 | 550.45 | 552.86 | 551.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-07 11:45:00 | 550.40 | 552.86 | 551.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-07 12:15:00 | 550.30 | 552.34 | 551.06 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:15:00 | 552.15 | 551.75 | 550.90 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 14:45:00 | 551.65 | 551.72 | 550.96 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-07 15:15:00 | 552.30 | 551.72 | 550.96 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-08 09:15:00 | 543.70 | 550.21 | 550.41 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 14 — SELL (started 2025-10-08 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-08 09:15:00 | 543.70 | 550.21 | 550.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-08 10:15:00 | 541.00 | 548.37 | 549.56 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-09 10:15:00 | 543.60 | 543.44 | 545.91 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-10-09 11:00:00 | 543.60 | 543.44 | 545.91 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 542.10 | 541.96 | 543.92 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 13:30:00 | 538.80 | 541.03 | 542.29 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-13 14:45:00 | 537.75 | 540.12 | 541.76 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 12:15:00 | 538.05 | 536.52 | 537.52 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-15 13:15:00 | 541.10 | 538.14 | 538.13 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 15 — BUY (started 2025-10-15 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-15 13:15:00 | 541.10 | 538.14 | 538.13 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2025-10-16 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-16 10:15:00 | 535.10 | 537.90 | 538.09 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2025-10-16 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-16 13:15:00 | 540.00 | 538.14 | 538.13 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-17 09:15:00 | 541.65 | 539.19 | 538.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-20 09:15:00 | 541.10 | 545.11 | 542.60 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-20 09:15:00 | 541.10 | 545.11 | 542.60 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 541.10 | 545.11 | 542.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 10:00:00 | 541.10 | 545.11 | 542.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 10:15:00 | 543.30 | 544.75 | 542.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:00:00 | 543.30 | 544.75 | 542.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 11:15:00 | 544.40 | 544.68 | 542.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 11:45:00 | 544.20 | 544.68 | 542.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 14:15:00 | 541.25 | 544.09 | 543.02 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-20 15:00:00 | 541.25 | 544.09 | 543.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 15:15:00 | 541.45 | 543.56 | 542.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-21 13:45:00 | 545.70 | 543.79 | 543.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-23 10:00:00 | 544.75 | 544.03 | 543.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-10-24 11:30:00 | 542.50 | 544.93 | 544.67 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-10-24 13:15:00 | 541.30 | 544.10 | 544.33 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 18 — SELL (started 2025-10-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 13:15:00 | 541.30 | 544.10 | 544.33 | EMA200 below EMA400 |

### Cycle 19 — BUY (started 2025-10-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-27 09:15:00 | 545.75 | 544.60 | 544.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-28 10:15:00 | 548.10 | 545.46 | 545.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-29 09:15:00 | 543.40 | 546.81 | 546.15 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-29 09:15:00 | 543.40 | 546.81 | 546.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 09:15:00 | 543.40 | 546.81 | 546.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 10:00:00 | 543.40 | 546.81 | 546.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-29 10:15:00 | 542.85 | 546.02 | 545.85 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-29 11:00:00 | 542.85 | 546.02 | 545.85 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 20 — SELL (started 2025-10-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-29 11:15:00 | 543.80 | 545.57 | 545.66 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-30 15:15:00 | 542.20 | 543.46 | 544.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-03 13:15:00 | 534.15 | 532.87 | 536.65 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-03 14:00:00 | 534.15 | 532.87 | 536.65 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 09:15:00 | 537.75 | 533.98 | 536.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:30:00 | 540.25 | 533.98 | 536.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 10:15:00 | 538.55 | 534.89 | 536.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 10:45:00 | 539.50 | 534.89 | 536.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 11:15:00 | 538.20 | 535.55 | 536.60 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-04 12:00:00 | 538.20 | 535.55 | 536.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-04 12:15:00 | 533.00 | 535.04 | 536.27 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-06 10:15:00 | 530.40 | 535.42 | 536.14 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-11-10 13:15:00 | 503.88 | 515.70 | 520.08 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-11-10 13:15:00 | 518.75 | 515.70 | 520.08 | SL hit (close>static) qty=0.50 sl=515.70 alert=retest2 |

### Cycle 21 — BUY (started 2025-11-11 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-11 13:15:00 | 524.35 | 521.99 | 521.82 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-11 15:15:00 | 525.90 | 523.28 | 522.46 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-13 09:15:00 | 521.60 | 524.90 | 523.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-13 09:15:00 | 521.60 | 524.90 | 523.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 09:15:00 | 521.60 | 524.90 | 523.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-13 10:00:00 | 521.60 | 524.90 | 523.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-13 10:15:00 | 527.60 | 525.44 | 524.30 | EMA400 retest candle locked (from upside) |

### Cycle 22 — SELL (started 2025-11-13 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 15:15:00 | 521.60 | 523.53 | 523.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 515.30 | 521.89 | 522.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-14 14:15:00 | 520.65 | 518.54 | 520.56 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-14 14:15:00 | 520.65 | 518.54 | 520.56 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 14:15:00 | 520.65 | 518.54 | 520.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-14 14:45:00 | 520.45 | 518.54 | 520.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-14 15:15:00 | 520.00 | 518.83 | 520.51 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 10:45:00 | 519.25 | 519.09 | 520.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-17 13:00:00 | 519.45 | 518.95 | 520.06 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 09:15:00 | 518.00 | 520.63 | 520.65 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-18 15:00:00 | 518.95 | 518.64 | 519.27 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-18 15:15:00 | 518.05 | 518.52 | 519.16 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 09:15:00 | 516.50 | 518.52 | 519.16 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-19 13:15:00 | 513.40 | 518.51 | 518.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-11-20 09:15:00 | 520.50 | 515.93 | 517.28 | SL hit (close>static) qty=1.00 sl=519.90 alert=retest2 |

### Cycle 23 — BUY (started 2025-11-20 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-20 15:15:00 | 520.00 | 517.95 | 517.82 | EMA200 above EMA400 |

### Cycle 24 — SELL (started 2025-11-21 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 10:15:00 | 515.45 | 517.56 | 517.67 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-21 11:15:00 | 514.75 | 516.99 | 517.41 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-25 11:15:00 | 509.00 | 506.76 | 510.15 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-11-25 12:00:00 | 509.00 | 506.76 | 510.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 12:15:00 | 511.40 | 507.69 | 510.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 12:45:00 | 511.30 | 507.69 | 510.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 13:15:00 | 515.00 | 509.15 | 510.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 14:00:00 | 515.00 | 509.15 | 510.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-25 14:15:00 | 515.50 | 510.42 | 511.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-25 15:00:00 | 515.50 | 510.42 | 511.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 25 — BUY (started 2025-11-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-26 10:15:00 | 513.40 | 511.72 | 511.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-27 09:15:00 | 527.15 | 515.77 | 513.69 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-01 09:15:00 | 524.55 | 528.01 | 524.75 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-01 09:15:00 | 524.55 | 528.01 | 524.75 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-01 09:15:00 | 524.55 | 528.01 | 524.75 | EMA400 retest candle locked (from upside) |

### Cycle 26 — SELL (started 2025-12-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-02 14:15:00 | 521.95 | 524.56 | 524.80 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 09:15:00 | 519.45 | 523.06 | 524.05 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-03 10:15:00 | 524.45 | 523.34 | 524.09 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-03 10:15:00 | 524.45 | 523.34 | 524.09 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 10:15:00 | 524.45 | 523.34 | 524.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-03 11:00:00 | 524.45 | 523.34 | 524.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-03 11:15:00 | 521.95 | 523.06 | 523.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-03 15:15:00 | 519.30 | 522.23 | 523.27 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-05 13:15:00 | 527.35 | 520.58 | 520.43 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 27 — BUY (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-05 13:15:00 | 527.35 | 520.58 | 520.43 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2025-12-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-08 10:15:00 | 517.75 | 520.53 | 520.58 | EMA200 below EMA400 |

### Cycle 29 — BUY (started 2025-12-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-08 11:15:00 | 521.05 | 520.63 | 520.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-08 12:15:00 | 523.60 | 521.22 | 520.89 | Break + close above crossover candle high |

### Cycle 30 — SELL (started 2025-12-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-09 09:15:00 | 513.25 | 520.01 | 520.50 | EMA200 below EMA400 |

### Cycle 31 — BUY (started 2025-12-10 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-10 09:15:00 | 526.60 | 521.65 | 521.09 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 11:15:00 | 528.10 | 525.26 | 523.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 10:15:00 | 528.75 | 529.52 | 526.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-12 10:45:00 | 528.35 | 529.52 | 526.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-16 15:15:00 | 539.60 | 542.57 | 539.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-17 09:15:00 | 539.15 | 542.57 | 539.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-17 09:15:00 | 538.35 | 541.73 | 539.82 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-17 10:45:00 | 541.60 | 541.77 | 540.01 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-17 15:15:00 | 535.15 | 538.80 | 539.11 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 32 — SELL (started 2025-12-17 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-17 15:15:00 | 535.15 | 538.80 | 539.11 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-18 09:15:00 | 523.55 | 535.75 | 537.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-22 09:15:00 | 524.35 | 522.51 | 526.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-12-22 10:00:00 | 524.35 | 522.51 | 526.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-23 09:15:00 | 524.05 | 523.73 | 525.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-24 13:15:00 | 517.85 | 520.00 | 522.09 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-29 14:15:00 | 536.00 | 518.47 | 517.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 33 — BUY (started 2025-12-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 14:15:00 | 536.00 | 518.47 | 517.88 | EMA200 above EMA400 |

### Cycle 34 — SELL (started 2026-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-02 09:15:00 | 523.65 | 525.29 | 525.43 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-02 12:15:00 | 521.70 | 524.22 | 524.88 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 15:15:00 | 524.00 | 523.31 | 524.21 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-05 09:15:00 | 524.40 | 523.31 | 524.21 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 525.55 | 523.75 | 524.33 | EMA400 retest candle locked (from downside) |

### Cycle 35 — BUY (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 11:15:00 | 526.30 | 524.92 | 524.80 | EMA200 above EMA400 |

### Cycle 36 — SELL (started 2026-01-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-06 10:15:00 | 521.50 | 524.31 | 524.65 | EMA200 below EMA400 |

### Cycle 37 — BUY (started 2026-01-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-07 11:15:00 | 528.35 | 524.57 | 524.33 | EMA200 above EMA400 |

### Cycle 38 — SELL (started 2026-01-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-08 10:15:00 | 520.30 | 524.31 | 524.56 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 13:15:00 | 517.15 | 522.00 | 523.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-12 11:15:00 | 510.05 | 509.80 | 514.52 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-12 12:00:00 | 510.05 | 509.80 | 514.52 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 13:15:00 | 516.35 | 511.51 | 514.50 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-12 14:00:00 | 516.35 | 511.51 | 514.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-12 14:15:00 | 512.95 | 511.80 | 514.36 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 09:15:00 | 509.25 | 513.24 | 514.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 13:30:00 | 510.00 | 509.51 | 511.98 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-13 15:00:00 | 510.45 | 509.70 | 511.84 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 09:15:00 | 509.95 | 509.95 | 511.76 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 10:15:00 | 509.20 | 509.45 | 511.19 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-14 10:30:00 | 509.50 | 509.45 | 511.19 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 12:15:00 | 502.85 | 508.24 | 510.35 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 13:45:00 | 501.50 | 506.96 | 509.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 15:00:00 | 501.70 | 505.91 | 508.86 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-19 09:15:00 | 500.85 | 505.63 | 507.06 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 09:15:00 | 499.40 | 502.49 | 504.36 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-20 09:15:00 | 497.60 | 501.51 | 503.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-20 14:00:00 | 495.00 | 499.77 | 502.18 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 13:15:00 | 483.79 | 490.11 | 495.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 13:15:00 | 484.50 | 490.11 | 495.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 13:15:00 | 484.93 | 490.11 | 495.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-01-21 13:15:00 | 484.45 | 490.11 | 495.48 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-01-22 09:15:00 | 493.55 | 488.80 | 493.39 | SL hit (close>ema200) qty=0.50 sl=488.80 alert=retest2 |

### Cycle 39 — BUY (started 2026-01-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-22 13:15:00 | 503.00 | 495.93 | 495.67 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 09:15:00 | 510.45 | 500.40 | 497.93 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 504.75 | 505.59 | 501.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-23 15:00:00 | 504.75 | 505.59 | 501.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 502.60 | 504.50 | 502.07 | EMA400 retest candle locked (from upside) |

### Cycle 40 — SELL (started 2026-01-27 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 15:15:00 | 498.20 | 501.47 | 501.49 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-28 09:15:00 | 490.00 | 499.17 | 500.45 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-30 09:15:00 | 485.25 | 483.64 | 488.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-30 09:15:00 | 485.25 | 483.64 | 488.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 09:15:00 | 485.25 | 483.64 | 488.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 09:30:00 | 488.55 | 483.64 | 488.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 14:15:00 | 483.45 | 482.76 | 485.86 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-30 15:00:00 | 483.45 | 482.76 | 485.86 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-30 15:15:00 | 484.00 | 483.01 | 485.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-01 09:15:00 | 480.65 | 483.01 | 485.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-01 09:15:00 | 480.50 | 482.51 | 485.22 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 12:15:00 | 478.50 | 482.15 | 484.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-01 13:00:00 | 477.30 | 481.18 | 483.92 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-03 12:45:00 | 479.25 | 479.98 | 480.25 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-03 14:15:00 | 483.30 | 480.94 | 480.65 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 41 — BUY (started 2026-02-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 14:15:00 | 483.30 | 480.94 | 480.65 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2026-02-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-04 10:15:00 | 477.30 | 479.97 | 480.30 | EMA200 below EMA400 |

### Cycle 43 — BUY (started 2026-02-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-04 14:15:00 | 488.95 | 481.92 | 481.06 | EMA200 above EMA400 |

### Cycle 44 — SELL (started 2026-02-10 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-10 14:15:00 | 497.65 | 501.32 | 501.60 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-12 10:15:00 | 494.80 | 499.04 | 500.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-13 11:15:00 | 496.20 | 495.55 | 497.45 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-13 12:00:00 | 496.20 | 495.55 | 497.45 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 492.85 | 490.48 | 492.48 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-17 10:00:00 | 492.85 | 490.48 | 492.48 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 10:15:00 | 491.20 | 490.63 | 492.37 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 12:45:00 | 489.30 | 490.19 | 491.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 14:00:00 | 489.35 | 490.02 | 491.63 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-17 15:15:00 | 489.50 | 490.02 | 491.48 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:15:00 | 464.83 | 473.00 | 475.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:15:00 | 464.88 | 473.00 | 475.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-27 10:15:00 | 465.02 | 473.00 | 475.28 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-02 14:15:00 | 460.50 | 459.79 | 465.20 | SL hit (close>ema200) qty=0.50 sl=459.79 alert=retest2 |

### Cycle 45 — BUY (started 2026-03-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-05 13:15:00 | 468.50 | 461.71 | 461.29 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-05 14:15:00 | 473.70 | 464.11 | 462.42 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-06 10:15:00 | 464.00 | 464.97 | 463.32 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-06 10:15:00 | 464.00 | 464.97 | 463.32 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 10:15:00 | 464.00 | 464.97 | 463.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 10:45:00 | 464.00 | 464.97 | 463.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 12:15:00 | 468.35 | 465.61 | 463.89 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 12:30:00 | 468.30 | 465.61 | 463.89 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 14:15:00 | 459.40 | 464.49 | 463.69 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-06 15:00:00 | 459.40 | 464.49 | 463.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-06 15:15:00 | 461.00 | 463.79 | 463.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-09 09:15:00 | 444.90 | 463.79 | 463.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 46 — SELL (started 2026-03-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-09 09:15:00 | 441.05 | 459.24 | 461.41 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-11 11:15:00 | 440.00 | 442.94 | 447.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 13:15:00 | 417.20 | 414.42 | 421.67 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 14:00:00 | 417.20 | 414.42 | 421.67 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 14:15:00 | 424.45 | 416.43 | 421.92 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-16 15:00:00 | 424.45 | 416.43 | 421.92 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-16 15:15:00 | 421.10 | 417.36 | 421.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 09:15:00 | 415.35 | 417.36 | 421.84 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 13:45:00 | 421.00 | 420.47 | 421.79 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-17 14:15:00 | 419.95 | 420.47 | 421.79 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-03-18 09:45:00 | 420.10 | 419.27 | 420.87 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 13:15:00 | 418.85 | 418.04 | 419.65 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 13:45:00 | 418.10 | 418.04 | 419.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 399.95 | 405.02 | 408.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 398.95 | 405.02 | 408.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-03-23 09:15:00 | 399.10 | 405.02 | 408.99 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-03-23 11:15:00 | 406.80 | 404.68 | 408.10 | SL hit (close>ema200) qty=0.50 sl=404.68 alert=retest2 |

### Cycle 47 — BUY (started 2026-03-25 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-25 10:15:00 | 415.70 | 407.47 | 406.58 | EMA200 above EMA400 |

### Cycle 48 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 399.40 | 406.64 | 407.21 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 395.50 | 401.35 | 404.26 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 397.35 | 395.64 | 398.71 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 397.35 | 395.64 | 398.71 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 397.35 | 395.64 | 398.71 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-01 10:15:00 | 395.90 | 395.64 | 398.71 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 09:15:00 | 392.30 | 394.77 | 395.41 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-06 11:00:00 | 394.35 | 394.24 | 395.03 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-06 12:15:00 | 398.60 | 395.48 | 395.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 49 — BUY (started 2026-04-06 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 12:15:00 | 398.60 | 395.48 | 395.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-06 14:15:00 | 401.05 | 397.12 | 396.26 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-09 09:15:00 | 416.00 | 416.31 | 410.97 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-04-09 09:45:00 | 415.85 | 416.31 | 410.97 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-13 09:15:00 | 423.30 | 423.18 | 419.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 10:15:00 | 424.35 | 423.18 | 419.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-13 15:00:00 | 424.60 | 425.28 | 422.26 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-15 09:15:00 | 425.55 | 424.72 | 422.28 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-16 10:15:00 | 424.95 | 426.14 | 424.92 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-16 10:15:00 | 427.50 | 426.42 | 425.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-17 09:15:00 | 434.40 | 425.18 | 424.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2026-04-22 11:15:00 | 466.79 | 457.04 | 450.29 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 50 — SELL (started 2026-04-24 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-24 13:15:00 | 455.70 | 459.79 | 460.12 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2026-04-27 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-27 11:15:00 | 462.35 | 460.43 | 460.25 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2026-04-27 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-27 13:15:00 | 459.40 | 460.09 | 460.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-27 15:15:00 | 458.90 | 459.89 | 460.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-28 09:15:00 | 460.00 | 459.91 | 460.02 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-28 09:15:00 | 460.00 | 459.91 | 460.02 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 09:15:00 | 460.00 | 459.91 | 460.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-28 10:15:00 | 461.25 | 459.91 | 460.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-28 10:15:00 | 458.25 | 459.58 | 459.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 11:15:00 | 456.10 | 459.58 | 459.86 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-28 12:00:00 | 455.90 | 458.84 | 459.50 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-04-29 10:00:00 | 457.20 | 456.53 | 457.88 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-29 11:15:00 | 465.45 | 459.12 | 458.88 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 53 — BUY (started 2026-04-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 11:15:00 | 465.45 | 459.12 | 458.88 | EMA200 above EMA400 |

### Cycle 54 — SELL (started 2026-04-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 09:15:00 | 450.20 | 457.76 | 458.46 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 10:15:00 | 445.75 | 455.36 | 457.31 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 09:15:00 | 459.15 | 450.68 | 453.34 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 09:15:00 | 459.15 | 450.68 | 453.34 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 09:15:00 | 459.15 | 450.68 | 453.34 | EMA400 retest candle locked (from downside) |

### Cycle 55 — BUY (started 2026-05-04 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 13:15:00 | 455.85 | 455.06 | 454.95 | EMA200 above EMA400 |

### Cycle 56 — SELL (started 2026-05-05 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-05-05 09:15:00 | 449.95 | 454.35 | 454.69 | EMA200 below EMA400 |

### Cycle 57 — BUY (started 2026-05-07 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-07 11:15:00 | 455.95 | 452.29 | 452.17 | EMA200 above EMA400 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2025-05-16 09:15:00 | 634.20 | 2025-05-16 13:15:00 | 625.00 | STOP_HIT | 1.00 | -1.45% |
| SELL | retest2 | 2025-05-23 15:00:00 | 594.45 | 2025-06-04 12:15:00 | 596.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2025-05-26 12:30:00 | 594.65 | 2025-06-04 12:15:00 | 596.00 | STOP_HIT | 1.00 | -0.23% |
| SELL | retest2 | 2025-05-27 10:45:00 | 594.50 | 2025-06-04 12:15:00 | 596.00 | STOP_HIT | 1.00 | -0.25% |
| SELL | retest2 | 2025-06-09 14:00:00 | 578.50 | 2025-06-10 14:15:00 | 583.10 | STOP_HIT | 1.00 | -0.80% |
| SELL | retest2 | 2025-06-10 09:45:00 | 579.25 | 2025-06-10 14:15:00 | 583.10 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-09-25 11:00:00 | 574.50 | 2025-09-29 12:15:00 | 545.77 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-09-25 11:00:00 | 574.50 | 2025-10-01 13:15:00 | 544.25 | STOP_HIT | 0.50 | 5.27% |
| BUY | retest2 | 2025-10-07 14:15:00 | 552.15 | 2025-10-08 09:15:00 | 543.70 | STOP_HIT | 1.00 | -1.53% |
| BUY | retest2 | 2025-10-07 14:45:00 | 551.65 | 2025-10-08 09:15:00 | 543.70 | STOP_HIT | 1.00 | -1.44% |
| BUY | retest2 | 2025-10-07 15:15:00 | 552.30 | 2025-10-08 09:15:00 | 543.70 | STOP_HIT | 1.00 | -1.56% |
| SELL | retest2 | 2025-10-13 13:30:00 | 538.80 | 2025-10-15 13:15:00 | 541.10 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2025-10-13 14:45:00 | 537.75 | 2025-10-15 13:15:00 | 541.10 | STOP_HIT | 1.00 | -0.62% |
| SELL | retest2 | 2025-10-15 12:15:00 | 538.05 | 2025-10-15 13:15:00 | 541.10 | STOP_HIT | 1.00 | -0.57% |
| BUY | retest2 | 2025-10-21 13:45:00 | 545.70 | 2025-10-24 13:15:00 | 541.30 | STOP_HIT | 1.00 | -0.81% |
| BUY | retest2 | 2025-10-23 10:00:00 | 544.75 | 2025-10-24 13:15:00 | 541.30 | STOP_HIT | 1.00 | -0.63% |
| BUY | retest2 | 2025-10-24 11:30:00 | 542.50 | 2025-10-24 13:15:00 | 541.30 | STOP_HIT | 1.00 | -0.22% |
| SELL | retest2 | 2025-11-06 10:15:00 | 530.40 | 2025-11-10 13:15:00 | 503.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-11-06 10:15:00 | 530.40 | 2025-11-10 13:15:00 | 518.75 | STOP_HIT | 0.50 | 2.20% |
| SELL | retest2 | 2025-11-17 10:45:00 | 519.25 | 2025-11-20 09:15:00 | 520.50 | STOP_HIT | 1.00 | -0.24% |
| SELL | retest2 | 2025-11-17 13:00:00 | 519.45 | 2025-11-20 09:15:00 | 520.50 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-11-18 09:15:00 | 518.00 | 2025-11-20 15:15:00 | 520.00 | STOP_HIT | 1.00 | -0.39% |
| SELL | retest2 | 2025-11-18 15:00:00 | 518.95 | 2025-11-20 15:15:00 | 520.00 | STOP_HIT | 1.00 | -0.20% |
| SELL | retest2 | 2025-11-19 09:15:00 | 516.50 | 2025-11-20 15:15:00 | 520.00 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-11-19 13:15:00 | 513.40 | 2025-11-20 15:15:00 | 520.00 | STOP_HIT | 1.00 | -1.29% |
| SELL | retest2 | 2025-11-20 14:15:00 | 515.95 | 2025-11-20 15:15:00 | 520.00 | STOP_HIT | 1.00 | -0.78% |
| SELL | retest2 | 2025-12-03 15:15:00 | 519.30 | 2025-12-05 13:15:00 | 527.35 | STOP_HIT | 1.00 | -1.55% |
| BUY | retest2 | 2025-12-17 10:45:00 | 541.60 | 2025-12-17 15:15:00 | 535.15 | STOP_HIT | 1.00 | -1.19% |
| SELL | retest2 | 2025-12-24 13:15:00 | 517.85 | 2025-12-29 14:15:00 | 536.00 | STOP_HIT | 1.00 | -3.50% |
| SELL | retest2 | 2026-01-13 09:15:00 | 509.25 | 2026-01-21 13:15:00 | 483.79 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 13:30:00 | 510.00 | 2026-01-21 13:15:00 | 484.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 15:00:00 | 510.45 | 2026-01-21 13:15:00 | 484.93 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-14 09:15:00 | 509.95 | 2026-01-21 13:15:00 | 484.45 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-01-13 09:15:00 | 509.25 | 2026-01-22 09:15:00 | 493.55 | STOP_HIT | 0.50 | 3.08% |
| SELL | retest2 | 2026-01-13 13:30:00 | 510.00 | 2026-01-22 09:15:00 | 493.55 | STOP_HIT | 0.50 | 3.23% |
| SELL | retest2 | 2026-01-13 15:00:00 | 510.45 | 2026-01-22 09:15:00 | 493.55 | STOP_HIT | 0.50 | 3.31% |
| SELL | retest2 | 2026-01-14 09:15:00 | 509.95 | 2026-01-22 09:15:00 | 493.55 | STOP_HIT | 0.50 | 3.22% |
| SELL | retest2 | 2026-01-14 13:45:00 | 501.50 | 2026-01-22 13:15:00 | 503.00 | STOP_HIT | 1.00 | -0.30% |
| SELL | retest2 | 2026-01-14 15:00:00 | 501.70 | 2026-01-22 13:15:00 | 503.00 | STOP_HIT | 1.00 | -0.26% |
| SELL | retest2 | 2026-01-19 09:15:00 | 500.85 | 2026-01-22 13:15:00 | 503.00 | STOP_HIT | 1.00 | -0.43% |
| SELL | retest2 | 2026-01-20 09:15:00 | 499.40 | 2026-01-22 13:15:00 | 503.00 | STOP_HIT | 1.00 | -0.72% |
| SELL | retest2 | 2026-01-20 14:00:00 | 495.00 | 2026-01-22 13:15:00 | 503.00 | STOP_HIT | 1.00 | -1.62% |
| SELL | retest2 | 2026-02-01 12:15:00 | 478.50 | 2026-02-03 14:15:00 | 483.30 | STOP_HIT | 1.00 | -1.00% |
| SELL | retest2 | 2026-02-01 13:00:00 | 477.30 | 2026-02-03 14:15:00 | 483.30 | STOP_HIT | 1.00 | -1.26% |
| SELL | retest2 | 2026-02-03 12:45:00 | 479.25 | 2026-02-03 14:15:00 | 483.30 | STOP_HIT | 1.00 | -0.85% |
| SELL | retest2 | 2026-02-17 12:45:00 | 489.30 | 2026-02-27 10:15:00 | 464.83 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 14:00:00 | 489.35 | 2026-02-27 10:15:00 | 464.88 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 15:15:00 | 489.50 | 2026-02-27 10:15:00 | 465.02 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-17 12:45:00 | 489.30 | 2026-03-02 14:15:00 | 460.50 | STOP_HIT | 0.50 | 5.89% |
| SELL | retest2 | 2026-02-17 14:00:00 | 489.35 | 2026-03-02 14:15:00 | 460.50 | STOP_HIT | 0.50 | 5.90% |
| SELL | retest2 | 2026-02-17 15:15:00 | 489.50 | 2026-03-02 14:15:00 | 460.50 | STOP_HIT | 0.50 | 5.92% |
| SELL | retest2 | 2026-03-17 09:15:00 | 415.35 | 2026-03-23 09:15:00 | 399.95 | PARTIAL | 0.50 | 3.71% |
| SELL | retest2 | 2026-03-17 13:45:00 | 421.00 | 2026-03-23 09:15:00 | 398.95 | PARTIAL | 0.50 | 5.24% |
| SELL | retest2 | 2026-03-17 14:15:00 | 419.95 | 2026-03-23 09:15:00 | 399.10 | PARTIAL | 0.50 | 4.97% |
| SELL | retest2 | 2026-03-17 09:15:00 | 415.35 | 2026-03-23 11:15:00 | 406.80 | STOP_HIT | 0.50 | 2.06% |
| SELL | retest2 | 2026-03-17 13:45:00 | 421.00 | 2026-03-23 11:15:00 | 406.80 | STOP_HIT | 0.50 | 3.37% |
| SELL | retest2 | 2026-03-17 14:15:00 | 419.95 | 2026-03-23 11:15:00 | 406.80 | STOP_HIT | 0.50 | 3.13% |
| SELL | retest2 | 2026-03-18 09:45:00 | 420.10 | 2026-03-25 09:15:00 | 416.25 | STOP_HIT | 1.00 | 0.92% |
| SELL | retest2 | 2026-03-24 09:30:00 | 401.40 | 2026-03-25 09:15:00 | 416.25 | STOP_HIT | 1.00 | -3.70% |
| SELL | retest2 | 2026-03-24 13:45:00 | 401.90 | 2026-03-25 09:15:00 | 416.25 | STOP_HIT | 1.00 | -3.57% |
| SELL | retest2 | 2026-03-24 14:15:00 | 402.20 | 2026-03-25 10:15:00 | 415.70 | STOP_HIT | 1.00 | -3.36% |
| SELL | retest2 | 2026-04-01 10:15:00 | 395.90 | 2026-04-06 12:15:00 | 398.60 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2026-04-06 09:15:00 | 392.30 | 2026-04-06 12:15:00 | 398.60 | STOP_HIT | 1.00 | -1.61% |
| SELL | retest2 | 2026-04-06 11:00:00 | 394.35 | 2026-04-06 12:15:00 | 398.60 | STOP_HIT | 1.00 | -1.08% |
| BUY | retest2 | 2026-04-13 10:15:00 | 424.35 | 2026-04-22 11:15:00 | 466.79 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-13 15:00:00 | 424.60 | 2026-04-22 11:15:00 | 467.06 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-15 09:15:00 | 425.55 | 2026-04-22 11:15:00 | 468.11 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-16 10:15:00 | 424.95 | 2026-04-22 11:15:00 | 467.45 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2026-04-17 09:15:00 | 434.40 | 2026-04-24 13:15:00 | 455.70 | STOP_HIT | 1.00 | 4.90% |
| SELL | retest2 | 2026-04-28 11:15:00 | 456.10 | 2026-04-29 11:15:00 | 465.45 | STOP_HIT | 1.00 | -2.05% |
| SELL | retest2 | 2026-04-28 12:00:00 | 455.90 | 2026-04-29 11:15:00 | 465.45 | STOP_HIT | 1.00 | -2.09% |
| SELL | retest2 | 2026-04-29 10:00:00 | 457.20 | 2026-04-29 11:15:00 | 465.45 | STOP_HIT | 1.00 | -1.80% |
