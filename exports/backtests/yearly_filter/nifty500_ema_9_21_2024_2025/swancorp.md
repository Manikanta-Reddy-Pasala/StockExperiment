# Swan Corp Ltd. (SWANCORP)

## Backtest Summary

- **Window:** 2024-03-13 10:15:00 → 2026-05-08 15:15:00 (3709 bars)
- **Last close:** 353.15
- **Strategy:** EMA 200/400 1H crossover (Strategy-1 spec)
- **Target:** entry × (1 ± 10%)
- **Partial:** 50% qty @ 5% (ENTRY1) / BUY-ENTRY2 15% / SELL-ENTRY2 5%, trail SL → EMA200
- **SL:** ENTRY1 = EMA400 close-based; ENTRY2 = retest2 candle low/high
- **Re-entry cap:** 1 initial + 3 re-entries at each of 2nd / 3rd alert

## Signal Counts

| Signal | Count |
|--------|-------|
| CROSSOVER | 149 |
| ALERT1 | 103 |
| ALERT2 | 100 |
| ALERT2_SKIP | 56 |
| ALERT3 | 241 |
| PENDING | 0 |
| PENDING_CANCEL | 0 |
| ENTRY1 | 5 |
| ENTRY2 | 112 |
| PARTIAL | 28 |
| TARGET_HIT | 19 |
| STOP_HIT | 98 |
| EXIT | 0 |

## P&L (combined)

- **Closed legs:** 145 (incl. partial bookings)
- **Trades open at end:** 0
- **Winners / losers:** 67 / 78
- **Target hits / Stop hits / Partials:** 19 / 98 / 28
- **Avg / median % per leg:** 1.59% / -0.36%
- **Sum % (uncompounded):** 230.82%

## Direction × Alert breakdown

| Bucket | Legs | Win | Win% | Tgt | SL | Prt | Avg % | Sum % |
|--------|------|-----|------|-----|----|-----|-------|-------|
| BUY (all) | 49 | 19 | 38.8% | 7 | 39 | 3 | 0.77% | 37.7% |
| BUY @ 2nd Alert (retest1) | 7 | 6 | 85.7% | 0 | 4 | 3 | 3.18% | 22.3% |
| BUY @ 3rd Alert (retest2) | 42 | 13 | 31.0% | 7 | 35 | 0 | 0.37% | 15.4% |
| SELL (all) | 96 | 48 | 50.0% | 12 | 59 | 25 | 2.01% | 193.1% |
| SELL @ 2nd Alert (retest1) | 1 | 0 | 0.0% | 0 | 1 | 0 | -2.18% | -2.2% |
| SELL @ 3rd Alert (retest2) | 95 | 48 | 50.5% | 12 | 58 | 25 | 2.06% | 195.3% |
| retest1 (combined) | 8 | 6 | 75.0% | 0 | 5 | 3 | 2.51% | 20.1% |
| retest2 (combined) | 137 | 61 | 44.5% | 19 | 93 | 25 | 1.54% | 210.7% |

## Strategy Cycles

Each cycle begins at a CROSSOVER (trend flip) and walks through the
configured stages: Trend ID → First Alert → Second Alert (Retest 1)
→ First Entry → Third Alert (Retest 2) → Second Entry → Exit.

### Cycle 1 — BUY (started 2024-05-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-05-14 11:15:00 | 562.10 | 542.51 | 541.24 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-05-15 09:15:00 | 582.25 | 559.71 | 550.98 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-05-16 13:15:00 | 574.10 | 578.02 | 570.31 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-05-16 13:45:00 | 574.15 | 578.02 | 570.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-05-21 15:15:00 | 598.50 | 599.20 | 594.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:15:00 | 600.70 | 599.20 | 594.73 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-22 09:45:00 | 602.25 | 599.65 | 595.34 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-05-24 09:15:00 | 605.35 | 603.67 | 602.32 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-05-29 11:15:00 | 614.00 | 621.17 | 622.00 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 2 — SELL (started 2024-05-29 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-05-29 11:15:00 | 614.00 | 621.17 | 622.00 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-05-30 14:15:00 | 608.85 | 615.65 | 618.32 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-06-03 09:15:00 | 589.30 | 584.03 | 595.67 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-03 09:15:00 | 589.30 | 584.03 | 595.67 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-03 09:15:00 | 589.30 | 584.03 | 595.67 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 11:00:00 | 578.00 | 582.82 | 594.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-03 13:00:00 | 578.45 | 578.69 | 590.03 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-06-04 09:15:00 | 567.35 | 585.68 | 590.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-05 09:15:00 | 549.10 | 564.76 | 575.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-06-05 09:15:00 | 549.53 | 564.76 | 575.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-06-05 14:15:00 | 550.50 | 550.14 | 562.98 | SL hit (close>ema200) qty=0.50 sl=550.14 alert=retest2 |

### Cycle 3 — BUY (started 2024-06-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-06 15:15:00 | 574.50 | 567.33 | 566.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-07 10:15:00 | 577.00 | 570.33 | 568.09 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-07 11:15:00 | 565.95 | 569.45 | 567.90 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-06-07 11:15:00 | 565.95 | 569.45 | 567.90 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 11:15:00 | 565.95 | 569.45 | 567.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-07 11:45:00 | 568.05 | 569.45 | 567.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-07 12:15:00 | 571.40 | 569.84 | 568.22 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-07 14:15:00 | 576.40 | 569.55 | 568.23 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-06-13 11:15:00 | 597.00 | 600.72 | 601.08 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 4 — SELL (started 2024-06-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-13 11:15:00 | 597.00 | 600.72 | 601.08 | EMA200 below EMA400 |

### Cycle 5 — BUY (started 2024-06-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-06-13 12:15:00 | 604.95 | 601.56 | 601.43 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-06-13 14:15:00 | 614.20 | 604.32 | 602.72 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-06-19 09:15:00 | 646.75 | 651.36 | 637.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-06-19 09:30:00 | 652.00 | 651.36 | 637.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 14:15:00 | 651.00 | 654.10 | 649.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-20 15:00:00 | 651.00 | 654.10 | 649.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-20 15:15:00 | 649.70 | 653.22 | 649.44 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 09:15:00 | 652.25 | 653.22 | 649.44 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 10:15:00 | 652.00 | 652.58 | 649.49 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 11:00:00 | 652.85 | 652.63 | 649.80 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-06-21 11:30:00 | 652.05 | 652.70 | 650.09 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 12:15:00 | 652.10 | 652.58 | 650.27 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 12:45:00 | 651.80 | 652.58 | 650.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-06-21 14:15:00 | 655.00 | 652.93 | 650.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-06-21 15:00:00 | 655.00 | 652.93 | 650.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2024-06-24 09:15:00 | 628.00 | 649.01 | 649.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 6 — SELL (started 2024-06-24 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-06-24 09:15:00 | 628.00 | 649.01 | 649.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-06-26 09:15:00 | 621.20 | 628.67 | 634.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-01 10:15:00 | 593.00 | 591.30 | 599.99 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-07-01 10:45:00 | 593.30 | 591.30 | 599.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 13:15:00 | 609.30 | 596.83 | 600.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 13:45:00 | 608.00 | 596.83 | 600.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-01 14:15:00 | 613.00 | 600.06 | 601.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-01 15:00:00 | 613.00 | 600.06 | 601.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 7 — BUY (started 2024-07-01 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-01 15:15:00 | 616.20 | 603.29 | 602.98 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-02 09:15:00 | 624.85 | 607.60 | 604.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-03 10:15:00 | 618.20 | 618.42 | 613.54 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-03 13:15:00 | 615.45 | 617.84 | 614.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 13:15:00 | 615.45 | 617.84 | 614.52 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 13:45:00 | 615.10 | 617.84 | 614.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-03 14:15:00 | 618.45 | 617.96 | 614.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-03 14:30:00 | 615.00 | 617.96 | 614.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-10 10:15:00 | 715.70 | 717.65 | 699.13 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-10 10:30:00 | 703.65 | 717.65 | 699.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 11:15:00 | 723.80 | 727.95 | 721.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 11:30:00 | 721.70 | 727.95 | 721.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 12:15:00 | 721.90 | 726.74 | 721.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-12 12:30:00 | 714.55 | 726.74 | 721.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-12 13:15:00 | 721.90 | 725.77 | 721.67 | EMA400 retest candle locked (from upside) |

### Cycle 8 — SELL (started 2024-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-15 10:15:00 | 710.00 | 718.18 | 719.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-15 12:15:00 | 707.05 | 714.63 | 717.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-16 09:15:00 | 719.20 | 711.56 | 714.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-16 09:15:00 | 719.20 | 711.56 | 714.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 09:15:00 | 719.20 | 711.56 | 714.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 10:00:00 | 719.20 | 711.56 | 714.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-16 10:15:00 | 722.00 | 713.65 | 715.20 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-16 10:45:00 | 722.25 | 713.65 | 715.20 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 9 — BUY (started 2024-07-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-16 12:15:00 | 724.05 | 717.18 | 716.62 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-18 09:15:00 | 726.70 | 720.07 | 718.23 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-18 10:15:00 | 715.40 | 719.14 | 717.97 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-18 10:15:00 | 715.40 | 719.14 | 717.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 10:15:00 | 715.40 | 719.14 | 717.97 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:00:00 | 715.40 | 719.14 | 717.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 11:15:00 | 720.10 | 719.33 | 718.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 11:30:00 | 716.25 | 719.33 | 718.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 12:15:00 | 723.00 | 720.06 | 718.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 12:30:00 | 719.40 | 720.06 | 718.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 13:15:00 | 719.20 | 719.89 | 718.66 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 14:00:00 | 719.20 | 719.89 | 718.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-18 14:15:00 | 714.00 | 718.71 | 718.23 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-07-18 15:00:00 | 714.00 | 718.71 | 718.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 10 — SELL (started 2024-07-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-18 15:15:00 | 709.90 | 716.95 | 717.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-07-19 09:15:00 | 692.25 | 712.01 | 715.18 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-07-24 09:15:00 | 702.55 | 682.32 | 688.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-07-24 09:15:00 | 702.55 | 682.32 | 688.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 09:15:00 | 702.55 | 682.32 | 688.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:00:00 | 702.55 | 682.32 | 688.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-24 10:15:00 | 702.55 | 686.36 | 689.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-07-24 10:30:00 | 702.55 | 686.36 | 689.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 11 — BUY (started 2024-07-24 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-24 12:15:00 | 702.55 | 692.19 | 691.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-25 10:15:00 | 721.75 | 702.80 | 697.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-07-26 15:15:00 | 730.00 | 730.20 | 720.47 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2024-07-29 09:15:00 | 739.05 | 730.20 | 720.47 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-07-29 14:15:00 | 732.55 | 732.42 | 726.12 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-29 15:15:00 | 736.00 | 732.42 | 726.12 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-07-30 10:15:00 | 734.00 | 732.53 | 727.30 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-07-30 13:15:00 | 723.70 | 729.56 | 727.48 | SL hit (close<ema400) qty=1.00 sl=727.48 alert=retest1 |

### Cycle 12 — SELL (started 2024-07-31 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-07-31 09:15:00 | 721.75 | 726.26 | 726.32 | EMA200 below EMA400 |

### Cycle 13 — BUY (started 2024-07-31 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-07-31 10:15:00 | 744.45 | 729.90 | 727.97 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-07-31 11:15:00 | 759.30 | 735.78 | 730.82 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-01 09:15:00 | 748.55 | 748.77 | 740.25 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-01 09:45:00 | 746.70 | 748.77 | 740.25 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-01 10:15:00 | 746.00 | 748.21 | 740.78 | EMA400 retest candle locked (from upside) |

### Cycle 14 — SELL (started 2024-08-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-01 14:15:00 | 727.20 | 736.95 | 737.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-02 11:15:00 | 722.30 | 730.17 | 733.61 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-06 09:15:00 | 703.70 | 695.48 | 707.87 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-08-06 10:00:00 | 703.70 | 695.48 | 707.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-06 10:15:00 | 687.05 | 693.79 | 705.98 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-06 11:30:00 | 683.60 | 691.49 | 703.83 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-08 09:45:00 | 677.35 | 677.41 | 683.88 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-08 11:15:00 | 707.05 | 683.10 | 685.33 | SL hit (close>static) qty=1.00 sl=706.80 alert=retest2 |

### Cycle 15 — BUY (started 2024-08-09 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-09 09:15:00 | 699.35 | 687.71 | 687.02 | EMA200 above EMA400 |

### Cycle 16 — SELL (started 2024-08-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 09:15:00 | 685.00 | 688.84 | 688.85 | EMA200 below EMA400 |

### Cycle 17 — BUY (started 2024-08-13 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-13 10:15:00 | 694.80 | 690.03 | 689.39 | EMA200 above EMA400 |

### Cycle 18 — SELL (started 2024-08-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-13 11:15:00 | 683.00 | 688.63 | 688.81 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-13 12:15:00 | 678.90 | 686.68 | 687.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-13 13:15:00 | 690.15 | 687.38 | 688.11 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-13 13:15:00 | 690.15 | 687.38 | 688.11 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 13:15:00 | 690.15 | 687.38 | 688.11 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-08-13 14:00:00 | 690.15 | 687.38 | 688.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-13 14:15:00 | 682.10 | 686.32 | 687.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 09:15:00 | 674.10 | 685.46 | 687.06 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-14 14:00:00 | 676.00 | 678.25 | 682.26 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 11:00:00 | 680.30 | 677.96 | 680.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 13:15:00 | 681.50 | 679.83 | 681.17 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-16 13:15:00 | 682.50 | 680.37 | 681.29 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-16 14:30:00 | 680.50 | 680.17 | 681.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-19 10:15:00 | 679.25 | 680.83 | 681.28 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2024-08-19 13:15:00 | 682.40 | 681.49 | 681.47 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 19 — BUY (started 2024-08-19 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 13:15:00 | 682.40 | 681.49 | 681.47 | EMA200 above EMA400 |

### Cycle 20 — SELL (started 2024-08-19 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-19 14:15:00 | 680.55 | 681.30 | 681.39 | EMA200 below EMA400 |

### Cycle 21 — BUY (started 2024-08-19 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-19 15:15:00 | 682.40 | 681.52 | 681.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-20 09:15:00 | 683.00 | 681.82 | 681.62 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-20 11:15:00 | 681.80 | 682.24 | 681.87 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-20 11:15:00 | 681.80 | 682.24 | 681.87 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 11:15:00 | 681.80 | 682.24 | 681.87 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 12:00:00 | 681.80 | 682.24 | 681.87 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-20 12:15:00 | 679.45 | 681.68 | 681.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-20 13:00:00 | 679.45 | 681.68 | 681.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 22 — SELL (started 2024-08-20 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-20 13:15:00 | 681.00 | 681.54 | 681.59 | EMA200 below EMA400 |

### Cycle 23 — BUY (started 2024-08-21 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-08-21 09:15:00 | 689.35 | 682.45 | 681.94 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-08-21 10:15:00 | 706.85 | 687.33 | 684.21 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-08-22 09:15:00 | 696.30 | 697.32 | 691.87 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-08-22 09:45:00 | 696.80 | 697.32 | 691.87 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 10:15:00 | 689.45 | 695.74 | 691.65 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 11:00:00 | 689.45 | 695.74 | 691.65 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-22 11:15:00 | 689.70 | 694.54 | 691.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-22 12:15:00 | 688.55 | 694.54 | 691.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-23 09:15:00 | 702.40 | 693.55 | 691.79 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-08-23 09:45:00 | 694.90 | 693.55 | 691.79 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-26 09:15:00 | 705.45 | 700.26 | 696.56 | EMA400 retest candle locked (from upside) |

### Cycle 24 — SELL (started 2024-08-27 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-08-27 09:15:00 | 688.50 | 695.86 | 696.23 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-08-27 14:15:00 | 683.95 | 690.90 | 693.52 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-08-28 09:15:00 | 691.90 | 690.58 | 692.89 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-08-28 09:15:00 | 691.90 | 690.58 | 692.89 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-08-28 09:15:00 | 691.90 | 690.58 | 692.89 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-08-28 10:30:00 | 687.00 | 689.67 | 692.26 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-09-04 09:15:00 | 652.65 | 662.77 | 667.25 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-09-04 09:15:00 | 663.00 | 662.77 | 667.25 | SL hit (close>static) qty=0.50 sl=662.77 alert=retest2 |

### Cycle 25 — BUY (started 2024-09-20 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-20 14:15:00 | 596.35 | 591.55 | 591.49 | EMA200 above EMA400 |

### Cycle 26 — SELL (started 2024-09-24 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-24 11:15:00 | 589.65 | 593.08 | 593.34 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-24 13:15:00 | 584.75 | 590.62 | 592.13 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-26 09:15:00 | 588.55 | 576.74 | 581.52 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-26 09:15:00 | 588.55 | 576.74 | 581.52 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 09:15:00 | 588.55 | 576.74 | 581.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:00:00 | 588.55 | 576.74 | 581.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-26 10:15:00 | 609.30 | 583.25 | 584.05 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-09-26 10:30:00 | 622.10 | 583.25 | 584.05 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 27 — BUY (started 2024-09-26 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-09-26 11:15:00 | 608.55 | 588.31 | 586.28 | EMA200 above EMA400 |

### Cycle 28 — SELL (started 2024-09-27 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-09-27 14:15:00 | 581.30 | 588.30 | 588.54 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-09-30 09:15:00 | 578.00 | 585.75 | 587.30 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-09-30 15:15:00 | 577.40 | 575.83 | 580.57 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-09-30 15:15:00 | 577.40 | 575.83 | 580.57 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-09-30 15:15:00 | 577.40 | 575.83 | 580.57 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-01 09:15:00 | 580.00 | 575.83 | 580.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-01 09:15:00 | 581.00 | 576.87 | 580.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 11:00:00 | 577.20 | 576.93 | 580.30 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-01 13:00:00 | 574.00 | 576.65 | 579.61 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 548.34 | 556.32 | 562.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-07 09:15:00 | 545.30 | 556.32 | 562.80 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2024-10-07 13:15:00 | 519.48 | 536.95 | 550.57 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 29 — BUY (started 2024-10-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-09 10:15:00 | 572.95 | 548.45 | 545.55 | EMA200 above EMA400 |

### Cycle 30 — SELL (started 2024-10-15 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-15 11:15:00 | 557.05 | 559.51 | 559.73 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-15 12:15:00 | 555.95 | 558.80 | 559.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-15 14:15:00 | 561.50 | 559.11 | 559.41 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-15 14:15:00 | 561.50 | 559.11 | 559.41 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 14:15:00 | 561.50 | 559.11 | 559.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-10-15 14:45:00 | 561.80 | 559.11 | 559.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-15 15:15:00 | 558.05 | 558.90 | 559.29 | EMA400 retest candle locked (from downside) |

### Cycle 31 — BUY (started 2024-10-16 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-16 09:15:00 | 566.20 | 560.36 | 559.92 | EMA200 above EMA400 |

### Cycle 32 — SELL (started 2024-10-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-10-16 12:15:00 | 558.50 | 559.63 | 559.68 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-10-16 15:15:00 | 552.00 | 557.25 | 558.51 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-10-18 13:15:00 | 541.25 | 540.08 | 545.72 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-10-18 13:45:00 | 542.00 | 540.08 | 545.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-21 09:15:00 | 525.60 | 536.60 | 542.70 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 12:45:00 | 521.85 | 530.56 | 538.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-10-21 14:45:00 | 520.55 | 527.75 | 535.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 12:15:00 | 495.76 | 511.42 | 523.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-10-22 13:15:00 | 494.52 | 508.19 | 521.21 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-10-24 11:15:00 | 489.05 | 487.44 | 497.63 | SL hit (close>ema200) qty=0.50 sl=487.44 alert=retest2 |

### Cycle 33 — BUY (started 2024-10-29 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-10-29 14:15:00 | 493.00 | 481.96 | 480.95 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-10-30 09:15:00 | 502.65 | 487.57 | 483.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-10-31 12:15:00 | 497.20 | 498.32 | 493.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-10-31 12:15:00 | 497.20 | 498.32 | 493.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-10-31 12:15:00 | 497.20 | 498.32 | 493.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-10-31 13:00:00 | 497.20 | 498.32 | 493.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-04 09:15:00 | 482.75 | 496.01 | 494.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-04 10:00:00 | 482.75 | 496.01 | 494.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 34 — SELL (started 2024-11-04 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-04 10:15:00 | 481.50 | 493.11 | 493.67 | EMA200 below EMA400 |

### Cycle 35 — BUY (started 2024-11-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-05 13:15:00 | 497.30 | 491.82 | 491.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-06 09:15:00 | 499.45 | 494.63 | 492.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-06 13:15:00 | 494.85 | 496.10 | 494.35 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-06 13:15:00 | 494.85 | 496.10 | 494.35 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 13:15:00 | 494.85 | 496.10 | 494.35 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 14:00:00 | 494.85 | 496.10 | 494.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 14:15:00 | 495.15 | 495.91 | 494.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-06 14:45:00 | 494.00 | 495.91 | 494.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-06 15:15:00 | 497.00 | 496.13 | 494.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-07 09:15:00 | 499.60 | 496.13 | 494.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Target hit | 2024-11-07 11:15:00 | 549.56 | 520.42 | 507.03 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 36 — SELL (started 2024-11-13 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-13 12:15:00 | 516.65 | 528.56 | 529.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-13 14:15:00 | 503.45 | 521.86 | 526.49 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-11-14 09:15:00 | 530.00 | 521.48 | 525.39 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-11-14 09:15:00 | 530.00 | 521.48 | 525.39 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-14 09:15:00 | 530.00 | 521.48 | 525.39 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 13:45:00 | 508.55 | 518.34 | 522.58 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 14:30:00 | 510.00 | 517.02 | 521.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-14 15:15:00 | 509.80 | 517.02 | 521.59 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:15:00 | 483.12 | 514.66 | 519.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:15:00 | 484.50 | 514.66 | 519.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2024-11-18 09:15:00 | 484.31 | 514.66 | 519.65 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-11-18 10:15:00 | 508.65 | 514.66 | 519.65 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-18 10:15:00 | 521.35 | 516.00 | 519.81 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2024-11-18 10:15:00 | 521.35 | 516.00 | 519.81 | SL hit (close>ema200) qty=0.50 sl=516.00 alert=retest2 |

### Cycle 37 — BUY (started 2024-11-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-18 12:15:00 | 550.60 | 525.70 | 523.73 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-19 11:15:00 | 557.40 | 542.64 | 534.15 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-19 14:15:00 | 538.80 | 543.32 | 536.72 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-19 15:00:00 | 538.80 | 543.32 | 536.72 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-19 15:15:00 | 535.00 | 541.66 | 536.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-21 09:15:00 | 540.90 | 541.66 | 536.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-21 09:15:00 | 538.60 | 541.05 | 536.75 | EMA400 retest candle locked (from upside) |

### Cycle 38 — SELL (started 2024-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-11-21 14:15:00 | 527.55 | 534.74 | 535.06 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-11-22 10:15:00 | 523.40 | 530.46 | 532.87 | Break + close below crossover candle low |

### Cycle 39 — BUY (started 2024-11-22 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-11-22 13:15:00 | 569.25 | 535.92 | 534.52 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2024-11-22 14:15:00 | 584.40 | 545.62 | 539.05 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2024-11-27 13:15:00 | 616.75 | 617.07 | 604.06 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2024-11-27 14:00:00 | 616.75 | 617.07 | 604.06 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-28 10:15:00 | 617.60 | 614.91 | 606.96 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-28 10:45:00 | 607.35 | 614.91 | 606.96 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 09:15:00 | 612.50 | 614.19 | 610.06 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2024-11-29 10:15:00 | 610.90 | 614.19 | 610.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-11-29 10:15:00 | 610.00 | 613.35 | 610.05 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 11:15:00 | 617.40 | 613.35 | 610.05 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-11-29 13:00:00 | 615.60 | 614.46 | 611.17 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2024-12-02 09:30:00 | 621.50 | 613.94 | 611.81 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2024-12-04 11:15:00 | 679.14 | 645.77 | 634.77 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 40 — SELL (started 2024-12-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-19 10:15:00 | 769.85 | 779.26 | 779.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-19 14:15:00 | 759.10 | 770.52 | 775.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-20 10:15:00 | 778.45 | 769.01 | 773.03 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2024-12-20 10:15:00 | 778.45 | 769.01 | 773.03 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 10:15:00 | 778.45 | 769.01 | 773.03 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 11:00:00 | 778.45 | 769.01 | 773.03 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 11:15:00 | 774.00 | 770.01 | 773.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2024-12-20 11:45:00 | 777.80 | 770.01 | 773.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-20 12:15:00 | 759.00 | 767.80 | 771.83 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2024-12-20 13:30:00 | 758.35 | 763.37 | 769.45 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2024-12-26 09:15:00 | 720.43 | 730.15 | 740.01 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2024-12-27 12:15:00 | 732.00 | 721.26 | 727.33 | SL hit (close>ema200) qty=0.50 sl=721.26 alert=retest2 |

### Cycle 41 — BUY (started 2024-12-30 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2024-12-30 09:15:00 | 737.15 | 730.83 | 730.57 | EMA200 above EMA400 |

### Cycle 42 — SELL (started 2024-12-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2024-12-30 13:15:00 | 720.95 | 728.80 | 729.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2024-12-30 14:15:00 | 714.90 | 726.02 | 728.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2024-12-31 10:15:00 | 720.75 | 720.48 | 724.90 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2024-12-31 10:45:00 | 719.50 | 720.48 | 724.90 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2024-12-31 12:15:00 | 718.55 | 719.39 | 723.59 | EMA400 retest candle locked (from downside) |

### Cycle 43 — BUY (started 2025-01-02 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-02 09:15:00 | 739.35 | 726.36 | 725.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-03 10:15:00 | 762.65 | 743.50 | 735.32 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-03 15:15:00 | 750.00 | 751.33 | 743.15 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-01-06 09:15:00 | 740.05 | 751.33 | 743.15 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 09:15:00 | 738.50 | 748.76 | 742.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 10:00:00 | 738.50 | 748.76 | 742.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-06 10:15:00 | 719.00 | 742.81 | 740.57 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-06 11:00:00 | 719.00 | 742.81 | 740.57 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 44 — SELL (started 2025-01-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-06 11:15:00 | 720.45 | 738.34 | 738.74 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-06 13:15:00 | 703.90 | 727.71 | 733.60 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-07 09:15:00 | 717.50 | 716.80 | 726.35 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-07 09:30:00 | 720.95 | 716.80 | 726.35 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 13:15:00 | 733.00 | 721.23 | 725.46 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-01-07 14:00:00 | 733.00 | 721.23 | 725.46 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-07 14:15:00 | 725.55 | 722.10 | 725.46 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-08 09:15:00 | 710.85 | 723.08 | 725.60 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-01-09 13:15:00 | 675.31 | 690.76 | 702.54 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-01-13 09:15:00 | 639.76 | 657.97 | 675.00 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 45 — BUY (started 2025-01-15 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-01-15 15:15:00 | 652.25 | 645.70 | 644.86 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-01-16 09:15:00 | 665.90 | 649.74 | 646.77 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-01-20 09:15:00 | 646.50 | 673.99 | 668.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-01-20 09:15:00 | 646.50 | 673.99 | 668.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 09:15:00 | 646.50 | 673.99 | 668.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-01-20 10:00:00 | 646.50 | 673.99 | 668.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-20 10:15:00 | 653.50 | 669.90 | 667.11 | EMA400 retest candle locked (from upside) |

### Cycle 46 — SELL (started 2025-01-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-01-20 12:15:00 | 646.45 | 662.21 | 663.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-01-20 14:15:00 | 639.95 | 656.01 | 660.69 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-01-23 09:15:00 | 614.20 | 614.00 | 626.49 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-01-23 10:00:00 | 614.20 | 614.00 | 626.49 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-01-24 12:15:00 | 606.10 | 603.58 | 611.73 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 13:30:00 | 600.50 | 603.86 | 611.12 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-01-24 14:30:00 | 598.60 | 601.69 | 609.48 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Target hit | 2025-01-27 09:15:00 | 540.45 | 587.07 | 601.31 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 47 — BUY (started 2025-02-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-01 09:15:00 | 559.35 | 550.49 | 549.79 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-01 13:15:00 | 566.45 | 556.95 | 553.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-03 09:15:00 | 535.55 | 554.56 | 553.37 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-03 09:15:00 | 535.55 | 554.56 | 553.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-03 09:15:00 | 535.55 | 554.56 | 553.37 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-03 09:45:00 | 535.20 | 554.56 | 553.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 48 — SELL (started 2025-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-03 10:15:00 | 542.25 | 552.10 | 552.36 | EMA200 below EMA400 |

### Cycle 49 — BUY (started 2025-02-05 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-05 12:15:00 | 554.25 | 548.06 | 547.73 | EMA200 above EMA400 |

### Cycle 50 — SELL (started 2025-02-05 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-05 15:15:00 | 545.00 | 547.65 | 547.66 | EMA200 below EMA400 |

### Cycle 51 — BUY (started 2025-02-06 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-06 09:15:00 | 547.70 | 547.66 | 547.66 | EMA200 above EMA400 |

### Cycle 52 — SELL (started 2025-02-06 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-06 10:15:00 | 539.50 | 546.03 | 546.92 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-07 09:15:00 | 538.35 | 542.75 | 544.70 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-07 10:15:00 | 543.50 | 542.90 | 544.59 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-02-07 11:00:00 | 543.50 | 542.90 | 544.59 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-07 11:15:00 | 539.20 | 542.16 | 544.10 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 12:45:00 | 539.00 | 541.48 | 543.61 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 13:45:00 | 538.05 | 540.81 | 543.12 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-07 14:45:00 | 538.55 | 541.32 | 543.14 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-10 09:15:00 | 527.60 | 541.77 | 543.18 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 512.05 | 517.28 | 527.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 511.15 | 517.28 | 527.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 511.62 | 517.28 | 527.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-11 09:15:00 | 501.22 | 517.28 | 527.60 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-02-11 12:15:00 | 485.10 | 503.31 | 518.03 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 53 — BUY (started 2025-02-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-02-20 12:15:00 | 443.05 | 438.04 | 437.46 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-02-20 13:15:00 | 447.05 | 439.84 | 438.34 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-02-21 09:15:00 | 434.25 | 439.70 | 438.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-21 09:15:00 | 434.25 | 439.70 | 438.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 09:15:00 | 434.25 | 439.70 | 438.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-02-21 10:00:00 | 434.25 | 439.70 | 438.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-21 10:15:00 | 434.25 | 438.61 | 438.33 | EMA400 retest candle locked (from upside) |

### Cycle 54 — SELL (started 2025-02-21 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-02-21 11:15:00 | 436.00 | 438.09 | 438.12 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-02-21 13:15:00 | 428.55 | 435.44 | 436.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-02-24 11:15:00 | 432.45 | 432.03 | 434.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-02-24 11:15:00 | 432.45 | 432.03 | 434.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 11:15:00 | 432.45 | 432.03 | 434.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 11:45:00 | 434.95 | 432.03 | 434.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-24 14:15:00 | 430.50 | 431.30 | 433.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-24 15:00:00 | 430.50 | 431.30 | 433.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 09:15:00 | 432.90 | 431.25 | 433.02 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-02-25 09:30:00 | 437.00 | 431.25 | 433.02 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-02-25 10:15:00 | 428.50 | 430.70 | 432.61 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-25 11:45:00 | 425.75 | 430.03 | 432.13 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-02-27 09:15:00 | 423.20 | 430.51 | 431.78 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 404.46 | 415.79 | 422.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-02-28 09:15:00 | 402.04 | 415.79 | 422.16 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2025-02-28 14:15:00 | 408.35 | 408.07 | 415.26 | SL hit (close>ema200) qty=0.50 sl=408.07 alert=retest2 |

### Cycle 55 — BUY (started 2025-03-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-04 11:15:00 | 415.25 | 411.01 | 410.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-04 13:15:00 | 424.10 | 414.70 | 412.44 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-07 13:15:00 | 447.45 | 449.60 | 442.39 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-03-07 14:00:00 | 447.45 | 449.60 | 442.39 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-07 15:15:00 | 445.00 | 448.05 | 442.90 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 09:15:00 | 440.85 | 448.05 | 442.90 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 09:15:00 | 440.95 | 446.63 | 442.72 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 10:00:00 | 440.95 | 446.63 | 442.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 10:15:00 | 438.10 | 444.93 | 442.30 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 11:00:00 | 438.10 | 444.93 | 442.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-10 11:15:00 | 436.10 | 443.16 | 441.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-10 12:00:00 | 436.10 | 443.16 | 441.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 56 — SELL (started 2025-03-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-10 13:15:00 | 432.35 | 439.76 | 440.36 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-10 14:15:00 | 429.35 | 437.68 | 439.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-13 09:15:00 | 411.00 | 409.28 | 416.26 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-03-13 10:00:00 | 411.00 | 409.28 | 416.26 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-18 09:15:00 | 405.75 | 400.45 | 404.67 | EMA400 retest candle locked (from downside) |

### Cycle 57 — BUY (started 2025-03-18 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-18 15:15:00 | 411.50 | 406.80 | 406.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-03-19 09:15:00 | 424.15 | 410.27 | 407.94 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-03-20 09:15:00 | 417.65 | 420.21 | 415.49 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-20 09:15:00 | 417.65 | 420.21 | 415.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-20 09:15:00 | 417.65 | 420.21 | 415.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-03-20 10:00:00 | 417.65 | 420.21 | 415.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-21 09:15:00 | 437.70 | 425.10 | 420.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 10:15:00 | 441.85 | 425.10 | 420.40 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 12:45:00 | 439.40 | 430.49 | 424.29 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-03-21 14:15:00 | 441.95 | 431.57 | 425.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Target hit | 2025-03-24 13:15:00 | 486.04 | 452.86 | 440.17 | Target hit (10%) qty=1.00 alert=retest2 |

### Cycle 58 — SELL (started 2025-03-26 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-26 13:15:00 | 438.50 | 444.58 | 445.32 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-26 14:15:00 | 430.20 | 441.71 | 443.95 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-03-27 12:15:00 | 441.70 | 434.01 | 438.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-03-27 12:15:00 | 441.70 | 434.01 | 438.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 12:15:00 | 441.70 | 434.01 | 438.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:00:00 | 441.70 | 434.01 | 438.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 13:15:00 | 440.35 | 435.28 | 438.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 13:45:00 | 441.10 | 435.28 | 438.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-03-27 14:15:00 | 441.50 | 436.52 | 438.91 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-03-27 15:00:00 | 441.50 | 436.52 | 438.91 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 59 — BUY (started 2025-03-28 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-03-28 11:15:00 | 444.80 | 441.08 | 440.58 | EMA200 above EMA400 |

### Cycle 60 — SELL (started 2025-03-28 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-03-28 13:15:00 | 435.10 | 439.67 | 440.01 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-03-28 14:15:00 | 431.00 | 437.94 | 439.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-01 12:15:00 | 434.35 | 434.20 | 436.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-04-01 12:15:00 | 434.35 | 434.20 | 436.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-01 12:15:00 | 434.35 | 434.20 | 436.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-01 12:30:00 | 435.55 | 434.20 | 436.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 10:15:00 | 440.00 | 434.08 | 435.45 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-04-02 11:00:00 | 440.00 | 434.08 | 435.45 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-02 11:15:00 | 436.80 | 434.62 | 435.58 | EMA400 retest candle locked (from downside) |

### Cycle 61 — BUY (started 2025-04-02 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 13:15:00 | 440.50 | 436.18 | 436.14 | EMA200 above EMA400 |

### Cycle 62 — SELL (started 2025-04-02 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-02 14:15:00 | 435.75 | 436.09 | 436.11 | EMA200 below EMA400 |

### Cycle 63 — BUY (started 2025-04-02 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-02 15:15:00 | 437.50 | 436.37 | 436.23 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-03 09:15:00 | 440.50 | 437.20 | 436.62 | Break + close above crossover candle high |

### Cycle 64 — SELL (started 2025-04-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-04 09:15:00 | 424.25 | 437.86 | 437.96 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-04 13:15:00 | 419.40 | 429.19 | 433.36 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-08 09:15:00 | 392.90 | 391.02 | 405.37 | EMA200 retest candle locked (from downside) |
| First Entry (SELL) — retest1 break (cap 3 attempts) | 2025-04-09 09:30:00 | 389.45 | 393.90 | 400.53 | SELL ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-11 09:15:00 | 397.95 | 395.09 | 397.73 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-04-11 09:15:00 | 397.95 | 395.09 | 397.73 | SL hit (close>ema400) qty=1.00 sl=397.73 alert=retest1 |

### Cycle 65 — BUY (started 2025-04-15 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-04-15 09:15:00 | 413.45 | 400.57 | 399.37 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-04-15 11:15:00 | 415.60 | 405.35 | 401.86 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-04-17 13:15:00 | 431.20 | 432.53 | 425.19 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-04-17 14:00:00 | 431.20 | 432.53 | 425.19 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-21 09:15:00 | 437.05 | 432.45 | 426.89 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-21 10:15:00 | 442.00 | 432.45 | 426.89 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 12:00:00 | 437.80 | 440.17 | 439.08 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 12:45:00 | 438.40 | 439.64 | 438.94 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-04-23 13:15:00 | 438.00 | 439.64 | 438.94 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-23 13:15:00 | 437.80 | 439.27 | 438.83 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-04-23 14:00:00 | 437.80 | 439.27 | 438.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-04-23 14:15:00 | 435.00 | 438.41 | 438.48 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 66 — SELL (started 2025-04-23 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-04-23 14:15:00 | 435.00 | 438.41 | 438.48 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-04-25 09:15:00 | 418.80 | 432.61 | 435.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-04-28 10:15:00 | 420.65 | 419.70 | 425.22 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-04-28 11:00:00 | 420.65 | 419.70 | 425.22 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-04-29 09:15:00 | 418.05 | 419.60 | 422.82 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 10:15:00 | 417.30 | 419.60 | 422.82 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 11:45:00 | 415.60 | 418.87 | 421.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 13:00:00 | 417.95 | 418.68 | 421.57 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-04-29 13:45:00 | 417.90 | 417.84 | 420.92 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 12:15:00 | 408.00 | 406.61 | 410.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 12:45:00 | 408.40 | 406.61 | 410.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 13:15:00 | 417.30 | 408.74 | 411.35 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-02 13:45:00 | 423.40 | 408.74 | 411.35 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-02 14:15:00 | 411.95 | 409.39 | 411.40 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-02 15:15:00 | 410.00 | 409.39 | 411.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-05 09:15:00 | 422.30 | 412.07 | 412.28 | SL hit (close>static) qty=1.00 sl=418.75 alert=retest2 |

### Cycle 67 — BUY (started 2025-05-05 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-05 10:15:00 | 418.35 | 413.32 | 412.83 | EMA200 above EMA400 |

### Cycle 68 — SELL (started 2025-05-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-06 11:15:00 | 408.75 | 414.05 | 414.09 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-06 12:15:00 | 404.80 | 412.20 | 413.25 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-07 10:15:00 | 404.40 | 403.74 | 408.02 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-07 11:00:00 | 404.40 | 403.74 | 408.02 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 11:15:00 | 404.95 | 403.98 | 407.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-07 11:30:00 | 406.70 | 403.98 | 407.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-07 15:15:00 | 407.65 | 404.18 | 406.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:15:00 | 411.00 | 404.18 | 406.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-08 09:15:00 | 415.35 | 406.42 | 407.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-08 09:30:00 | 416.00 | 406.42 | 407.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 69 — BUY (started 2025-05-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-08 10:15:00 | 414.55 | 408.04 | 408.03 | EMA200 above EMA400 |

### Cycle 70 — SELL (started 2025-05-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-08 13:15:00 | 404.15 | 408.18 | 408.20 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-08 14:15:00 | 399.70 | 406.49 | 407.43 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-12 09:15:00 | 420.20 | 399.70 | 401.27 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-05-12 09:15:00 | 420.20 | 399.70 | 401.27 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-12 09:15:00 | 420.20 | 399.70 | 401.27 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-05-12 10:00:00 | 420.20 | 399.70 | 401.27 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 71 — BUY (started 2025-05-12 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-12 10:15:00 | 421.10 | 403.98 | 403.07 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-12 12:15:00 | 424.75 | 410.92 | 406.56 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-14 15:15:00 | 433.00 | 433.28 | 427.94 | EMA200 retest candle locked (from upside) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 09:15:00 | 438.00 | 433.28 | 427.94 | BUY ENTRY1 attempt 1/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 13:45:00 | 435.90 | 435.77 | 431.44 | BUY ENTRY1 attempt 2/4 (v1.4 edge cross+close) |
| First Entry (BUY) — retest1 break (cap 3 attempts) | 2025-05-15 14:15:00 | 435.80 | 435.77 | 431.44 | BUY ENTRY1 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-16 14:15:00 | 459.90 | 446.44 | 439.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-16 14:15:00 | 457.69 | 446.44 | 439.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-05-16 14:15:00 | 457.59 | 446.44 | 439.61 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest1 |
| Stop hit — per-position SL triggered | 2025-05-19 13:15:00 | 450.15 | 451.61 | 445.79 | SL hit (close<ema200) qty=0.50 sl=451.61 alert=retest1 |

### Cycle 72 — SELL (started 2025-05-20 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-20 12:15:00 | 434.50 | 444.22 | 444.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-20 14:15:00 | 430.75 | 439.87 | 442.38 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-05-21 15:15:00 | 432.50 | 431.82 | 435.86 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-05-22 09:15:00 | 434.00 | 431.82 | 435.86 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-22 09:15:00 | 434.75 | 432.40 | 435.76 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 12:45:00 | 429.05 | 431.97 | 434.73 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-05-22 13:30:00 | 428.80 | 430.76 | 433.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-05-26 10:15:00 | 439.85 | 433.52 | 432.84 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 73 — BUY (started 2025-05-26 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-26 10:15:00 | 439.85 | 433.52 | 432.84 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-05-27 15:15:00 | 442.10 | 439.67 | 437.36 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-05-28 14:15:00 | 441.65 | 443.06 | 440.50 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-05-28 15:00:00 | 441.65 | 443.06 | 440.50 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-28 15:15:00 | 439.65 | 442.38 | 440.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:15:00 | 438.55 | 442.38 | 440.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 09:15:00 | 437.60 | 441.42 | 440.16 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 09:45:00 | 436.60 | 441.42 | 440.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 10:15:00 | 437.75 | 440.69 | 439.95 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 10:30:00 | 437.20 | 440.69 | 439.95 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-05-29 11:15:00 | 437.75 | 440.10 | 439.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-05-29 11:30:00 | 437.80 | 440.10 | 439.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 74 — SELL (started 2025-05-29 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-29 12:15:00 | 436.80 | 439.44 | 439.48 | EMA200 below EMA400 |

### Cycle 75 — BUY (started 2025-05-30 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-05-30 12:15:00 | 442.00 | 439.03 | 439.00 | EMA200 above EMA400 |

### Cycle 76 — SELL (started 2025-05-30 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-05-30 13:15:00 | 434.90 | 438.21 | 438.63 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-05-30 15:15:00 | 432.50 | 436.38 | 437.68 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-04 10:15:00 | 423.40 | 416.41 | 420.42 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-04 10:15:00 | 423.40 | 416.41 | 420.42 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 10:15:00 | 423.40 | 416.41 | 420.42 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-04 11:00:00 | 423.40 | 416.41 | 420.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-04 11:15:00 | 422.45 | 417.62 | 420.60 | EMA400 retest candle locked (from downside) |

### Cycle 77 — BUY (started 2025-06-04 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-04 12:15:00 | 447.20 | 423.53 | 423.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-04 13:15:00 | 450.30 | 428.89 | 425.50 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-06 09:15:00 | 447.60 | 452.75 | 444.13 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-06-06 10:00:00 | 447.60 | 452.75 | 444.13 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 13:15:00 | 458.55 | 466.57 | 463.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-06-11 13:45:00 | 459.50 | 466.57 | 463.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-11 14:15:00 | 464.75 | 466.20 | 463.61 | EMA400 retest candle locked (from upside) |

### Cycle 78 — SELL (started 2025-06-12 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-06-12 11:15:00 | 454.70 | 461.11 | 461.82 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-06-12 13:15:00 | 443.20 | 456.50 | 459.53 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-06-16 13:15:00 | 431.00 | 430.59 | 438.44 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-06-16 13:45:00 | 430.85 | 430.59 | 438.44 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-19 09:15:00 | 426.75 | 424.26 | 427.56 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-19 09:30:00 | 431.60 | 424.26 | 427.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 10:15:00 | 421.65 | 419.13 | 422.38 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 10:30:00 | 422.10 | 419.13 | 422.38 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 11:15:00 | 421.40 | 419.58 | 422.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-06-20 11:30:00 | 421.70 | 419.58 | 422.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-20 14:15:00 | 418.50 | 419.73 | 421.72 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-06-20 15:15:00 | 418.00 | 419.73 | 421.72 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-06-23 09:15:00 | 426.45 | 420.80 | 421.84 | SL hit (close>static) qty=1.00 sl=423.45 alert=retest2 |

### Cycle 79 — BUY (started 2025-06-23 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-06-23 12:15:00 | 428.20 | 423.29 | 422.81 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-06-24 09:15:00 | 452.45 | 430.35 | 426.35 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-06-24 15:15:00 | 445.00 | 445.23 | 437.30 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-06-26 12:15:00 | 443.05 | 444.80 | 442.16 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-06-26 12:15:00 | 443.05 | 444.80 | 442.16 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-06-26 14:30:00 | 445.65 | 445.31 | 442.84 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-01 09:15:00 | 441.50 | 447.75 | 447.65 | SL hit (close<static) qty=1.00 sl=442.15 alert=retest2 |

### Cycle 80 — SELL (started 2025-07-01 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-01 10:15:00 | 441.90 | 446.58 | 447.13 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-01 12:15:00 | 439.70 | 444.74 | 446.17 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-02 12:15:00 | 442.00 | 441.32 | 443.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-02 12:15:00 | 442.00 | 441.32 | 443.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-02 12:15:00 | 442.00 | 441.32 | 443.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-02 12:45:00 | 443.65 | 441.32 | 443.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 09:15:00 | 442.40 | 441.12 | 442.58 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-03 09:30:00 | 442.30 | 441.12 | 442.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-03 10:15:00 | 443.65 | 441.62 | 442.68 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 12:15:00 | 441.40 | 441.74 | 442.63 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 14:15:00 | 441.65 | 442.07 | 442.64 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-03 15:00:00 | 441.95 | 442.05 | 442.58 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-04 09:15:00 | 449.25 | 443.32 | 443.05 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 81 — BUY (started 2025-07-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-04 09:15:00 | 449.25 | 443.32 | 443.05 | EMA200 above EMA400 |

### Cycle 82 — SELL (started 2025-07-08 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-08 11:15:00 | 439.85 | 444.32 | 444.45 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-08 15:15:00 | 439.00 | 442.12 | 443.27 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-09 09:15:00 | 443.40 | 442.38 | 443.29 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-09 09:15:00 | 443.40 | 442.38 | 443.29 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-09 09:15:00 | 443.40 | 442.38 | 443.29 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-09 09:45:00 | 443.10 | 442.38 | 443.29 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 83 — BUY (started 2025-07-09 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-09 10:15:00 | 452.80 | 444.46 | 444.15 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-09 14:15:00 | 462.25 | 453.25 | 448.81 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-11 09:15:00 | 459.50 | 462.72 | 457.77 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-11 10:00:00 | 459.50 | 462.72 | 457.77 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 10:15:00 | 455.95 | 461.36 | 457.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-11 11:00:00 | 455.95 | 461.36 | 457.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-11 11:15:00 | 456.00 | 460.29 | 457.46 | EMA400 retest candle locked (from upside) |

### Cycle 84 — SELL (started 2025-07-14 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-14 11:15:00 | 455.45 | 456.41 | 456.44 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-14 12:15:00 | 451.95 | 455.52 | 456.03 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-15 09:15:00 | 464.15 | 455.55 | 455.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-07-15 09:15:00 | 464.15 | 455.55 | 455.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-15 09:15:00 | 464.15 | 455.55 | 455.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-15 10:00:00 | 464.15 | 455.55 | 455.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 85 — BUY (started 2025-07-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-07-15 10:15:00 | 464.40 | 457.32 | 456.47 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-07-15 11:15:00 | 468.00 | 459.45 | 457.52 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-07-17 11:15:00 | 511.85 | 514.28 | 499.63 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-07-17 12:00:00 | 511.85 | 514.28 | 499.63 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 14:15:00 | 505.00 | 509.35 | 505.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-18 15:00:00 | 505.00 | 509.35 | 505.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-18 15:15:00 | 504.20 | 508.32 | 505.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-21 09:15:00 | 511.00 | 508.32 | 505.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-21 09:15:00 | 507.95 | 508.25 | 505.40 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-21 10:30:00 | 516.15 | 509.07 | 506.04 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-22 14:15:00 | 514.50 | 508.93 | 507.52 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 11:00:00 | 514.50 | 511.27 | 509.25 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-23 11:45:00 | 512.75 | 511.62 | 509.59 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 14:15:00 | 511.70 | 512.32 | 510.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-07-23 15:00:00 | 511.70 | 512.32 | 510.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-23 15:15:00 | 511.50 | 512.16 | 510.56 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-07-24 09:15:00 | 515.00 | 512.16 | 510.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-07-24 12:15:00 | 510.30 | 513.37 | 511.89 | SL hit (close<static) qty=1.00 sl=510.45 alert=retest2 |

### Cycle 86 — SELL (started 2025-07-25 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-07-25 09:15:00 | 490.55 | 507.41 | 509.51 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-07-25 10:15:00 | 484.25 | 502.78 | 507.21 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-07-29 09:15:00 | 473.75 | 473.61 | 483.31 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-07-29 09:45:00 | 470.75 | 473.61 | 483.31 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 13:15:00 | 481.55 | 475.75 | 481.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 14:00:00 | 481.55 | 475.75 | 481.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 14:15:00 | 482.00 | 477.00 | 481.30 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-29 15:15:00 | 484.75 | 477.00 | 481.30 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-29 15:15:00 | 484.75 | 478.55 | 481.61 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-07-30 09:15:00 | 482.10 | 478.55 | 481.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-07-30 09:15:00 | 480.40 | 478.92 | 481.50 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 10:30:00 | 478.35 | 479.23 | 481.40 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 13:15:00 | 476.85 | 479.40 | 481.11 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-07-30 15:00:00 | 478.75 | 479.25 | 480.74 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 11:15:00 | 454.43 | 464.36 | 470.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 11:15:00 | 454.81 | 464.36 | 470.81 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2025-08-01 12:15:00 | 453.01 | 462.19 | 469.23 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Target hit | 2025-08-04 12:15:00 | 430.52 | 441.88 | 453.90 | Target hit (10%) qty=0.50 alert=retest2 |

### Cycle 87 — BUY (started 2025-08-11 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-11 14:15:00 | 431.00 | 424.82 | 424.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-12 09:15:00 | 433.65 | 427.14 | 425.25 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-12 14:15:00 | 426.60 | 428.23 | 426.62 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-12 14:15:00 | 426.60 | 428.23 | 426.62 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 14:15:00 | 426.60 | 428.23 | 426.62 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-12 15:00:00 | 426.60 | 428.23 | 426.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-12 15:15:00 | 424.00 | 427.39 | 426.38 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 09:45:00 | 427.15 | 427.05 | 426.32 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-13 10:30:00 | 430.20 | 427.97 | 426.81 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-08-18 09:30:00 | 427.15 | 433.21 | 432.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-08-18 11:15:00 | 425.60 | 430.98 | 431.35 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 88 — SELL (started 2025-08-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-18 11:15:00 | 425.60 | 430.98 | 431.35 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-18 14:15:00 | 424.50 | 428.51 | 430.02 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-08-19 09:15:00 | 431.80 | 428.60 | 429.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-19 09:15:00 | 431.80 | 428.60 | 429.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 09:15:00 | 431.80 | 428.60 | 429.77 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 10:15:00 | 434.05 | 428.60 | 429.77 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-19 10:15:00 | 436.75 | 430.23 | 430.41 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-08-19 11:00:00 | 436.75 | 430.23 | 430.41 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 89 — BUY (started 2025-08-19 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-08-19 11:15:00 | 434.10 | 431.01 | 430.74 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-08-20 12:15:00 | 442.40 | 436.14 | 433.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-08-26 09:15:00 | 475.10 | 477.69 | 471.61 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-08-26 09:15:00 | 475.10 | 477.69 | 471.61 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 09:15:00 | 475.10 | 477.69 | 471.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 09:30:00 | 468.00 | 477.69 | 471.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 11:15:00 | 469.20 | 475.25 | 471.51 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 12:00:00 | 469.20 | 475.25 | 471.51 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-08-26 12:15:00 | 462.30 | 472.66 | 470.67 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-08-26 13:00:00 | 462.30 | 472.66 | 470.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 90 — SELL (started 2025-08-26 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-08-26 14:15:00 | 453.60 | 466.41 | 468.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-08-29 12:15:00 | 450.00 | 458.67 | 461.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-01 12:15:00 | 452.00 | 450.23 | 454.89 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-01 12:45:00 | 451.20 | 450.23 | 454.89 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 14:15:00 | 452.40 | 450.97 | 454.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-01 14:30:00 | 454.30 | 450.97 | 454.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-01 15:15:00 | 455.00 | 451.77 | 454.49 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 09:15:00 | 455.90 | 451.77 | 454.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 09:15:00 | 461.10 | 453.64 | 455.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:00:00 | 461.10 | 453.64 | 455.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-02 10:15:00 | 461.50 | 455.21 | 455.67 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-02 10:45:00 | 462.60 | 455.21 | 455.67 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 91 — BUY (started 2025-09-02 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-02 11:15:00 | 461.40 | 456.45 | 456.19 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-03 09:15:00 | 464.00 | 458.18 | 457.12 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-04 09:15:00 | 457.30 | 459.59 | 458.64 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-04 09:15:00 | 457.30 | 459.59 | 458.64 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 09:15:00 | 457.30 | 459.59 | 458.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:00:00 | 457.30 | 459.59 | 458.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-04 10:15:00 | 454.35 | 458.54 | 458.25 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-04 10:45:00 | 455.80 | 458.54 | 458.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 92 — SELL (started 2025-09-04 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-04 11:15:00 | 455.25 | 457.88 | 457.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-04 13:15:00 | 451.00 | 455.86 | 457.00 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-05 12:15:00 | 455.00 | 451.69 | 453.83 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-05 12:15:00 | 455.00 | 451.69 | 453.83 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 12:15:00 | 455.00 | 451.69 | 453.83 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:00:00 | 455.00 | 451.69 | 453.83 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 13:15:00 | 457.50 | 452.85 | 454.16 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-05 13:45:00 | 458.65 | 452.85 | 454.16 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-05 15:15:00 | 453.50 | 453.35 | 454.18 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 09:15:00 | 458.10 | 453.35 | 454.18 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-08 09:15:00 | 459.45 | 454.57 | 454.66 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-08 10:00:00 | 459.45 | 454.57 | 454.66 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 93 — BUY (started 2025-09-08 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-08 10:15:00 | 455.40 | 454.74 | 454.73 | EMA200 above EMA400 |

### Cycle 94 — SELL (started 2025-09-08 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-08 13:15:00 | 452.75 | 454.77 | 454.79 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-08 14:15:00 | 449.95 | 453.80 | 454.35 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-10 09:15:00 | 451.30 | 449.77 | 451.49 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-10 09:15:00 | 451.30 | 449.77 | 451.49 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-10 09:15:00 | 451.30 | 449.77 | 451.49 | EMA400 retest candle locked (from downside) |

### Cycle 95 — BUY (started 2025-09-10 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 12:15:00 | 455.20 | 452.41 | 452.37 | EMA200 above EMA400 |

### Cycle 96 — SELL (started 2025-09-10 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-10 13:15:00 | 449.85 | 451.90 | 452.14 | EMA200 below EMA400 |

### Cycle 97 — BUY (started 2025-09-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-10 15:15:00 | 457.95 | 453.00 | 452.59 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-11 10:15:00 | 463.30 | 455.89 | 454.03 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-12 14:15:00 | 468.40 | 468.77 | 463.99 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-09-12 14:45:00 | 466.00 | 468.77 | 463.99 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-15 09:15:00 | 468.15 | 468.17 | 464.52 | EMA400 retest candle locked (from upside) |

### Cycle 98 — SELL (started 2025-09-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-16 12:15:00 | 462.40 | 463.71 | 463.78 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-16 14:15:00 | 459.25 | 462.51 | 463.20 | Break + close below crossover candle low |

### Cycle 99 — BUY (started 2025-09-17 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-17 09:15:00 | 471.15 | 463.78 | 463.63 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-18 09:15:00 | 474.90 | 466.79 | 465.31 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-22 15:15:00 | 489.00 | 490.20 | 484.91 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-23 09:15:00 | 483.25 | 488.81 | 484.76 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 09:15:00 | 483.25 | 488.81 | 484.76 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 10:00:00 | 483.25 | 488.81 | 484.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 10:15:00 | 481.15 | 487.28 | 484.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 11:00:00 | 481.15 | 487.28 | 484.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 13:15:00 | 482.75 | 485.33 | 484.15 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 14:00:00 | 482.75 | 485.33 | 484.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-23 14:15:00 | 482.35 | 484.74 | 483.98 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-23 15:00:00 | 482.35 | 484.74 | 483.98 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 100 — SELL (started 2025-09-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-24 10:15:00 | 475.60 | 482.04 | 482.87 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-24 11:15:00 | 474.35 | 480.50 | 482.10 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-24 13:15:00 | 487.50 | 481.50 | 482.25 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-24 13:15:00 | 487.50 | 481.50 | 482.25 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 13:15:00 | 487.50 | 481.50 | 482.25 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-09-24 13:45:00 | 488.40 | 481.50 | 482.25 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-24 14:15:00 | 487.50 | 482.70 | 482.73 | EMA400 retest candle locked (from downside) |

### Cycle 101 — BUY (started 2025-09-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-09-24 15:15:00 | 489.05 | 483.97 | 483.30 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-09-25 10:15:00 | 491.50 | 486.52 | 484.64 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-09-25 13:15:00 | 481.85 | 485.84 | 484.82 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-09-25 13:15:00 | 481.85 | 485.84 | 484.82 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-09-25 13:15:00 | 481.85 | 485.84 | 484.82 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-09-25 14:00:00 | 481.85 | 485.84 | 484.82 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 102 — SELL (started 2025-09-25 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-09-25 14:15:00 | 476.90 | 484.05 | 484.10 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-09-26 11:15:00 | 475.75 | 481.53 | 482.81 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-09-30 09:15:00 | 461.00 | 460.35 | 467.80 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2025-09-30 10:00:00 | 461.00 | 460.35 | 467.80 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 09:15:00 | 457.85 | 456.06 | 461.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 09:45:00 | 458.70 | 456.06 | 461.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 10:15:00 | 460.65 | 456.98 | 461.33 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 10:45:00 | 462.00 | 456.98 | 461.33 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 11:15:00 | 465.45 | 458.67 | 461.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 12:00:00 | 465.45 | 458.67 | 461.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-01 12:15:00 | 466.40 | 460.22 | 462.13 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-01 13:00:00 | 466.40 | 460.22 | 462.13 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-03 11:15:00 | 460.40 | 461.50 | 462.23 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-03 12:00:00 | 460.40 | 461.50 | 462.23 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-06 09:15:00 | 457.35 | 459.22 | 460.69 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-06 09:30:00 | 461.50 | 459.22 | 460.69 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 09:15:00 | 444.35 | 439.37 | 442.52 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 09:45:00 | 446.10 | 439.37 | 442.52 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-10 10:15:00 | 444.95 | 440.48 | 442.74 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-10 10:30:00 | 444.60 | 440.48 | 442.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 103 — BUY (started 2025-10-10 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-10 15:15:00 | 445.00 | 443.82 | 443.81 | EMA200 above EMA400 |

### Cycle 104 — SELL (started 2025-10-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-13 09:15:00 | 435.35 | 442.12 | 443.04 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-10-13 10:15:00 | 433.70 | 440.44 | 442.19 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-10-14 14:15:00 | 433.85 | 432.34 | 435.54 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-10-14 14:15:00 | 433.85 | 432.34 | 435.54 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-14 14:15:00 | 433.85 | 432.34 | 435.54 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-10-14 15:00:00 | 433.85 | 432.34 | 435.54 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-15 10:15:00 | 432.30 | 432.81 | 435.00 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-15 14:00:00 | 431.70 | 432.71 | 434.43 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 10:45:00 | 431.65 | 432.21 | 433.56 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-16 11:30:00 | 431.50 | 431.81 | 433.26 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-10-17 10:45:00 | 431.35 | 431.44 | 432.37 | SELL ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-20 09:15:00 | 426.15 | 428.53 | 430.37 | EMA400 retest candle locked (from downside) |
| Stop hit — per-position SL triggered | 2025-10-23 09:15:00 | 434.25 | 429.54 | 429.46 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 105 — BUY (started 2025-10-23 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-23 09:15:00 | 434.25 | 429.54 | 429.46 | EMA200 above EMA400 |

### Cycle 106 — SELL (started 2025-10-24 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-10-24 10:15:00 | 428.00 | 429.86 | 429.91 | EMA200 below EMA400 |

### Cycle 107 — BUY (started 2025-10-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-10-24 15:15:00 | 432.50 | 430.37 | 430.10 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-10-27 09:15:00 | 434.30 | 431.16 | 430.48 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-10-30 09:15:00 | 478.35 | 478.37 | 467.71 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-10-30 10:00:00 | 478.35 | 478.37 | 467.71 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 12:15:00 | 473.60 | 476.37 | 473.11 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 12:45:00 | 473.00 | 476.37 | 473.11 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 13:15:00 | 472.65 | 475.63 | 473.07 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 13:45:00 | 472.85 | 475.63 | 473.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 14:15:00 | 469.55 | 474.41 | 472.75 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-10-31 15:00:00 | 469.55 | 474.41 | 472.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-10-31 15:15:00 | 471.80 | 473.89 | 472.66 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 09:15:00 | 475.60 | 473.89 | 472.66 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 12:45:00 | 473.85 | 472.73 | 472.43 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 14:15:00 | 472.75 | 472.50 | 472.35 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-11-03 15:15:00 | 473.95 | 472.33 | 472.29 | BUY ENTRY2 attempt 4/4 (v1.4 edge cross+close) |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-03 15:15:00 | 473.95 | 472.65 | 472.44 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-11-04 09:15:00 | 471.50 | 472.65 | 472.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Stop hit — per-position SL triggered | 2025-11-04 09:15:00 | 466.05 | 471.33 | 471.86 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 108 — SELL (started 2025-11-04 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-04 09:15:00 | 466.05 | 471.33 | 471.86 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-04 10:15:00 | 462.50 | 469.57 | 471.01 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-07 10:15:00 | 473.50 | 448.12 | 453.12 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-07 10:15:00 | 473.50 | 448.12 | 453.12 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 10:15:00 | 473.50 | 448.12 | 453.12 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 11:00:00 | 473.50 | 448.12 | 453.12 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-07 11:15:00 | 476.65 | 453.83 | 455.26 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-07 12:00:00 | 476.65 | 453.83 | 455.26 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 109 — BUY (started 2025-11-07 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-07 12:15:00 | 476.00 | 458.26 | 457.15 | EMA200 above EMA400 |

### Cycle 110 — SELL (started 2025-11-13 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-13 13:15:00 | 461.60 | 465.96 | 466.39 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-14 09:15:00 | 456.75 | 462.81 | 464.73 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-17 14:15:00 | 465.20 | 452.95 | 456.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-17 14:15:00 | 465.20 | 452.95 | 456.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 14:15:00 | 465.20 | 452.95 | 456.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-11-17 15:00:00 | 465.20 | 452.95 | 456.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-17 15:15:00 | 462.00 | 454.76 | 456.88 | EMA400 retest candle locked (from downside) |

### Cycle 111 — BUY (started 2025-11-18 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-11-18 11:15:00 | 465.75 | 459.25 | 458.60 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-11-18 12:15:00 | 470.90 | 461.58 | 459.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-11-18 15:15:00 | 462.90 | 463.41 | 461.18 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-11-19 09:15:00 | 466.65 | 463.41 | 461.18 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-19 09:15:00 | 469.40 | 464.61 | 461.92 | EMA400 retest candle locked (from upside) |

### Cycle 112 — SELL (started 2025-11-21 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-11-21 14:15:00 | 462.35 | 466.75 | 467.03 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-11-25 09:15:00 | 454.65 | 458.79 | 461.65 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-11-27 09:15:00 | 443.20 | 442.89 | 448.26 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-11-27 09:15:00 | 443.20 | 442.89 | 448.26 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-11-27 09:15:00 | 443.20 | 442.89 | 448.26 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 13:15:00 | 440.45 | 442.55 | 446.78 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-27 15:00:00 | 438.80 | 441.59 | 445.59 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-11-28 10:00:00 | 440.15 | 441.05 | 444.63 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-01 09:15:00 | 448.05 | 445.94 | 445.81 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 113 — BUY (started 2025-12-01 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 09:15:00 | 448.05 | 445.94 | 445.81 | EMA200 above EMA400 |

### Cycle 114 — SELL (started 2025-12-01 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-01 11:15:00 | 442.05 | 445.18 | 445.49 | EMA200 below EMA400 |

### Cycle 115 — BUY (started 2025-12-01 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-01 14:15:00 | 453.45 | 446.82 | 446.17 | EMA200 above EMA400 |

### Cycle 116 — SELL (started 2025-12-03 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-03 14:15:00 | 446.90 | 448.11 | 448.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-03 15:15:00 | 446.00 | 447.69 | 447.96 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-04 10:15:00 | 447.80 | 447.24 | 447.68 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-04 10:15:00 | 447.80 | 447.24 | 447.68 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 10:15:00 | 447.80 | 447.24 | 447.68 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-04 10:30:00 | 453.15 | 447.24 | 447.68 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-04 11:15:00 | 446.50 | 447.09 | 447.57 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-04 12:45:00 | 445.40 | 446.68 | 447.34 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-04 14:15:00 | 454.40 | 448.07 | 447.85 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 117 — BUY (started 2025-12-04 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-04 14:15:00 | 454.40 | 448.07 | 447.85 | EMA200 above EMA400 |

### Cycle 118 — SELL (started 2025-12-05 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-05 13:15:00 | 447.60 | 447.94 | 447.98 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-08 09:15:00 | 436.95 | 445.52 | 446.84 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-08 14:15:00 | 446.30 | 441.43 | 443.97 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-08 14:15:00 | 446.30 | 441.43 | 443.97 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 14:15:00 | 446.30 | 441.43 | 443.97 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-08 15:00:00 | 446.30 | 441.43 | 443.97 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-08 15:15:00 | 445.00 | 442.15 | 444.07 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-09 09:15:00 | 437.30 | 442.15 | 444.07 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-09 11:15:00 | 447.45 | 442.86 | 443.88 | SL hit (close>static) qty=1.00 sl=446.45 alert=retest2 |

### Cycle 119 — BUY (started 2025-12-09 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-09 14:15:00 | 458.00 | 446.77 | 445.49 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2025-12-11 14:15:00 | 459.40 | 454.54 | 451.87 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2025-12-12 12:15:00 | 456.15 | 456.35 | 453.98 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2025-12-12 13:00:00 | 456.15 | 456.35 | 453.98 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 14:15:00 | 458.40 | 456.72 | 454.56 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-12 14:45:00 | 453.60 | 456.72 | 454.56 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-12 15:15:00 | 455.00 | 456.38 | 454.60 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2025-12-15 09:15:00 | 454.40 | 456.38 | 454.60 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-15 09:15:00 | 456.00 | 456.30 | 454.73 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2025-12-15 12:45:00 | 463.60 | 457.36 | 455.54 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-16 12:15:00 | 452.85 | 456.10 | 456.04 | SL hit (close<static) qty=1.00 sl=453.00 alert=retest2 |

### Cycle 120 — SELL (started 2025-12-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-23 10:15:00 | 466.05 | 468.16 | 468.24 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2025-12-23 11:15:00 | 464.50 | 467.43 | 467.90 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2025-12-24 14:15:00 | 464.50 | 463.95 | 465.15 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2025-12-24 14:15:00 | 464.50 | 463.95 | 465.15 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 14:15:00 | 464.50 | 463.95 | 465.15 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2025-12-24 15:00:00 | 464.50 | 463.95 | 465.15 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2025-12-24 15:15:00 | 461.90 | 463.54 | 464.86 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2025-12-26 09:30:00 | 459.40 | 463.04 | 464.51 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2025-12-26 11:15:00 | 466.50 | 464.13 | 464.77 | SL hit (close>static) qty=1.00 sl=465.50 alert=retest2 |

### Cycle 121 — BUY (started 2025-12-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2025-12-29 09:15:00 | 488.70 | 469.65 | 467.10 | EMA200 above EMA400 |

### Cycle 122 — SELL (started 2025-12-31 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2025-12-31 14:15:00 | 468.70 | 471.77 | 471.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-01 12:15:00 | 467.55 | 470.16 | 470.93 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-02 10:15:00 | 467.70 | 467.54 | 469.16 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-02 11:00:00 | 467.70 | 467.54 | 469.16 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-02 11:15:00 | 464.40 | 466.91 | 468.72 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-02 11:30:00 | 466.85 | 466.91 | 468.72 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 09:15:00 | 474.70 | 466.27 | 467.44 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 10:00:00 | 474.70 | 466.27 | 467.44 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-05 10:15:00 | 470.90 | 467.20 | 467.75 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-05 11:15:00 | 475.00 | 467.20 | 467.75 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 123 — BUY (started 2026-01-05 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-05 11:15:00 | 475.00 | 468.76 | 468.41 | EMA200 above EMA400 |

### Cycle 124 — SELL (started 2026-01-07 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-07 09:15:00 | 464.70 | 469.97 | 470.05 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-08 10:15:00 | 459.25 | 466.05 | 467.78 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-13 15:15:00 | 424.50 | 421.90 | 428.68 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-14 09:15:00 | 423.45 | 421.90 | 428.68 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-14 09:15:00 | 419.50 | 421.42 | 427.84 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 10:30:00 | 418.90 | 420.89 | 427.02 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-14 11:00:00 | 418.75 | 420.89 | 427.02 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-16 11:15:00 | 437.00 | 425.05 | 425.82 | SL hit (close>static) qty=1.00 sl=428.15 alert=retest2 |

### Cycle 125 — BUY (started 2026-01-16 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-16 12:15:00 | 434.30 | 426.90 | 426.59 | EMA200 above EMA400 |

### Cycle 126 — SELL (started 2026-01-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-19 10:15:00 | 419.95 | 427.19 | 427.29 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-19 11:15:00 | 416.85 | 425.12 | 426.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-21 12:15:00 | 407.00 | 406.48 | 412.04 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-21 13:00:00 | 407.00 | 406.48 | 412.04 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-22 09:15:00 | 406.00 | 403.78 | 408.74 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:15:00 | 403.70 | 403.78 | 408.74 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-01-22 10:45:00 | 403.10 | 403.54 | 408.18 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-01-23 09:15:00 | 421.30 | 406.27 | 407.22 | SL hit (close>static) qty=1.00 sl=411.00 alert=retest2 |

### Cycle 127 — BUY (started 2026-01-23 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-23 10:15:00 | 424.95 | 410.01 | 408.83 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-23 11:15:00 | 432.65 | 414.54 | 411.00 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-23 14:15:00 | 412.35 | 415.38 | 412.40 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-01-23 14:15:00 | 412.35 | 415.38 | 412.40 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 14:15:00 | 412.35 | 415.38 | 412.40 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-23 15:00:00 | 412.35 | 415.38 | 412.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-23 15:15:00 | 406.30 | 413.56 | 411.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-27 09:15:00 | 404.30 | 413.56 | 411.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-27 09:15:00 | 404.10 | 411.67 | 411.14 | EMA400 retest candle locked (from upside) |

### Cycle 128 — SELL (started 2026-01-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-01-27 10:15:00 | 401.55 | 409.65 | 410.27 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-01-27 13:15:00 | 398.50 | 405.21 | 407.91 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-01-27 15:15:00 | 405.00 | 404.69 | 407.17 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-01-28 09:15:00 | 418.60 | 404.69 | 407.17 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-28 09:15:00 | 421.00 | 407.95 | 408.43 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-01-28 10:00:00 | 421.00 | 407.95 | 408.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 129 — BUY (started 2026-01-28 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-01-28 10:15:00 | 423.55 | 411.07 | 409.80 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-01-28 12:15:00 | 424.80 | 415.87 | 412.33 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-01-29 09:15:00 | 417.40 | 419.61 | 415.54 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-01-29 10:00:00 | 417.40 | 419.61 | 415.54 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 10:15:00 | 416.30 | 418.94 | 415.61 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 10:45:00 | 416.90 | 418.94 | 415.61 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 11:15:00 | 414.25 | 418.01 | 415.49 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 12:00:00 | 414.25 | 418.01 | 415.49 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 12:15:00 | 414.90 | 417.38 | 415.43 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-01-29 13:00:00 | 414.90 | 417.38 | 415.43 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-01-29 13:15:00 | 421.60 | 418.23 | 415.99 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-29 14:45:00 | 422.80 | 419.03 | 416.56 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-01-30 09:30:00 | 422.80 | 421.00 | 417.92 | BUY ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-02-01 10:15:00 | 422.80 | 421.92 | 420.24 | BUY ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-01 12:15:00 | 412.40 | 419.87 | 419.73 | SL hit (close<static) qty=1.00 sl=413.30 alert=retest2 |

### Cycle 130 — SELL (started 2026-02-01 13:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-01 13:15:00 | 409.95 | 417.88 | 418.84 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-01 14:15:00 | 407.05 | 415.72 | 417.77 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-02 12:15:00 | 410.50 | 410.16 | 413.83 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-02-02 12:45:00 | 411.10 | 410.16 | 413.83 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 13:15:00 | 413.00 | 410.73 | 413.76 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 14:00:00 | 413.00 | 410.73 | 413.76 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 14:15:00 | 413.20 | 411.22 | 413.71 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-02 14:45:00 | 414.30 | 411.22 | 413.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-02 15:15:00 | 412.75 | 411.53 | 413.62 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-02-03 09:15:00 | 423.50 | 411.53 | 413.62 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-03 09:15:00 | 421.80 | 413.58 | 414.36 | EMA400 retest candle locked (from downside) |

### Cycle 131 — BUY (started 2026-02-03 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-03 10:15:00 | 429.55 | 416.78 | 415.75 | EMA200 above EMA400 |

### Cycle 132 — SELL (started 2026-02-06 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-06 11:15:00 | 420.10 | 423.31 | 423.47 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-06 15:15:00 | 418.00 | 420.73 | 422.06 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-09 11:15:00 | 421.00 | 419.03 | 420.77 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-09 11:15:00 | 421.00 | 419.03 | 420.77 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-09 11:15:00 | 421.00 | 419.03 | 420.77 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 09:15:00 | 414.05 | 418.20 | 419.36 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-11 12:45:00 | 413.80 | 416.43 | 417.93 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-02-11 14:15:00 | 425.00 | 418.64 | 418.70 | SL hit (close>static) qty=1.00 sl=424.50 alert=retest2 |

### Cycle 133 — BUY (started 2026-02-11 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-02-11 15:15:00 | 424.65 | 419.84 | 419.25 | EMA200 above EMA400 |

### Cycle 134 — SELL (started 2026-02-12 14:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-02-12 14:15:00 | 416.45 | 418.61 | 418.89 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-02-13 09:15:00 | 408.65 | 416.50 | 417.87 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-02-17 09:15:00 | 408.90 | 408.18 | 410.80 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-02-17 09:15:00 | 408.90 | 408.18 | 410.80 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-02-17 09:15:00 | 408.90 | 408.18 | 410.80 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 11:30:00 | 405.05 | 408.56 | 409.99 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-19 13:30:00 | 404.65 | 407.41 | 409.20 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-02-23 09:15:00 | 405.20 | 403.89 | 405.49 | SELL ENTRY2 attempt 3/4 (v1.4 edge cross+close) |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 13:15:00 | 384.80 | 392.79 | 399.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 13:15:00 | 384.42 | 392.79 | 399.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Partial book — 50% qty, trail SL → EMA200 | 2026-02-23 13:15:00 | 384.94 | 392.79 | 399.30 | Partial book 0.50 @ 5%; trail SL->EMA200 alert=retest2 |
| Stop hit — per-position SL triggered | 2026-02-24 14:15:00 | 383.65 | 381.56 | 388.70 | SL hit (close>ema200) qty=0.50 sl=381.56 alert=retest2 |

### Cycle 135 — BUY (started 2026-03-11 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-11 10:15:00 | 355.00 | 351.69 | 351.32 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-12 10:15:00 | 359.10 | 355.00 | 353.54 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-13 09:15:00 | 350.55 | 359.46 | 357.17 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-13 09:15:00 | 350.55 | 359.46 | 357.17 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 09:15:00 | 350.55 | 359.46 | 357.17 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:00:00 | 350.55 | 359.46 | 357.17 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-13 10:15:00 | 347.75 | 357.12 | 356.32 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-13 10:30:00 | 348.00 | 357.12 | 356.32 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 136 — SELL (started 2026-03-13 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-13 11:15:00 | 348.20 | 355.34 | 355.58 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-13 12:15:00 | 341.95 | 352.66 | 354.34 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-16 14:15:00 | 338.15 | 337.32 | 343.57 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-16 15:00:00 | 338.15 | 337.32 | 343.57 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 09:15:00 | 341.70 | 334.37 | 338.06 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 10:00:00 | 341.70 | 334.37 | 338.06 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-18 10:15:00 | 344.50 | 336.40 | 338.64 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-18 11:00:00 | 344.50 | 336.40 | 338.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 137 — BUY (started 2026-03-18 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-18 12:15:00 | 348.70 | 340.39 | 340.16 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-18 13:15:00 | 350.50 | 342.41 | 341.10 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-19 09:15:00 | 335.55 | 342.22 | 341.47 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-19 09:15:00 | 335.55 | 342.22 | 341.47 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-19 09:15:00 | 335.55 | 342.22 | 341.47 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-19 10:15:00 | 334.95 | 342.22 | 341.47 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 138 — SELL (started 2026-03-19 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-19 10:15:00 | 333.60 | 340.50 | 340.75 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-19 12:15:00 | 331.60 | 337.73 | 339.39 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-03-24 11:15:00 | 313.55 | 313.53 | 319.05 | EMA200 retest candle locked (from downside) |
| ALERT2_SIDEWAYS | 2026-03-24 11:45:00 | 315.15 | 313.53 | 319.05 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 12:15:00 | 322.85 | 315.39 | 319.40 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-03-24 13:00:00 | 322.85 | 315.39 | 319.40 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-24 13:15:00 | 329.30 | 318.17 | 320.30 | EMA400 retest candle locked (from downside) |

### Cycle 139 — BUY (started 2026-03-24 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-03-24 15:15:00 | 330.00 | 321.77 | 321.64 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-03-25 09:15:00 | 333.40 | 324.09 | 322.71 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-03-25 14:15:00 | 326.20 | 327.96 | 325.58 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-03-25 14:15:00 | 326.20 | 327.96 | 325.58 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 14:15:00 | 326.20 | 327.96 | 325.58 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-25 15:00:00 | 326.20 | 327.96 | 325.58 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-25 15:15:00 | 324.70 | 327.31 | 325.50 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 09:15:00 | 318.90 | 327.31 | 325.50 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-03-27 09:15:00 | 313.60 | 324.57 | 324.42 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-03-27 10:00:00 | 313.60 | 324.57 | 324.42 | Sideways (15m bar) within 4 candles — back to Stage 4 |

### Cycle 140 — SELL (started 2026-03-27 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-03-27 10:15:00 | 310.50 | 321.75 | 323.16 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-03-27 14:15:00 | 309.10 | 315.15 | 319.20 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-01 09:15:00 | 322.30 | 308.65 | 312.24 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-01 09:15:00 | 322.30 | 308.65 | 312.24 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 09:15:00 | 322.30 | 308.65 | 312.24 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-01 10:00:00 | 322.30 | 308.65 | 312.24 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-01 10:15:00 | 319.00 | 310.72 | 312.85 | EMA400 retest candle locked (from downside) |

### Cycle 141 — BUY (started 2026-04-01 12:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-01 12:15:00 | 324.20 | 315.69 | 314.88 | EMA200 above EMA400 |

### Cycle 142 — SELL (started 2026-04-02 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-02 10:15:00 | 304.40 | 313.34 | 314.38 | EMA200 below EMA400 |

### Cycle 143 — BUY (started 2026-04-06 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-06 15:15:00 | 314.50 | 313.06 | 313.02 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-07 11:15:00 | 316.05 | 314.38 | 313.70 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-07 12:15:00 | 314.10 | 314.33 | 313.74 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-07 12:15:00 | 314.10 | 314.33 | 313.74 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 12:15:00 | 314.10 | 314.33 | 313.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 13:00:00 | 314.10 | 314.33 | 313.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 13:15:00 | 313.50 | 314.16 | 313.71 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 14:15:00 | 313.95 | 314.16 | 313.71 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 14:15:00 | 314.00 | 314.13 | 313.74 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-07 15:15:00 | 313.85 | 314.13 | 313.74 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-07 15:15:00 | 313.85 | 314.07 | 313.75 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-08 09:15:00 | 330.45 | 314.07 | 313.75 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-13 09:15:00 | 316.65 | 323.80 | 324.17 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 144 — SELL (started 2026-04-13 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-13 09:15:00 | 316.65 | 323.80 | 324.17 | EMA200 below EMA400 |

### Cycle 145 — BUY (started 2026-04-15 10:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-15 10:15:00 | 334.05 | 324.35 | 323.48 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-04-16 10:15:00 | 364.00 | 337.45 | 330.97 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-04-20 09:15:00 | 352.00 | 353.38 | 347.84 | EMA200 retest candle locked (from upside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-20 09:15:00 | 352.00 | 353.38 | 347.84 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 09:15:00 | 352.00 | 353.38 | 347.84 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 09:45:00 | 351.30 | 353.38 | 347.84 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 13:15:00 | 349.45 | 352.12 | 348.99 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 14:00:00 | 349.45 | 352.12 | 348.99 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 14:15:00 | 347.80 | 351.25 | 348.88 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-04-20 15:00:00 | 347.80 | 351.25 | 348.88 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-20 15:15:00 | 348.90 | 350.78 | 348.88 | EMA400 retest candle locked (from upside) |
| Second Entry (BUY) — retest2 break (cap 3 attempts) | 2026-04-21 09:30:00 | 352.10 | 351.22 | 349.25 | BUY ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-04-22 09:15:00 | 345.60 | 348.82 | 348.93 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 146 — SELL (started 2026-04-22 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-22 09:15:00 | 345.60 | 348.82 | 348.93 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-23 14:15:00 | 341.25 | 345.46 | 346.74 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-04-27 09:15:00 | 336.80 | 333.41 | 338.07 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-04-27 09:15:00 | 336.80 | 333.41 | 338.07 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 09:15:00 | 336.80 | 333.41 | 338.07 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 09:45:00 | 338.00 | 333.41 | 338.07 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 10:15:00 | 338.30 | 334.39 | 338.09 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-04-27 10:45:00 | 338.85 | 334.39 | 338.09 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-04-27 11:15:00 | 338.35 | 335.18 | 338.12 | EMA400 retest candle locked (from downside) |

### Cycle 147 — BUY (started 2026-04-29 09:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-04-29 09:15:00 | 356.35 | 341.92 | 340.09 | EMA200 above EMA400 |

### Cycle 148 — SELL (started 2026-04-30 11:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (SELL) | 2026-04-30 11:15:00 | 338.50 | 341.68 | 341.70 | EMA200 below EMA400 |
| First Alert — break + close below crossover candle low | 2026-04-30 13:15:00 | 336.40 | 340.16 | 340.98 | Break + close below crossover candle low |
| Second Alert (Retest 1) — EMA200 retest from below | 2026-05-04 11:15:00 | 341.00 | 338.12 | 339.37 | EMA200 retest candle locked (from downside) |
| Retest1 invalidated — EMA400 touched before ENTRY1 | 2026-05-04 11:15:00 | 341.00 | 338.12 | 339.37 | EMA400 touched before retest1 break — omit ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 11:15:00 | 341.00 | 338.12 | 339.37 | EMA400 retest candle locked (from downside) |
| ALERT3_SIDEWAYS | 2026-05-04 11:45:00 | 342.80 | 338.12 | 339.37 | Sideways (15m bar) within 4 candles — back to Stage 4 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-04 12:15:00 | 340.15 | 338.53 | 339.44 | EMA400 retest candle locked (from downside) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 13:15:00 | 337.55 | 338.53 | 339.44 | SELL ENTRY2 attempt 1/4 (v1.4 edge cross+close) |
| Second Entry (SELL) — retest2 break (cap 3 attempts) | 2026-05-04 14:30:00 | 338.55 | 339.62 | 339.80 | SELL ENTRY2 attempt 2/4 (v1.4 edge cross+close) |
| Stop hit — per-position SL triggered | 2026-05-04 15:15:00 | 346.00 | 340.90 | 340.36 | Force close (CROSSOVER_FLIP) qty=1.00 alert=retest2 |

### Cycle 149 — BUY (started 2026-05-04 15:15:00)

| Stage | Time | Price | EMA200 | EMA400 | Note |
|-------|------|-------|--------|--------|------|
| Trend Identification (BUY) | 2026-05-04 15:15:00 | 346.00 | 340.90 | 340.36 | EMA200 above EMA400 |
| First Alert — break + close above crossover candle high | 2026-05-06 15:15:00 | 348.85 | 345.60 | 344.13 | Break + close above crossover candle high |
| Second Alert (Retest 1) — EMA200 retest from above | 2026-05-08 09:15:00 | 353.20 | 353.38 | 349.96 | EMA200 retest candle locked (from upside) |
| ALERT2_SIDEWAYS | 2026-05-08 09:30:00 | 352.00 | 353.38 | 349.96 | Sideways (15m bar) within 4 candles — skip ENTRY1 |
| Third Alert (Retest 2) — price touches/crosses EMA400 | 2026-05-08 14:15:00 | 351.90 | 353.86 | 351.64 | EMA400 retest candle locked (from upside) |
| ALERT3_SIDEWAYS | 2026-05-08 15:00:00 | 351.90 | 353.86 | 351.64 | Sideways (15m bar) within 4 candles — back to Stage 4 |


## Closed Legs

| Trend | Alert | Entry Time | Entry | Exit Time | Exit | Reason | Qty | % |
|-------|-------|-----------|-------|-----------|------|--------|-----|---|
| BUY | retest2 | 2024-05-22 09:15:00 | 600.70 | 2024-05-29 11:15:00 | 614.00 | STOP_HIT | 1.00 | 2.21% |
| BUY | retest2 | 2024-05-22 09:45:00 | 602.25 | 2024-05-29 11:15:00 | 614.00 | STOP_HIT | 1.00 | 1.95% |
| BUY | retest2 | 2024-05-24 09:15:00 | 605.35 | 2024-05-29 11:15:00 | 614.00 | STOP_HIT | 1.00 | 1.43% |
| SELL | retest2 | 2024-06-03 11:00:00 | 578.00 | 2024-06-05 09:15:00 | 549.10 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 13:00:00 | 578.45 | 2024-06-05 09:15:00 | 549.53 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-06-03 11:00:00 | 578.00 | 2024-06-05 14:15:00 | 550.50 | STOP_HIT | 0.50 | 4.76% |
| SELL | retest2 | 2024-06-03 13:00:00 | 578.45 | 2024-06-05 14:15:00 | 550.50 | STOP_HIT | 0.50 | 4.83% |
| SELL | retest2 | 2024-06-04 09:15:00 | 567.35 | 2024-06-06 15:15:00 | 574.50 | STOP_HIT | 1.00 | -1.26% |
| BUY | retest2 | 2024-06-07 14:15:00 | 576.40 | 2024-06-13 11:15:00 | 597.00 | STOP_HIT | 1.00 | 3.57% |
| BUY | retest2 | 2024-06-21 09:15:00 | 652.25 | 2024-06-24 09:15:00 | 628.00 | STOP_HIT | 1.00 | -3.72% |
| BUY | retest2 | 2024-06-21 10:15:00 | 652.00 | 2024-06-24 09:15:00 | 628.00 | STOP_HIT | 1.00 | -3.68% |
| BUY | retest2 | 2024-06-21 11:00:00 | 652.85 | 2024-06-24 09:15:00 | 628.00 | STOP_HIT | 1.00 | -3.81% |
| BUY | retest2 | 2024-06-21 11:30:00 | 652.05 | 2024-06-24 09:15:00 | 628.00 | STOP_HIT | 1.00 | -3.69% |
| BUY | retest1 | 2024-07-29 09:15:00 | 739.05 | 2024-07-30 13:15:00 | 723.70 | STOP_HIT | 1.00 | -2.08% |
| BUY | retest2 | 2024-07-29 15:15:00 | 736.00 | 2024-07-30 13:15:00 | 723.70 | STOP_HIT | 1.00 | -1.67% |
| BUY | retest2 | 2024-07-30 10:15:00 | 734.00 | 2024-07-30 13:15:00 | 723.70 | STOP_HIT | 1.00 | -1.40% |
| SELL | retest2 | 2024-08-06 11:30:00 | 683.60 | 2024-08-08 11:15:00 | 707.05 | STOP_HIT | 1.00 | -3.43% |
| SELL | retest2 | 2024-08-08 09:45:00 | 677.35 | 2024-08-08 11:15:00 | 707.05 | STOP_HIT | 1.00 | -4.38% |
| SELL | retest2 | 2024-08-08 15:00:00 | 679.70 | 2024-08-09 09:15:00 | 699.35 | STOP_HIT | 1.00 | -2.89% |
| SELL | retest2 | 2024-08-14 09:15:00 | 674.10 | 2024-08-19 13:15:00 | 682.40 | STOP_HIT | 1.00 | -1.23% |
| SELL | retest2 | 2024-08-14 14:00:00 | 676.00 | 2024-08-19 13:15:00 | 682.40 | STOP_HIT | 1.00 | -0.95% |
| SELL | retest2 | 2024-08-16 11:00:00 | 680.30 | 2024-08-19 13:15:00 | 682.40 | STOP_HIT | 1.00 | -0.31% |
| SELL | retest2 | 2024-08-16 13:15:00 | 681.50 | 2024-08-19 13:15:00 | 682.40 | STOP_HIT | 1.00 | -0.13% |
| SELL | retest2 | 2024-08-16 14:30:00 | 680.50 | 2024-08-19 13:15:00 | 682.40 | STOP_HIT | 1.00 | -0.28% |
| SELL | retest2 | 2024-08-19 10:15:00 | 679.25 | 2024-08-19 13:15:00 | 682.40 | STOP_HIT | 1.00 | -0.46% |
| SELL | retest2 | 2024-08-28 10:30:00 | 687.00 | 2024-09-04 09:15:00 | 652.65 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-08-28 10:30:00 | 687.00 | 2024-09-04 09:15:00 | 663.00 | STOP_HIT | 0.50 | 3.49% |
| SELL | retest2 | 2024-10-01 11:00:00 | 577.20 | 2024-10-07 09:15:00 | 548.34 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 13:00:00 | 574.00 | 2024-10-07 09:15:00 | 545.30 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-01 11:00:00 | 577.20 | 2024-10-07 13:15:00 | 519.48 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-01 13:00:00 | 574.00 | 2024-10-07 14:15:00 | 516.60 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2024-10-21 12:45:00 | 521.85 | 2024-10-22 12:15:00 | 495.76 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 14:45:00 | 520.55 | 2024-10-22 13:15:00 | 494.52 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-10-21 12:45:00 | 521.85 | 2024-10-24 11:15:00 | 489.05 | STOP_HIT | 0.50 | 6.29% |
| SELL | retest2 | 2024-10-21 14:45:00 | 520.55 | 2024-10-24 11:15:00 | 489.05 | STOP_HIT | 0.50 | 6.05% |
| BUY | retest2 | 2024-11-07 09:15:00 | 499.60 | 2024-11-07 11:15:00 | 549.56 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-11-14 13:45:00 | 508.55 | 2024-11-18 09:15:00 | 483.12 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-14 14:30:00 | 510.00 | 2024-11-18 09:15:00 | 484.50 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-14 15:15:00 | 509.80 | 2024-11-18 09:15:00 | 484.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-11-14 13:45:00 | 508.55 | 2024-11-18 10:15:00 | 521.35 | STOP_HIT | 0.50 | -2.52% |
| SELL | retest2 | 2024-11-14 14:30:00 | 510.00 | 2024-11-18 10:15:00 | 521.35 | STOP_HIT | 0.50 | -2.23% |
| SELL | retest2 | 2024-11-14 15:15:00 | 509.80 | 2024-11-18 10:15:00 | 521.35 | STOP_HIT | 0.50 | -2.27% |
| SELL | retest2 | 2024-11-18 10:15:00 | 508.65 | 2024-11-18 12:15:00 | 550.60 | STOP_HIT | 1.00 | -8.25% |
| BUY | retest2 | 2024-11-29 11:15:00 | 617.40 | 2024-12-04 11:15:00 | 679.14 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-11-29 13:00:00 | 615.60 | 2024-12-04 11:15:00 | 677.16 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2024-12-02 09:30:00 | 621.50 | 2024-12-04 11:15:00 | 683.65 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2024-12-20 13:30:00 | 758.35 | 2024-12-26 09:15:00 | 720.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2024-12-20 13:30:00 | 758.35 | 2024-12-27 12:15:00 | 732.00 | STOP_HIT | 0.50 | 3.47% |
| SELL | retest2 | 2025-01-08 09:15:00 | 710.85 | 2025-01-09 13:15:00 | 675.31 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-08 09:15:00 | 710.85 | 2025-01-13 09:15:00 | 639.76 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-01-24 13:30:00 | 600.50 | 2025-01-27 09:15:00 | 540.45 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest2 | 2025-01-24 14:30:00 | 598.60 | 2025-01-27 09:15:00 | 568.67 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-01-24 14:30:00 | 598.60 | 2025-01-27 10:15:00 | 538.74 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-07 12:45:00 | 539.00 | 2025-02-11 09:15:00 | 512.05 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 13:45:00 | 538.05 | 2025-02-11 09:15:00 | 511.15 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 14:45:00 | 538.55 | 2025-02-11 09:15:00 | 511.62 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 527.60 | 2025-02-11 09:15:00 | 501.22 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-07 12:45:00 | 539.00 | 2025-02-11 12:15:00 | 485.10 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-07 13:45:00 | 538.05 | 2025-02-11 12:15:00 | 484.24 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-07 14:45:00 | 538.55 | 2025-02-11 12:15:00 | 484.69 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-10 09:15:00 | 527.60 | 2025-02-12 09:15:00 | 474.84 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-02-19 13:15:00 | 430.80 | 2025-02-20 10:15:00 | 443.60 | STOP_HIT | 1.00 | -2.97% |
| SELL | retest2 | 2025-02-25 11:45:00 | 425.75 | 2025-02-28 09:15:00 | 404.46 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-27 09:15:00 | 423.20 | 2025-02-28 09:15:00 | 402.04 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-02-25 11:45:00 | 425.75 | 2025-02-28 14:15:00 | 408.35 | STOP_HIT | 0.50 | 4.09% |
| SELL | retest2 | 2025-02-27 09:15:00 | 423.20 | 2025-02-28 14:15:00 | 408.35 | STOP_HIT | 0.50 | 3.51% |
| BUY | retest2 | 2025-03-21 10:15:00 | 441.85 | 2025-03-24 13:15:00 | 486.04 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-21 12:45:00 | 439.40 | 2025-03-24 13:15:00 | 483.34 | TARGET_HIT | 1.00 | 10.00% |
| BUY | retest2 | 2025-03-21 14:15:00 | 441.95 | 2025-03-24 13:15:00 | 486.15 | TARGET_HIT | 1.00 | 10.00% |
| SELL | retest1 | 2025-04-09 09:30:00 | 389.45 | 2025-04-11 09:15:00 | 397.95 | STOP_HIT | 1.00 | -2.18% |
| SELL | retest2 | 2025-04-11 15:00:00 | 396.85 | 2025-04-15 09:15:00 | 413.45 | STOP_HIT | 1.00 | -4.18% |
| BUY | retest2 | 2025-04-21 10:15:00 | 442.00 | 2025-04-23 14:15:00 | 435.00 | STOP_HIT | 1.00 | -1.58% |
| BUY | retest2 | 2025-04-23 12:00:00 | 437.80 | 2025-04-23 14:15:00 | 435.00 | STOP_HIT | 1.00 | -0.64% |
| BUY | retest2 | 2025-04-23 12:45:00 | 438.40 | 2025-04-23 14:15:00 | 435.00 | STOP_HIT | 1.00 | -0.78% |
| BUY | retest2 | 2025-04-23 13:15:00 | 438.00 | 2025-04-23 14:15:00 | 435.00 | STOP_HIT | 1.00 | -0.68% |
| SELL | retest2 | 2025-04-29 10:15:00 | 417.30 | 2025-05-05 09:15:00 | 422.30 | STOP_HIT | 1.00 | -1.20% |
| SELL | retest2 | 2025-04-29 11:45:00 | 415.60 | 2025-05-05 10:15:00 | 418.35 | STOP_HIT | 1.00 | -0.66% |
| SELL | retest2 | 2025-04-29 13:00:00 | 417.95 | 2025-05-05 10:15:00 | 418.35 | STOP_HIT | 1.00 | -0.10% |
| SELL | retest2 | 2025-04-29 13:45:00 | 417.90 | 2025-05-05 10:15:00 | 418.35 | STOP_HIT | 1.00 | -0.11% |
| SELL | retest2 | 2025-05-02 15:15:00 | 410.00 | 2025-05-05 10:15:00 | 418.35 | STOP_HIT | 1.00 | -2.04% |
| BUY | retest1 | 2025-05-15 09:15:00 | 438.00 | 2025-05-16 14:15:00 | 459.90 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-15 13:45:00 | 435.90 | 2025-05-16 14:15:00 | 457.69 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-15 14:15:00 | 435.80 | 2025-05-16 14:15:00 | 457.59 | PARTIAL | 0.50 | 5.00% |
| BUY | retest1 | 2025-05-15 09:15:00 | 438.00 | 2025-05-19 13:15:00 | 450.15 | STOP_HIT | 0.50 | 2.77% |
| BUY | retest1 | 2025-05-15 13:45:00 | 435.90 | 2025-05-19 13:15:00 | 450.15 | STOP_HIT | 0.50 | 3.27% |
| BUY | retest1 | 2025-05-15 14:15:00 | 435.80 | 2025-05-19 13:15:00 | 450.15 | STOP_HIT | 0.50 | 3.29% |
| SELL | retest2 | 2025-05-22 12:45:00 | 429.05 | 2025-05-26 10:15:00 | 439.85 | STOP_HIT | 1.00 | -2.52% |
| SELL | retest2 | 2025-05-22 13:30:00 | 428.80 | 2025-05-26 10:15:00 | 439.85 | STOP_HIT | 1.00 | -2.58% |
| SELL | retest2 | 2025-06-20 15:15:00 | 418.00 | 2025-06-23 09:15:00 | 426.45 | STOP_HIT | 1.00 | -2.02% |
| BUY | retest2 | 2025-06-26 14:30:00 | 445.65 | 2025-07-01 09:15:00 | 441.50 | STOP_HIT | 1.00 | -0.93% |
| SELL | retest2 | 2025-07-03 12:15:00 | 441.40 | 2025-07-04 09:15:00 | 449.25 | STOP_HIT | 1.00 | -1.78% |
| SELL | retest2 | 2025-07-03 14:15:00 | 441.65 | 2025-07-04 09:15:00 | 449.25 | STOP_HIT | 1.00 | -1.72% |
| SELL | retest2 | 2025-07-03 15:00:00 | 441.95 | 2025-07-04 09:15:00 | 449.25 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-07-21 10:30:00 | 516.15 | 2025-07-24 12:15:00 | 510.30 | STOP_HIT | 1.00 | -1.13% |
| BUY | retest2 | 2025-07-22 14:15:00 | 514.50 | 2025-07-25 09:15:00 | 490.55 | STOP_HIT | 1.00 | -4.66% |
| BUY | retest2 | 2025-07-23 11:00:00 | 514.50 | 2025-07-25 09:15:00 | 490.55 | STOP_HIT | 1.00 | -4.66% |
| BUY | retest2 | 2025-07-23 11:45:00 | 512.75 | 2025-07-25 09:15:00 | 490.55 | STOP_HIT | 1.00 | -4.33% |
| BUY | retest2 | 2025-07-24 09:15:00 | 515.00 | 2025-07-25 09:15:00 | 490.55 | STOP_HIT | 1.00 | -4.75% |
| SELL | retest2 | 2025-07-30 10:30:00 | 478.35 | 2025-08-01 11:15:00 | 454.43 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2025-07-30 13:15:00 | 476.85 | 2025-08-01 11:15:00 | 454.81 | PARTIAL | 0.50 | 4.62% |
| SELL | retest2 | 2025-07-30 15:00:00 | 478.75 | 2025-08-01 12:15:00 | 453.01 | PARTIAL | 0.50 | 5.38% |
| SELL | retest2 | 2025-07-30 10:30:00 | 478.35 | 2025-08-04 12:15:00 | 430.52 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-30 13:15:00 | 476.85 | 2025-08-04 12:15:00 | 429.17 | TARGET_HIT | 0.50 | 10.00% |
| SELL | retest2 | 2025-07-30 15:00:00 | 478.75 | 2025-08-04 12:15:00 | 430.88 | TARGET_HIT | 0.50 | 10.00% |
| BUY | retest2 | 2025-08-13 09:45:00 | 427.15 | 2025-08-18 11:15:00 | 425.60 | STOP_HIT | 1.00 | -0.36% |
| BUY | retest2 | 2025-08-13 10:30:00 | 430.20 | 2025-08-18 11:15:00 | 425.60 | STOP_HIT | 1.00 | -1.07% |
| BUY | retest2 | 2025-08-18 09:30:00 | 427.15 | 2025-08-18 11:15:00 | 425.60 | STOP_HIT | 1.00 | -0.36% |
| SELL | retest2 | 2025-10-15 14:00:00 | 431.70 | 2025-10-23 09:15:00 | 434.25 | STOP_HIT | 1.00 | -0.59% |
| SELL | retest2 | 2025-10-16 10:45:00 | 431.65 | 2025-10-23 09:15:00 | 434.25 | STOP_HIT | 1.00 | -0.60% |
| SELL | retest2 | 2025-10-16 11:30:00 | 431.50 | 2025-10-23 09:15:00 | 434.25 | STOP_HIT | 1.00 | -0.64% |
| SELL | retest2 | 2025-10-17 10:45:00 | 431.35 | 2025-10-23 09:15:00 | 434.25 | STOP_HIT | 1.00 | -0.67% |
| BUY | retest2 | 2025-11-03 09:15:00 | 475.60 | 2025-11-04 09:15:00 | 466.05 | STOP_HIT | 1.00 | -2.01% |
| BUY | retest2 | 2025-11-03 12:45:00 | 473.85 | 2025-11-04 09:15:00 | 466.05 | STOP_HIT | 1.00 | -1.65% |
| BUY | retest2 | 2025-11-03 14:15:00 | 472.75 | 2025-11-04 09:15:00 | 466.05 | STOP_HIT | 1.00 | -1.42% |
| BUY | retest2 | 2025-11-03 15:15:00 | 473.95 | 2025-11-04 09:15:00 | 466.05 | STOP_HIT | 1.00 | -1.67% |
| SELL | retest2 | 2025-11-27 13:15:00 | 440.45 | 2025-12-01 09:15:00 | 448.05 | STOP_HIT | 1.00 | -1.73% |
| SELL | retest2 | 2025-11-27 15:00:00 | 438.80 | 2025-12-01 09:15:00 | 448.05 | STOP_HIT | 1.00 | -2.11% |
| SELL | retest2 | 2025-11-28 10:00:00 | 440.15 | 2025-12-01 09:15:00 | 448.05 | STOP_HIT | 1.00 | -1.79% |
| SELL | retest2 | 2025-12-04 12:45:00 | 445.40 | 2025-12-04 14:15:00 | 454.40 | STOP_HIT | 1.00 | -2.02% |
| SELL | retest2 | 2025-12-09 09:15:00 | 437.30 | 2025-12-09 11:15:00 | 447.45 | STOP_HIT | 1.00 | -2.32% |
| SELL | retest2 | 2025-12-09 13:30:00 | 443.80 | 2025-12-09 14:15:00 | 458.00 | STOP_HIT | 1.00 | -3.20% |
| SELL | retest2 | 2025-12-09 14:15:00 | 444.55 | 2025-12-09 14:15:00 | 458.00 | STOP_HIT | 1.00 | -3.03% |
| BUY | retest2 | 2025-12-15 12:45:00 | 463.60 | 2025-12-16 12:15:00 | 452.85 | STOP_HIT | 1.00 | -2.32% |
| BUY | retest2 | 2025-12-16 14:00:00 | 460.05 | 2025-12-23 10:15:00 | 466.05 | STOP_HIT | 1.00 | 1.30% |
| BUY | retest2 | 2025-12-17 09:30:00 | 460.05 | 2025-12-23 10:15:00 | 466.05 | STOP_HIT | 1.00 | 1.30% |
| SELL | retest2 | 2025-12-26 09:30:00 | 459.40 | 2025-12-26 11:15:00 | 466.50 | STOP_HIT | 1.00 | -1.55% |
| SELL | retest2 | 2026-01-14 10:30:00 | 418.90 | 2026-01-16 11:15:00 | 437.00 | STOP_HIT | 1.00 | -4.32% |
| SELL | retest2 | 2026-01-14 11:00:00 | 418.75 | 2026-01-16 11:15:00 | 437.00 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest2 | 2026-01-22 10:15:00 | 403.70 | 2026-01-23 09:15:00 | 421.30 | STOP_HIT | 1.00 | -4.36% |
| SELL | retest2 | 2026-01-22 10:45:00 | 403.10 | 2026-01-23 09:15:00 | 421.30 | STOP_HIT | 1.00 | -4.52% |
| BUY | retest2 | 2026-01-29 14:45:00 | 422.80 | 2026-02-01 12:15:00 | 412.40 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2026-01-30 09:30:00 | 422.80 | 2026-02-01 12:15:00 | 412.40 | STOP_HIT | 1.00 | -2.46% |
| BUY | retest2 | 2026-02-01 10:15:00 | 422.80 | 2026-02-01 12:15:00 | 412.40 | STOP_HIT | 1.00 | -2.46% |
| SELL | retest2 | 2026-02-11 09:15:00 | 414.05 | 2026-02-11 14:15:00 | 425.00 | STOP_HIT | 1.00 | -2.64% |
| SELL | retest2 | 2026-02-11 12:45:00 | 413.80 | 2026-02-11 14:15:00 | 425.00 | STOP_HIT | 1.00 | -2.71% |
| SELL | retest2 | 2026-02-19 11:30:00 | 405.05 | 2026-02-23 13:15:00 | 384.80 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 13:30:00 | 404.65 | 2026-02-23 13:15:00 | 384.42 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-23 09:15:00 | 405.20 | 2026-02-23 13:15:00 | 384.94 | PARTIAL | 0.50 | 5.00% |
| SELL | retest2 | 2026-02-19 11:30:00 | 405.05 | 2026-02-24 14:15:00 | 383.65 | STOP_HIT | 0.50 | 5.28% |
| SELL | retest2 | 2026-02-19 13:30:00 | 404.65 | 2026-02-24 14:15:00 | 383.65 | STOP_HIT | 0.50 | 5.19% |
| SELL | retest2 | 2026-02-23 09:15:00 | 405.20 | 2026-02-24 14:15:00 | 383.65 | STOP_HIT | 0.50 | 5.32% |
| BUY | retest2 | 2026-04-08 09:15:00 | 330.45 | 2026-04-13 09:15:00 | 316.65 | STOP_HIT | 1.00 | -4.18% |
| BUY | retest2 | 2026-04-21 09:30:00 | 352.10 | 2026-04-22 09:15:00 | 345.60 | STOP_HIT | 1.00 | -1.85% |
| SELL | retest2 | 2026-05-04 13:15:00 | 337.55 | 2026-05-04 15:15:00 | 346.00 | STOP_HIT | 1.00 | -2.50% |
| SELL | retest2 | 2026-05-04 14:30:00 | 338.55 | 2026-05-04 15:15:00 | 346.00 | STOP_HIT | 1.00 | -2.20% |
